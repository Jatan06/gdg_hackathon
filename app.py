import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpointEmbeddings


# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# ==========================================
# 1. INITIALIZE DATA & MODELS
# ==========================================
# Load the hospital dataset into memory exactly ONCE when the server starts.
# Ensure you have a 'mock_hospitals.csv' in the same directory with columns: 
# ['name', 'latitude', 'longitude', 'capabilities']
try:
    df_hospitals = pd.read_csv("mock_datasets.csv")
    
    # --- THE FIX: Force lat/lon to be numbers, not strings ---
    df_hospitals['latitude'] = pd.to_numeric(df_hospitals['latitude'], errors='coerce')
    df_hospitals['longitude'] = pd.to_numeric(df_hospitals['longitude'], errors='coerce')
    
    # Drop any rows where the CSV had blank or corrupted coordinates
    df_hospitals = df_hospitals.dropna(subset=['latitude', 'longitude'])
    
    # Map columns to match expected schema
    if 'Hospital_Name' in df_hospitals.columns:
        df_hospitals = df_hospitals.rename(columns={'Hospital_Name': 'name'})
    
    # Combine Specialties and Facilities into capabilities for vector search
    if 'Specialties' in df_hospitals.columns and 'Facilities' in df_hospitals.columns:
        df_hospitals['capabilities'] = df_hospitals['Specialties'].fillna('') + ", " + df_hospitals['Facilities'].fillna('')
    elif 'capabilities' not in df_hospitals.columns:
        df_hospitals['capabilities'] = "General Hospital"
        
except FileNotFoundError:
    # Dummy data fallback so the server doesn't crash if the CSV is missing during testing
    print("Warning: mock_hospitals.csv not found. Using fallback dummy data.")
    df_hospitals = pd.DataFrame({
        "name": ["City General", "Apollo Highway Care", "Metro Burn Center"],
        "latitude": [23.0225, 23.5000, 23.0300],
        "longitude": [72.5714, 72.6000, 72.5800],
        "capabilities": ["basic trauma, bcls_ambulance", "oxygen_respirators, trauma_surgeon", "burn_unit, advanced_life_support_ambulance"]
    })

# Initialize the LLM for Triage and the Embedding model for FAISS
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=2)
embeddings_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)
# ==========================================
# 2. SCHEMAS (Input & Output)
# ==========================================
class EmergencyRequest(BaseModel):
    latitude: float = Field(..., description="GPS Latitude from the frontend device")
    longitude: float = Field(..., description="GPS Longitude from the frontend device")
    description: str = Field(..., description="The user's frantic text or transcribed voice report")

class LocationMetadata(BaseModel):
    latitude: float = Field(description="The GPS latitude.")
    longitude: float = Field(description="The GPS longitude.")
    venue_specifics: str = Field(description="Exact location details. If none, say 'Unknown'.")

class EmergencyPayload(BaseModel):
    crisis_category: Literal["MEDICAL", "FIRE", "SECURITY", "CHEMICAL", "STRUCTURAL", "NATURAL_DISASTER"]
    severity_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL", "MASS_CASUALTY"]
    location: LocationMetadata
    estimated_victims: int
    resource_vector: List[str] = Field(description="List of 1-5 specific medical/physical resources needed.")
    tts_summary: str = Field(description="A concise, urgent 2-sentence summary for a TTS robot.")

class HospitalMatch(BaseModel):
    hospital_name: str
    distance_km: float
    matched_capabilities: str
    latitude: float
    longitude: float

# New Output Schema: Combines the AI's triage with the Geospatial Match
class DispatchResponse(BaseModel):
    triage_analysis: EmergencyPayload
    dispatched_hospital: HospitalMatch

# ==========================================
# 3. LANGCHAIN TRIAGE SETUP
# ==========================================
structured_llm = llm.with_structured_output(EmergencyPayload)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert emergency dispatch AI for a hospitality crisis coordination system. "
               "Extract the incident details from the frantic user report and structure them perfectly. "
               "Use the provided GPS coordinates for the location data."),
    ("human", "Latitude: {lat}\nLongitude: {lon}\nReport: {user_report}")
])

triage_chain = prompt | structured_llm

# ==========================================
# 4. GEOSPATIAL & VECTOR SEARCH LOGIC
# ==========================================
def vectorized_haversine(lat1, lon1, lat2, lon2):
    """Pandas-optimized Haversine math."""
    R = 6371.0 # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def dispatch_best_hospital(user_lat, user_lon, required_resources: List[str]) -> dict:
    # 1. GEO-FIRST: Calculate distances instantly
    df_hospitals['distance_km'] = vectorized_haversine(
        user_lat, user_lon, df_hospitals['latitude'], df_hospitals['longitude']
    )
    
    # 2. FILTER: Take all hospitals strictly within a 25 km radius
    local_hospitals = df_hospitals[df_hospitals['distance_km'] <= 25.0]
    
    # EDGE CASE: User is stranded on a highway and NO hospitals are within 25km.
    if local_hospitals.empty:
        # Fallback: Just grab the 5 absolute closest ones, regardless of distance
        local_hospitals = df_hospitals.nsmallest(5, 'distance_km')

    # 3. VECTOR SECOND: Convert the local slice to LangChain Documents
    docs = []
    for _, row in local_hospitals.iterrows():
        docs.append(
            Document(
                page_content=row['capabilities'], 
                metadata={
                    "name": row['name'],
                    "distance": row['distance_km'],
                    "latitude": row['latitude'],
                    "longitude": row['longitude']
                }
            )
        )
    
    # 4. FAISS SEARCH: Spin up an instant memory DB and query it
    temp_vector_db = FAISS.from_documents(docs, embeddings_model)
    query_string = ", ".join(required_resources)
    
    best_match = temp_vector_db.similarity_search(query_string, k=1)[0]
    
    return {
        "hospital_name": best_match.metadata['name'],
        "distance_km": round(best_match.metadata['distance'], 2),
        "matched_capabilities": best_match.page_content,
        "latitude": best_match.metadata['latitude'],
        "longitude": best_match.metadata['longitude']
    }

# ==========================================
# 5. FASTAPI ROUTES
# ==========================================
app = FastAPI(title="Hospitality Triage & Dispatch API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/triage", response_model=DispatchResponse)
async def triage_and_dispatch(request: EmergencyRequest):
    """
    1. Analyzes the emergency using Gemini.
    2. Filters hospitals within a 25km radius.
    3. Uses FAISS to find the best-equipped hospital in that radius.
    """
    try:
        # STEP 1: AI Triage (Predict the needs)
        triage_result = triage_chain.invoke({
            "lat": request.latitude,
            "lon": request.longitude,
            "user_report": request.description
        })
        
        # STEP 2: Geo-Vector Dispatch (Find the hospital based on predicted needs)
        hospital_match = dispatch_best_hospital(
            user_lat=request.latitude,
            user_lon=request.longitude,
            required_resources=triage_result.resource_vector
        )
        
        # STEP 3: Return the combined payload to the frontend
        return {
            "triage_analysis": triage_result,
            "dispatched_hospital": hospital_match
        }
        
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dispatch Pipeline Failed: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "Active", "message": "API is running. Send POST to /api/v1/triage"}