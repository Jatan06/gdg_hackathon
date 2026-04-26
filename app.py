import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
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
    # 1. Load the file
    df_hospitals = pd.read_csv("mock_datasets.csv")
    
    # 2. Rename the column to match our code
    df_hospitals = df_hospitals.rename(columns={
        "Hospital_Name": "name"
    })
    
    # 3. Merge 'Specialties' and 'Facilities' into one giant 'capabilities' string for Gemini
    # We use fillna('') just in case a hospital left one of those columns blank
    df_hospitals['capabilities'] = df_hospitals['Specialties'].astype(str).fillna('') + ", " + df_hospitals['Facilities'].astype(str).fillna('')
    
    # 4. Force lat/lon to be pure numbers
    # Note: This will catch the word "Error" in row 1 (Holy Family) and turn it into NaN
    df_hospitals['latitude'] = pd.to_numeric(df_hospitals['latitude'], errors='coerce')
    df_hospitals['longitude'] = pd.to_numeric(df_hospitals['longitude'], errors='coerce')
    
    # 5. Drop any rows with broken GPS coordinates
    df_hospitals = df_hospitals.dropna(subset=['latitude', 'longitude'])

except FileNotFoundError:
    print("CRITICAL WARNING: mock_hospitals.csv not found in the directory! Using fallback dummy data.")
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
class HospitalDecision(BaseModel):
    hospital_name: str = Field(description="The exact name of the chosen hospital from the provided list.")
    reasoning: str = Field(description="A 1-sentence explanation of why this hospital's vague capabilities imply they can handle the needed resources.")

class HospitalMatch(BaseModel):
    hospital_name: str
    distance_km: float
    matched_capabilities: str
    ai_reasoning: str
# New Output Schema: Combines the AI's triage with the Geospatial Match
class DispatchResponse(BaseModel):
    triage_analysis: EmergencyPayload
    dispatched_hospital: HospitalMatch
# ==========================================
# 3. LANGCHAIN TRIAGE SETUP
# ==========================================
# --- NEW: MATCHMAKER CHAIN ---
matchmaker_llm = llm.with_structured_output(HospitalDecision)

matchmaker_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert medical logistics AI. "
               "You are given a list of Required Resources for an emergency, and a list of the 5 closest hospitals with brief, vague capabilities. "
               "Using your deep medical knowledge, deduce which hospital is most likely equipped to handle the resources needed. "
               "Choose the closest one if multiple are equally capable."),
    ("human", "Required Resources: {resources}\n\nNearby Hospitals:\n{hospitals_list}")
])

matchmaker_chain = matchmaker_prompt | matchmaker_llm
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
    
    # 2. FILTER: Grab the 5 absolute closest hospitals
    closest_5 = df_hospitals.nsmallest(5, 'distance_km')
    
    # 3. FORMAT FOR THE LLM: Turn the Pandas rows into a clean string
    hospitals_text = ""
    for index, row in closest_5.iterrows():
        hospitals_text += f"- Name: {row['name']} | Distance: {row['distance_km']:.2f} km | Stated Capabilities: {row['capabilities']}\n"
    
    # 4. AGENTIC MATCHMAKING: Ask Gemini to figure out the best fit
    decision = matchmaker_chain.invoke({
        "resources": ", ".join(required_resources),
        "hospitals_list": hospitals_text
    })
    
    # 5. Extract the final distance and capabilities for the chosen hospital
    chosen_row = closest_5[closest_5['name'] == decision.hospital_name].iloc[0]
    
    return {
        "hospital_name": decision.hospital_name,
        "distance_km": round(chosen_row['distance_km'], 2),
        "matched_capabilities": chosen_row['capabilities'],
        "ai_reasoning": decision.reasoning
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