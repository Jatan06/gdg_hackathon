import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv
import uuid
import os
import json
from datetime import datetime
import sqlite3
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
load_dotenv()
# ==========================================
# 0. DATABASE INITIALIZATION
# ==========================================
DB_PATH = "dispatch_v2.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id TEXT,
            incident_type TEXT,
            description TEXT,
            latitude REAL,
            longitude REAL,
            media_url TEXT,
            assigned_agency TEXT,
            assigned_station_name TEXT,
            status TEXT,
            timestamp TEXT,
            triage_json TEXT,
            dispatch_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 1. INITIALIZE DATA & MODELS
# ==========================================

# Global history for the dashboard (keeping for legacy compatibility, but using DB now)
incident_history = []

# --- LOAD HOSPITALS ---
try:
    df_hospitals = pd.read_csv("hospital_directory_cleaned.csv", low_memory=False)
    df_hospitals = df_hospitals.rename(columns={"Hospital_Name": "name"})
    # Ensure capabilities column exists or merge from specialties/facilities
    if 'capabilities' not in df_hospitals.columns:
        df_hospitals['capabilities'] = df_hospitals['Specialties'].astype(str).fillna('') + ", " + df_hospitals['Facilities'].astype(str).fillna('')
    
    df_hospitals['latitude'] = pd.to_numeric(df_hospitals['latitude'], errors='coerce')
    df_hospitals['longitude'] = pd.to_numeric(df_hospitals['longitude'], errors='coerce')
    df_hospitals = df_hospitals.dropna(subset=['latitude', 'longitude'])
except Exception as e:
    print(f"Warning: hospital_directory_cleaned.csv load failed ({e}). Using fallback.")
    df_hospitals = pd.DataFrame({
        "name": ["City General", "Apollo Highway Care"],
        "latitude": [23.0225, 23.5000],
        "longitude": [72.5714, 72.6000],
        "capabilities": ["basic trauma", "trauma_surgeon"]
    })

# --- LOAD FIRE STATIONS ---
try:
    df_fire = pd.read_csv("OpenStreetMap_-_Fire_Station.csv")
    # OSM CSV has latitude and longitude swapped or just needs naming? 
    # Let's check the first line again: latitude,longitude,osm_id,name...
    # But wait, looking at the data: 78.53, 21.59 ... 78 is longitude for India. 
    # So the CSV header might be 'latitude,longitude' but the values are 'lon,lat'.
    # I will swap them if they look like lon,lat.
    if df_fire['latitude'].mean() > 60: # Likely longitude
        df_fire = df_fire.rename(columns={'latitude': 'temp_lat', 'longitude': 'temp_lon'})
        df_fire = df_fire.rename(columns={'temp_lat': 'longitude', 'temp_lon': 'latitude'})
    
    # Fill empty names
    df_fire['name'] = df_fire['name'].replace(r'^\s*$', np.nan, regex=True)
    df_fire['name'] = df_fire['name'].fillna("Fire Station " + df_fire['district'].astype(str))
    
    df_fire['latitude'] = pd.to_numeric(df_fire['latitude'], errors='coerce')
    df_fire['longitude'] = pd.to_numeric(df_fire['longitude'], errors='coerce')
    df_fire = df_fire.dropna(subset=['latitude', 'longitude'])
    df_fire['capabilities'] = "fire_suppression, rescue_operations, basic_medical"
except Exception as e:
    print(f"Warning: OpenStreetMap_-_Fire_Station.csv load failed ({e}). Using fallback.")
    df_fire = pd.DataFrame({
        "name": ["Central Fire Command"],
        "latitude": [23.0200],
        "longitude": [72.5700],
        "capabilities": ["standard_engine"]
    })

# --- LOAD POLICE STATIONS ---
try:
    df_police = pd.read_csv("INDIA_POLICE_STATIONS.csv")
    df_police['latitude'] = pd.to_numeric(df_police['latitude'], errors='coerce')
    df_police['longitude'] = pd.to_numeric(df_police['longitude'], errors='coerce')
    df_police = df_police.dropna(subset=['latitude', 'longitude'])
    df_police['capabilities'] = "patrol_cars, basic_response"
except Exception as e:
    print(f"Warning: INDIA_POLICE_STATIONS.csv load failed ({e}). Using fallback.")
    df_police = pd.DataFrame({
        "name": ["Precinct 1 HQ"],
        "latitude": [23.0250],
        "longitude": [72.5750],
        "capabilities": ["patrol_cars"]
    })
# This replaces your standard Pydantic request passing
class DispatchState(TypedDict):
    # Inputs
    latitude: float
    longitude: float
    user_report: str
    
    # AI Triage Output
    triage_result: Optional[dict]
    
    # Parallel Dispatch Outputs
    medical_dispatch: Optional[dict]
    fire_dispatch: Optional[dict]
    police_dispatch: Optional[dict]
# Initialize the LLM for Triage and the Embedding model for FAISS
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0
)# embeddings_model = HuggingFaceEndpointEmbeddings(
#     model="sentence-transformers/all-MiniLM-L6-v2",
#     task="feature-extraction",
#     huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
# )
# ==========================================
# 2. SCHEMAS (Input & Output)
# ==========================================
class EmergencyRequest(BaseModel):
    latitude: float = Field(..., description="GPS Latitude from the frontend device")
    longitude: float = Field(..., description="GPS Longitude from the frontend device")
    description: str = Field(..., description="The user's frantic text or transcribed voice report")
    media_url: Optional[str] = None
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

class LoginRequest(BaseModel):
    agency_type: Literal["Police", "Fire", "Hospital"]
    station_name: str

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
class ResponderDecision(BaseModel):
    unit_name: str = Field(description="The exact name of the chosen responding unit from the provided list.")
    reasoning: str = Field(description="A 1-sentence explanation of why this unit's vague capabilities imply they can handle the needed resources.")

matchmaker_llm = llm.with_structured_output(ResponderDecision)

# Notice the prompt is now generalized for ANY type of first responder
matchmaker_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert emergency logistics AI. "
               "You are given a list of Required Resources for an emergency, and a list of the 5 closest responder units (Police, Fire, or Medical) with brief, vague capabilities. "
               "Using your logical reasoning, deduce which unit is most likely equipped to handle the resources needed. "
               "Choose the closest one if multiple are equally capable."),
    ("human", "Required Resources: {resources}\n\nNearby Units:\n{units_list}")
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
        "units_list": hospitals_text # <-- FIXED THIS KEY
    })
    chosen_row = closest_5[closest_5['name'] == decision.unit_name].iloc[0]
    
    eta_minutes = max(1, round(chosen_row['distance_km'] / 40 * 60))
    return {
        "hospital_name": decision.unit_name,
        "latitude": float(chosen_row['latitude']),
        "longitude": float(chosen_row['longitude']),
        "distance_km": round(chosen_row['distance_km'], 2),
        "estimated_eta_minutes": eta_minutes,
        "matched_capabilities": chosen_row['capabilities'],
        "ai_reasoning": decision.reasoning
    }
def dispatch_best_fire_station(user_lat, user_lon, required_resources: List[str]) -> dict:
    # 1. GEO-FIRST Math
    df_fire['distance_km'] = vectorized_haversine(
        user_lat, user_lon, df_fire['latitude'], df_fire['longitude']
    )
    
    # 2. Get 5 closest
    closest_5 = df_fire.nsmallest(5, 'distance_km')
    
    # 3. Format for LLM
    stations_text = ""
    for index, row in closest_5.iterrows():
        stations_text += f"- Name: {row['name']} | Distance: {row['distance_km']:.2f} km | Stated Capabilities: {row['capabilities']}\n"
    
    # 4. Agentic Decision
    decision = matchmaker_chain.invoke({
        "resources": ", ".join(required_resources),
        "units_list": stations_text
    })
    
    # 5. Extract output
    chosen_row = closest_5[closest_5['name'] == decision.unit_name].iloc[0]
    
    eta_minutes = max(1, round(chosen_row['distance_km'] / 40 * 60))
    return {
        "unit_name": decision.unit_name,
        "latitude": float(chosen_row['latitude']),
        "longitude": float(chosen_row['longitude']),
        "distance_km": round(chosen_row['distance_km'], 2),
        "estimated_eta_minutes": eta_minutes,
        "matched_capabilities": chosen_row['capabilities'],
        "ai_reasoning": decision.reasoning
    }

def dispatch_best_police_station(user_lat, user_lon, required_resources: List[str]) -> dict:
    # 1. GEO-FIRST Math
    df_police['distance_km'] = vectorized_haversine(
        user_lat, user_lon, df_police['latitude'], df_police['longitude']
    )
    
    # 2. Get 5 closest
    closest_5 = df_police.nsmallest(5, 'distance_km')
    
    # 3. Format for LLM
    stations_text = ""
    for index, row in closest_5.iterrows():
        stations_text += f"- Name: {row['name']} | Distance: {row['distance_km']:.2f} km | Stated Capabilities: {row['capabilities']}\n"
    
    # 4. Agentic Decision
    decision = matchmaker_chain.invoke({
        "resources": ", ".join(required_resources),
        "units_list": stations_text
    })
    
    # 5. Extract output
    chosen_row = closest_5[closest_5['name'] == decision.unit_name].iloc[0]
    
    eta_minutes = max(1, round(chosen_row['distance_km'] / 40 * 60))
    return {
        "unit_name": decision.unit_name,
        "latitude": float(chosen_row['latitude']),
        "longitude": float(chosen_row['longitude']),
        "distance_km": round(chosen_row['distance_km'], 2),
        "estimated_eta_minutes": eta_minutes,
        "matched_capabilities": chosen_row['capabilities'],
        "ai_reasoning": decision.reasoning
    }
# ==========================================
# 5. FASTAPI ROUTES
# ==========================================

# Create uploads directory if not exists
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app = FastAPI(title="Hospitality Triage & Dispatch API", version="1.0")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- NODE 1: The Master Triage ---
def triage_node(state: DispatchState):
    print("Running Master Triage...")
    result = triage_chain.invoke({
        "lat": state["latitude"],
        "lon": state["longitude"],
        "user_report": state["user_report"]
    })
    # Updates only the triage_result key
    return {"triage_result": result}

# --- NODE 2: Medical Dispatch (Runs in Parallel) ---
def medical_dispatch_node(state: DispatchState):
    triage = state["triage_result"]
    keywords = ["ambulance", "medical", "doctor", "hospital", "injury", "burn"]
    # Optimization: Only run if medical resources are actually needed
    if not any(kw in r.lower() for r in triage.resource_vector for kw in keywords):
        return {"medical_dispatch": {"status": "Not Required"}}
        
    print("Dispatching Medical...")
    # This is your existing Pandas + Matchmaker function!
    match = dispatch_best_hospital(state["latitude"], state["longitude"], triage.resource_vector)
    return {"medical_dispatch": match}

# --- NODE 3: Fire Dispatch (Runs in Parallel) ---
def fire_dispatch_node(state: DispatchState):
    triage = state["triage_result"]
    if triage.crisis_category != "FIRE" and "fire" not in "".join(triage.resource_vector).lower():
         return {"fire_dispatch": {"status": "Not Required"}}
         
    print("Dispatching Fire...")
    # Husain's logic pointing to df_fire_stations
    match = dispatch_best_fire_station(state["latitude"], state["longitude"], triage.resource_vector)
    return {"fire_dispatch": match}

# --- NODE 4: Police Dispatch (Runs in Parallel) ---
def police_dispatch_node(state: DispatchState):
    triage = state["triage_result"]
    if triage.crisis_category not in ["SECURITY", "MASS_CASUALTY"]:
        return {"police_dispatch": {"status": "Not Required"}}
        
    print("Dispatching Police...")
    # Husain's logic pointing to df_police_stations
    match = dispatch_best_police_station(state["latitude"], state["longitude"], triage.resource_vector)
    return {"police_dispatch": match}
# Initialize the graph
workflow = StateGraph(DispatchState)

# Add all nodes to the graph
workflow.add_node("triage", triage_node)
workflow.add_node("medical", medical_dispatch_node)
workflow.add_node("fire", fire_dispatch_node)
workflow.add_node("police", police_dispatch_node)

# --- THE ROUTING LOGIC ---
# Start -> Triage
workflow.add_edge(START, "triage")

# FAN-OUT: Triage triggers all three dispatchers simultaneously
workflow.add_edge("triage", "medical")
workflow.add_edge("triage", "fire")
workflow.add_edge("triage", "police")

# FAN-IN: All dispatchers must finish before the graph ends
workflow.add_edge("medical", END)
workflow.add_edge("fire", END)
workflow.add_edge("police", END)

# Compile it into a runnable app
try:
    dispatch_graph = workflow.compile()
except Exception as e:
    print(f"WARNING: LangGraph compile error: {e}")
    dispatch_graph = None
@app.post("/api/v1/triage")
async def triage_and_dispatch(request: EmergencyRequest):
    if dispatch_graph is None:
        raise HTTPException(status_code=503, detail="LangGraph workflow failed to compile. Check server logs.")
    try:
        initial_state = {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "user_report": request.description,
            "triage_result": None,
            "medical_dispatch": None,
            "fire_dispatch": None,
            "police_dispatch": None
        }

        final_state = dispatch_graph.invoke(initial_state)

        triage_result = final_state["triage_result"]
        # Convert Pydantic model to plain dict so it serializes correctly
        # when stored and re-read via /api/v1/incidents
        triage_dict = triage_result.model_dump() if hasattr(triage_result, 'model_dump') else dict(triage_result)

        response_payload = {
            "id": f"INC-{str(uuid.uuid4())[:8].upper()}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "triage_analysis": triage_dict,
            "dispatched_units": {
                "medical": final_state["medical_dispatch"],
                "fire": final_state["fire_dispatch"],
                "police": final_state["police_dispatch"]
            }
        }

        incident_history.append(response_payload)
        
        # --- SAVE TO DATABASE ---
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # We save the incident for each agency dispatched
            dispatched = response_payload["dispatched_units"]
            agencies = [
                ("Medical", dispatched["medical"]),
                ("Fire", dispatched["fire"]),
                ("Police", dispatched["police"])
            ]
            
            for agency_type, unit in agencies:
                if unit and unit.get("status") != "Not Required":
                    station_name = unit.get("unit_name") or unit.get("hospital_name") or unit.get("name") or "Unknown"
                    print(f"Saving incident {response_payload['id']} for {agency_type} at {station_name}")
                    cursor.execute('''
                        INSERT INTO incidents (id, incident_type, description, latitude, longitude, media_url, assigned_agency, assigned_station_name, status, timestamp, triage_json, dispatch_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        response_payload["id"],
                        response_payload["triage_analysis"]["crisis_category"],
                        request.description,
                        request.latitude,
                        request.longitude,
                        getattr(request, 'media_url', None),
                        agency_type,
                        station_name,
                        "ACTIVE",
                        response_payload["timestamp"],
                        json.dumps(response_payload["triage_analysis"]),
                        json.dumps(response_payload["dispatched_units"])
                    ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving to DB: {e}")

        return response_payload
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/login")
async def login(request: LoginRequest):
    input_name = request.station_name.strip().upper()
    actual_name = None
    
    if request.agency_type == "Police":
        if not df_police.empty:
            matches = df_police[df_police['name'].str.upper() == input_name]
            if not matches.empty:
                actual_name = matches.iloc[0]['name']
    elif request.agency_type == "Fire":
        if not df_fire.empty:
            matches = df_fire[df_fire['name'].str.upper() == input_name]
            if not matches.empty:
                actual_name = matches.iloc[0]['name']
    elif request.agency_type == "Hospital":
        if not df_hospitals.empty:
            matches = df_hospitals[df_hospitals['name'].str.upper() == input_name]
            if not matches.empty:
                actual_name = matches.iloc[0]['name']
            
    if not actual_name:
        raise HTTPException(status_code=401, detail=f"Station '{request.station_name}' not found in our {request.agency_type} registry.")
    
    # Get coordinates for the station
    lat, lon = 23.0225, 72.5714 # Default Ahmedabad
    if request.agency_type == "Police" and not df_police.empty:
        m = df_police[df_police['name'] == actual_name]
        if not m.empty: lat, lon = m.iloc[0]['latitude'], m.iloc[0]['longitude']
    elif request.agency_type == "Fire" and not df_fire.empty:
        m = df_fire[df_fire['name'] == actual_name]
        if not m.empty: lat, lon = m.iloc[0]['latitude'], m.iloc[0]['longitude']
    elif request.agency_type == "Hospital" and not df_hospitals.empty:
        m = df_hospitals[df_hospitals['name'] == actual_name]
        if not m.empty: lat, lon = m.iloc[0]['latitude'], m.iloc[0]['longitude']

    return {
        "status": "success", 
        "station_name": actual_name, 
        "agency_type": request.agency_type,
        "latitude": float(lat),
        "longitude": float(lon)
    }

@app.get("/api/v1/incidents/{station_name}")
async def get_station_incidents(station_name: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM incidents WHERE UPPER(assigned_station_name) = UPPER(?) ORDER BY timestamp DESC', (station_name,))
        rows = cursor.fetchall()
        conn.close()
        
        incidents = []
        for row in rows:
            inc = dict(row)
            # Reconstruct the expected frontend format
            incidents.append({
                "id": inc["id"],
                "timestamp": inc["timestamp"],
                "user_location": {"latitude": inc["latitude"], "longitude": inc["longitude"]},
                "media_url": inc["media_url"],
                "assigned_agency": inc["assigned_agency"],
                "assigned_station_name": inc["assigned_station_name"],
                "status": inc["status"],
                "triage_analysis": json.loads(inc["triage_json"]) if inc["triage_json"] else {},
                "dispatched_units": json.loads(inc["dispatch_json"]) if inc["dispatch_json"] else {}
            })
        return incidents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/incidents/{incident_id}/resolve")
async def resolve_incident(incident_id: str, station_name: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('UPDATE incidents SET status = "RESOLVED" WHERE id = ?', (incident_id,))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Incident {incident_id} resolved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/upload")
async def upload_media(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1]
        file_id = str(uuid.uuid4())[:8]
        filename = f"{file_id}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"media_url": f"http://127.0.0.1:8000/uploads/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def health_check():
    return {"status": "Active", "message": "API is running. Send POST to /api/v1/triage"}

@app.get("/api/v1/incidents")
def get_incidents():
    """Returns all processed incidents for the monitoring dashboard."""
    return {"incidents": incident_history}

@app.get("/api/v1/all_incidents")
async def get_all_incidents():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM incidents ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        
        incidents = []
        for row in rows:
            inc = dict(row)
            incidents.append({
                "id": inc["id"],
                "timestamp": inc["timestamp"],
                "user_location": {"latitude": inc["latitude"], "longitude": inc["longitude"]},
                "status": inc["status"],
                "triage_analysis": json.loads(inc["triage_json"]) if inc["triage_json"] else {},
            })
        return incidents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))