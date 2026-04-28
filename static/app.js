// =============================================
// Emergency Dispatch — Main Application Logic
// =============================================

const API_URL = "/api/v1/triage"; // Simplified for relative path
let map,
  userMarker,
  markersGroup,
  routeLines = [];
let etaIntervals = [];
let recognition = null;

let currentFileBase64 = null;
let currentFileMimeType = null;

// =============================================
// 1. MAP INITIALIZATION
// =============================================
function initMap() {
  map = L.map("map", {
    center: [19.076, 72.8777],
    zoom: 12,
    zoomControl: false,
    attributionControl: false,
  });

  L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    {
      maxZoom: 19,
    },
  ).addTo(map);

  L.control.zoom({ position: "topright" }).addTo(map);

  markersGroup = L.layerGroup().addTo(map);
}

function createPulsingIcon(color) {
  return L.divIcon({
    className: "custom-marker",
    html: `<div style="
      width: 18px; height: 18px; border-radius: 50%;
      background: ${color};
      box-shadow: 0 0 0 0 ${color};
      animation: marker-pulse 2s infinite;
      border: 3px solid white;
    "></div>
    <style>
      @keyframes marker-pulse {
        0% { box-shadow: 0 0 0 0 ${color}80; }
        70% { box-shadow: 0 0 0 15px ${color}00; }
        100% { box-shadow: 0 0 0 0 ${color}00; }
      }
    </style>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
    popupAnchor: [0, -12],
  });
}

function createLabelIcon(emoji, color) {
  return L.divIcon({
    className: "custom-marker",
    html: `<div style="
      width: 36px; height: 36px; border-radius: 8px;
      background: ${color}; display: flex;
      align-items: center; justify-content: center;
      font-size: 18px; border: 2px solid white;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    ">${emoji}</div>`,
    iconSize: [36, 36],
    iconAnchor: [18, 18],
    popupAnchor: [0, -22],
  });
}

function setUserMarker(lat, lng) {
  if (userMarker) map.removeLayer(userMarker);
  userMarker = L.marker([lat, lng], { icon: createPulsingIcon("#3b82f6") })
    .addTo(map)
    .bindPopup("<strong>📍 Your Location</strong>")
    .openPopup();
  map.setView([lat, lng], 13);
}

// =============================================
// 2. GEOLOCATION
// =============================================
function detectLocation() {
  const statusEl = document.getElementById("gpsStatus");
  const latInput = document.getElementById("lat");
  const lonInput = document.getElementById("lon");

  if (!navigator.geolocation) {
    statusEl.innerText = "❌ Not Supported";
    statusEl.className = "gps-status denied";
    return;
  }

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const { latitude, longitude } = pos.coords;
      latInput.value = latitude.toFixed(6);
      lonInput.value = longitude.toFixed(6);
      statusEl.innerText = "✅ Locked";
      statusEl.className = "gps-status locked";
      setUserMarker(latitude, longitude);
    },
    (err) => {
      statusEl.innerText = "❌ Denied";
      statusEl.className = "gps-status denied";
    }
  );
}

// =============================================
// 3. FILE UPLOAD HANDLING
// =============================================
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show preview
    const preview = document.getElementById('filePreview');
    const fileName = document.getElementById('fileName');
    fileName.innerText = `📎 ${file.name}`;
    preview.style.display = 'flex';

    // Convert to base64 for processing
    const reader = new FileReader();
    reader.onload = (e) => {
        currentFileBase64 = e.target.result.split(',')[1];
        currentFileMimeType = file.type;
    };
    reader.readAsDataURL(file);
}

function removeFile() {
    document.getElementById('fileUpload').value = '';
    document.getElementById('filePreview').style.display = 'none';
    currentFileBase64 = null;
    currentFileMimeType = null;
}

// =============================================
// 4. DISPATCH LOGIC
// =============================================
async function sendDispatch() {
  const btn = document.getElementById("dispatchBtn");
  const desc = document.getElementById("description").value.trim();
  const lat = parseFloat(document.getElementById("lat").value);
  const lon = parseFloat(document.getElementById("lon").value);

  if (!desc) return alert("Please describe the emergency.");

  btn.disabled = true;
  btn.classList.add("loading");

  try {
    // 1. Upload File first if exists
    let mediaUrl = null;
    const fileInput = document.getElementById('fileUpload');
    if (fileInput.files[0]) {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        const uploadRes = await fetch('/api/v1/upload', {
            method: 'POST',
            body: formData
        });
        const uploadData = await uploadRes.json();
        mediaUrl = uploadData.media_url;
    }

    // 2. Run Triage
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latitude: lat,
        longitude: lon,
        description: desc,
        media_url: mediaUrl
      }),
    });

    const data = await response.json();
    renderResults(data);
    plotUnits(data);
  } catch (err) {
    console.error(err);
    alert("System error. Please try again.");
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
  }
}

function renderResults(data) {
  const area = document.getElementById("resultsArea");
  const t = data.triage_analysis;
  const u = data.dispatched_units;

  let unitsHtml = "";

  // Medical
  if (u.medical && u.medical.status !== "Not Required") {
    unitsHtml += createUnitCard("medical", u.medical);
  }
  // Fire
  if (u.fire && u.fire.status !== "Not Required") {
    unitsHtml += createUnitCard("fire", u.fire);
  }
  // Police
  if (u.police && u.police.status !== "Not Required") {
    unitsHtml += createUnitCard("police", u.police);
  }

  area.innerHTML = `
    <div class="triage-card">
      <div class="triage-header">
        <h3>🚨 Incident Analysis</h3>
        <span class="severity-badge severity-${t.severity_level}">${t.severity_level}</span>
      </div>
      <div class="triage-body">
        <div class="triage-stat">
          <div class="label">Category</div>
          <div class="value">${t.crisis_category}</div>
        </div>
        <div class="triage-stat">
          <div class="label">Est. Victims</div>
          <div class="value">${t.estimated_victims}</div>
        </div>
        <div class="triage-summary">
          <div class="label">AI Summary (TTS)</div>
          <div class="value">${t.tts_summary}</div>
        </div>
      </div>
    </div>
    <div class="dispatch-cards">
      ${unitsHtml || '<div class="not-required-label">No active units required for this level of threat.</div>'}
    </div>
  `;
}

function createUnitCard(type, data) {
  const icons = { medical: "🚑", fire: "🚒", police: "🚓" };
  const names = { medical: "Ambulance", fire: "Fire Engine", police: "Police Unit" };
  const unitName = data.hospital_name || data.unit_name || "Unknown Unit";

  return `
    <div class="dispatch-card ${type}">
      <div class="card-header">
        <div class="unit-info">
          <div class="unit-icon ${type}-icon">${icons[type]}</div>
          <div>
            <div class="unit-name">${unitName}</div>
            <div class="unit-type">${names[type]}</div>
          </div>
        </div>
        <div class="eta-badge ${type}-eta">
          <div class="eta-time" id="eta-${data.id || type}">${data.estimated_eta_minutes}m</div>
          <div class="eta-label">ETA</div>
        </div>
      </div>
      <div class="card-details">
        <div class="card-detail">
          <div class="label">Distance</div>
          <div class="value">${data.distance_km} km</div>
        </div>
        <div class="card-detail">
          <div class="label">Status</div>
          <div class="value" style="color:var(--accent-green)">En Route</div>
        </div>
      </div>
      <div class="card-reasoning">
        <div class="label">AI Dispatch Reasoning</div>
        <div class="value">${data.ai_reasoning}</div>
      </div>
      <button class="call-btn">📞 Integrated Comms Offline</button>
    </div>
  `;
}

function plotUnits(data) {
  markersGroup.clearLayers();
  routeLines.forEach((l) => map.removeLayer(l));
  routeLines = [];

  const u = data.dispatched_units;
  const userLat = parseFloat(document.getElementById("lat").value);
  const userLon = parseFloat(document.getElementById("lon").value);

  const colors = { medical: "#dc2626", fire: "#ea580c", police: "#2563eb" };
  const icons = { medical: "🏥", fire: "🔥", police: "🚓" };

  Object.entries(u).forEach(([type, unit]) => {
    if (unit && unit.status !== "Not Required") {
      const lat = unit.latitude;
      const lon = unit.longitude;

      // Add marker
      L.marker([lat, lon], { icon: createLabelIcon(icons[type], colors[type]) })
        .addTo(markersGroup)
        .bindPopup(`<strong>${unit.hospital_name || unit.unit_name}</strong><br>Dispatching ${type} unit...`);

      // Add line
      const line = L.polyline([[lat, lon], [userLat, userLon]], {
        color: colors[type],
        weight: 3,
        dashArray: "10, 10",
        opacity: 0.6,
      }).addTo(map);
      routeLines.push(line);
    }
  });

  // Zoom map to fit
  const bounds = L.latLngBounds([userLat, userLon]);
  markersGroup.eachLayer((l) => bounds.extend(l.getLatLng()));
  map.fitBounds(bounds, { padding: [50, 50] });
}

// =============================================
// 5. VOICE RECOGNITION
// =============================================
function toggleMic() {
    const btn = document.getElementById('micBtn');
    
    if (recognition && btn.classList.contains('recording')) {
        recognition.stop();
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return alert("Speech recognition not supported in this browser.");

    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
        btn.classList.add('recording');
        btn.innerText = '🔴';
    };

    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        document.getElementById('description').value = transcript;
    };

    recognition.onend = () => {
        btn.classList.remove('recording');
        btn.innerText = '🎙️';
    };

    recognition.onerror = () => {
        btn.classList.remove('recording');
        btn.innerText = '🎙️';
    };

    recognition.start();
}

// =============================================
// 6. INIT
// =============================================
window.onload = () => {
  initMap();
  detectLocation();
};
