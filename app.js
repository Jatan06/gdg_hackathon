// =============================================
// Emergency Dispatch — Main Application Logic
// =============================================

const API_URL = "http://127.0.0.1:8000/api/v1/triage";
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
  L.control
    .attribution({ position: "bottomright", prefix: false })
    .addAttribution(
      '&copy; <a href="https://www.openstreetmap.org/">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    )
    .addTo(map);

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

  statusEl.textContent = "⏳ Detecting...";
  statusEl.className = "gps-status detecting";

  if (!navigator.geolocation) {
    statusEl.textContent = "❌ Not Supported";
    statusEl.className = "gps-status denied";
    return;
  }

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      latInput.value = pos.coords.latitude.toFixed(7);
      lonInput.value = pos.coords.longitude.toFixed(7);
      statusEl.textContent = "🟢 GPS Locked";
      statusEl.className = "gps-status locked";
      setUserMarker(pos.coords.latitude, pos.coords.longitude);
    },
    (err) => {
      console.warn("Geolocation error:", err.message);
      statusEl.textContent = "⚠️ Using Default";
      statusEl.className = "gps-status denied";
      setUserMarker(parseFloat(latInput.value), parseFloat(lonInput.value));
    },
    { enableHighAccuracy: true, timeout: 10000 },
  );
}

// =============================================
// 3. SPEECH-TO-TEXT
// =============================================
function initSpeechRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return;

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    const textarea = document.getElementById("description");
    let transcript = "";
    for (let i = 0; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    textarea.value = transcript;
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    stopRecording();
  };

  recognition.onend = () => {
    stopRecording();
  };
}

function toggleMic() {
  const micBtn = document.getElementById("micBtn");
  if (micBtn.classList.contains("recording")) {
    stopRecording();
  } else {
    startRecording();
  }
}

function startRecording() {
  if (!recognition) {
    alert("Speech recognition is not supported in this browser. Try Chrome.");
    return;
  }
  const micBtn = document.getElementById("micBtn");
  micBtn.classList.add("recording");
  micBtn.innerHTML = "⏹️";
  recognition.start();
}

function stopRecording() {
  if (!recognition) return;
  const micBtn = document.getElementById("micBtn");
  micBtn.classList.remove("recording");
  micBtn.innerHTML = "🎙️";
  try {
    recognition.stop();
  } catch (e) {}
}

// =============================================
// 4. DISPATCH REQUEST
// =============================================
async function sendDispatch() {
  const btn = document.getElementById("dispatchBtn");
  const description = document.getElementById("description").value.trim();

  if (!description) {
    shakeElement(document.getElementById("description"));
    return;
  }

  const lat = parseFloat(document.getElementById("lat").value);
  const lon = parseFloat(document.getElementById("lon").value);

  // UI: Loading state
  btn.classList.add("loading");
  btn.disabled = true;
  document.getElementById("resultsArea").innerHTML = `
    <div class="results-placeholder">
      <div class="spinner" style="width:40px;height:40px;border-width:3px;"></div>
      <p>AI is analyzing the crisis and computing optimal dispatch vectors...</p>
    </div>`;

  const payload = { latitude: lat, longitude: lon, description: description };
  if (currentFileBase64) {
    payload.file_data = currentFileBase64;
    payload.file_type = currentFileMimeType;
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error(`Server Error: ${response.status}`);
    const data = await response.json();
    renderResults(data, lat, lon);
  } catch (error) {
    document.getElementById("resultsArea").innerHTML = `
      <div class="results-placeholder" style="color: var(--accent-red);">
        <div class="icon">⚠️</div>
        <p><strong>Connection Failed</strong><br>${error.message}<br><br>
        Make sure FastAPI is running on<br><code>http://127.0.0.1:8000</code></p>
      </div>`;
  } finally {
    btn.classList.remove("loading");
    btn.disabled = false;
  }
}

function shakeElement(el) {
  el.style.animation = "none";
  el.offsetHeight; // trigger reflow
  el.style.animation = "shake 0.4s ease";
  el.style.borderColor = "var(--accent-red)";
  setTimeout(() => {
    el.style.borderColor = "";
    el.style.animation = "";
  }, 600);
}

// =============================================
// 5. RENDER RESULTS
// =============================================
function renderResults(data, userLat, userLon) {
  // Clear old intervals and markers
  etaIntervals.forEach(clearInterval);
  etaIntervals = [];
  markersGroup.clearLayers();
  routeLines.forEach((l) => map.removeLayer(l));
  routeLines = [];

  setUserMarker(userLat, userLon);

  const triage = data.triage_analysis;
  const units = data.dispatched_units;

  let html = "";

  // --- Triage Card ---
  html += buildTriageCard(triage);

  // --- Dispatch Cards ---
  html += '<div class="dispatch-cards">';
  html += buildDispatchCard(
    units.medical,
    "medical",
    "🏥",
    "Medical Response",
    userLat,
    userLon,
  );
  html += buildDispatchCard(
    units.fire,
    "fire",
    "🚒",
    "Fire Response",
    userLat,
    userLon,
  );
  html += buildDispatchCard(
    units.police,
    "police",
    "🚓",
    "Police Response",
    userLat,
    userLon,
  );
  html += "</div>";

  document.getElementById("resultsArea").innerHTML = html;

  // Start ETA countdowns
  startAllCountdowns();

  // Fit map to all markers
  const bounds = L.latLngBounds([[userLat, userLon]]);
  markersGroup.eachLayer((m) => bounds.extend(m.getLatLng()));
  if (userMarker) bounds.extend(userMarker.getLatLng());
  map.fitBounds(bounds, { padding: [50, 50] });
}

function buildTriageCard(triage) {
  return `
    <div class="triage-card">
      <div class="triage-header">
        <h3>🧠 AI Triage Analysis</h3>
        <span class="severity-badge severity-${triage.severity_level}">${triage.severity_level}</span>
      </div>
      <div class="triage-body">
        <div class="triage-stat">
          <div class="label">Crisis Type</div>
          <div class="value">${triage.crisis_category}</div>
        </div>
        <div class="triage-stat">
          <div class="label">Est. Victims</div>
          <div class="value">${triage.estimated_victims}</div>
        </div>
        <div class="triage-stat" style="grid-column: 1 / -1;">
          <div class="label">Resources Needed</div>
          <div class="value" style="font-size:0.78rem; font-weight:500; color:var(--text-secondary);">${triage.resource_vector.join(", ")}</div>
        </div>
        <div class="triage-summary">
          <div class="label">TTS Summary</div>
          <div class="value">${triage.tts_summary}</div>
        </div>
      </div>
    </div>`;
}

function buildDispatchCard(unit, type, emoji, label, userLat, userLon) {
  const isNotRequired = unit && unit.status === "Not Required";
  const cardClass = `dispatch-card ${type} ${isNotRequired ? "not-required" : ""}`;
  const iconClass = `unit-icon ${type}-icon`;

  const colorMap = { medical: "#ef4444", fire: "#f97316", police: "#3b82f6" };

  if (isNotRequired) {
    return `
      <div class="${cardClass}">
        <div class="card-header">
          <div class="unit-info">
            <div class="${iconClass}">${emoji}</div>
            <div><div class="unit-name">${label}</div><div class="unit-type">${type}</div></div>
          </div>
        </div>
        <div class="not-required-label">✅ Not Required for This Incident</div>
      </div>`;
  }

  if (!unit) return "";

  const unitName = unit.hospital_name || unit.unit_name || "Unknown";
  const eta = unit.estimated_eta_minutes || 0;
  const etaId = `eta-${type}`;

  // Add marker to map
  const marker = L.marker([unit.latitude, unit.longitude], {
    icon: createLabelIcon(emoji, colorMap[type]),
  }).bindPopup(`<strong>${unitName}</strong><br>
    📏 ${unit.distance_km} km away<br>
    ⏱️ ETA: ~${eta} min<br>
    🤖 ${unit.ai_reasoning}`);
  markersGroup.addLayer(marker);

  // Draw route line
  const line = L.polyline(
    [
      [userLat, userLon],
      [unit.latitude, unit.longitude],
    ],
    { color: colorMap[type], weight: 2, opacity: 0.6, dashArray: "8, 8" },
  ).addTo(map);
  routeLines.push(line);

  return `
    <div class="${cardClass}" style="animation-delay: ${type === "fire" ? "0.1s" : type === "police" ? "0.2s" : "0s"}">
      <div class="card-header">
        <div class="unit-info">
          <div class="${iconClass}">${emoji}</div>
          <div>
            <div class="unit-name">${unitName}</div>
            <div class="unit-type">${type}</div>
          </div>
        </div>
        <div class="eta-badge ${type}-eta">
          <div class="eta-time" id="${etaId}" data-seconds="${eta * 60}">${formatEta(eta * 60)}</div>
          <div class="eta-label">ETA</div>
        </div>
      </div>
      <div class="card-details">
        <div class="card-detail">
          <div class="label">Distance</div>
          <div class="value">${unit.distance_km} km</div>
        </div>
        <div class="card-detail">
          <div class="label">Coordinates</div>
          <div class="value">${unit.latitude.toFixed(4)}, ${unit.longitude.toFixed(4)}</div>
        </div>
      </div>
      <div class="card-reasoning">
        <div class="label">AI Reasoning</div>
        <div class="value">${unit.ai_reasoning}</div>
      </div>
      <button class="call-btn" disabled>📞 Call via Twilio (Coming Soon)</button>
    </div>`;
}

// =============================================
// 6. ETA COUNTDOWN TIMERS
// =============================================
function formatEta(totalSeconds) {
  if (totalSeconds <= 0) return "0:00";
  const m = Math.floor(totalSeconds / 60);
  const s = Math.floor(totalSeconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function startAllCountdowns() {
  ["medical", "fire", "police"].forEach((type) => {
    const el = document.getElementById(`eta-${type}`);
    if (!el) return;
    let seconds = parseInt(el.dataset.seconds) || 0;
    if (seconds <= 0) return;

    const interval = setInterval(() => {
      seconds--;
      if (seconds <= 0) {
        el.textContent = "ARRIVED";
        el.style.color = "var(--accent-green)";
        clearInterval(interval);
      } else {
        el.textContent = formatEta(seconds);
      }
    }, 1000);
    etaIntervals.push(interval);
  });
}

// =============================================
// 7. FILE HANDLING
// =============================================
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    const dataUrl = e.target.result;
    currentFileMimeType = dataUrl.split(";")[0].split(":")[1];
    currentFileBase64 = dataUrl.split(",")[1];

    document.getElementById("fileName").textContent = file.name;
    document.getElementById("filePreview").style.display = "flex";
  };
  reader.readAsDataURL(file);
}

function removeFile() {
  document.getElementById("fileUpload").value = "";
  currentFileBase64 = null;
  currentFileMimeType = null;
  document.getElementById("filePreview").style.display = "none";
}

// =============================================
// 8. INITIALIZATION
// =============================================
async function requestMicPermission() {
  try {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
    }
  } catch (err) {
    console.warn("Microphone permission denied:", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  detectLocation();
  initSpeechRecognition();
  requestMicPermission();
});
