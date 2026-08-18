import { BrowserASLModel } from "./model.mjs";

const $ = (id) => document.getElementById(id);
const camera = $("camera");
const stillImage = $("still-image");
const overlay = $("hand-overlay");
const overlayContext = overlay.getContext("2d");
const modelCanvas = $("model-input");
const modelContext = modelCanvas.getContext("2d", { willReadFrequently: true });
const state = {
  model: null, stream: null, source: stillImage, timer: null, running: false,
  objectUrl: null, handLandmarker: null, handBox: null, handLoad: null,
};

function setStatus(message) { $("prediction-status").textContent = message; }

function cropFor(source) {
  const width = source === camera ? camera.videoWidth : source.naturalWidth;
  const height = source === camera ? camera.videoHeight : source.naturalHeight;
  const zoom = Number($("crop-zoom").value);
  if (!width || !height) return null;
  let side = Math.min(width, height) / zoom;
  let centerX = width / 2;
  let centerY = height / 2;
  if (source === camera && $("auto-frame").checked && state.handBox) {
    const box = state.handBox;
    side = Math.min(Math.max(box.width, box.height) * 1.65, Math.min(width, height));
    centerX = box.x + box.width / 2;
    centerY = box.y + box.height / 2;
  }
  const x = Math.max(0, Math.min(width - side, centerX - side / 2));
  const y = Math.max(0, Math.min(height - side, centerY - side / 2));
  return { x, y, side };
}

function preprocess(source) {
  const size = state.model.size;
  const crop = cropFor(source);
  if (!crop) throw new Error("The selected image has no readable pixels yet.");
  const { mean, std } = state.model.manifest.normalization;
  modelContext.save();
  modelContext.clearRect(0, 0, size, size);
  if (source === camera && $("model-mirror").checked) {
    modelContext.translate(size, 0);
    modelContext.scale(-1, 1);
  }
  modelContext.drawImage(source, crop.x, crop.y, crop.side, crop.side, 0, 0, size, size);
  modelContext.restore();
  const pixels = modelContext.getImageData(0, 0, size, size).data;
  const input = new Float32Array(3 * size * size);
  for (let pixel = 0, byte = 0; pixel < size * size; pixel += 1, byte += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      input[channel * size * size + pixel] = (pixels[byte + channel] / 255 - mean[channel]) / std[channel];
    }
  }
  return input;
}

function renderPrediction(top) {
  const winner = top[0];
  $("prediction-title").textContent = winner.label;
  $("prediction-confidence").textContent = `${(winner.probability * 100).toFixed(1)}% confidence · top of 26 classes`;
  $("probabilities").replaceChildren(...top.map(({ label, probability }) => {
    const item = document.createElement("li");
    item.className = "probability";
    const percentage = (probability * 100).toFixed(probability >= 0.1 ? 1 : 2);
    item.innerHTML = `<span class="probability__label">${label}</span><span class="probability__track"><span class="probability__fill" style="width:${Math.max(probability * 100, 0.4)}%"></span></span><span class="probability__value">${percentage}%</span>`;
    return item;
  }));
}

function drawHandGuide() {
  const box = state.handBox;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  if (!box || !camera.videoWidth || !camera.videoHeight) return;
  overlay.width = camera.videoWidth;
  overlay.height = camera.videoHeight;
  overlayContext.strokeStyle = "#d9ed53";
  overlayContext.lineWidth = Math.max(3, camera.videoWidth / 180);
  overlayContext.setLineDash([12, 9]);
  overlayContext.strokeRect(box.x, box.y, box.width, box.height);
  overlayContext.setLineDash([]);
}

async function ensureHandLandmarker() {
  if (state.handLandmarker) return state.handLandmarker;
  if (!state.handLoad) {
    state.handLoad = (async () => {
      const vision = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm");
      const fileset = await vision.FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm",
      );
      state.handLandmarker = await vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" },
        runningMode: "VIDEO", numHands: 1,
      });
      return state.handLandmarker;
    })();
  }
  return state.handLoad;
}

async function updateHandFrame() {
  if (!$("auto-frame").checked || !state.handLandmarker || camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  const result = state.handLandmarker.detectForVideo(camera, performance.now());
  const points = result.landmarks?.[0];
  if (!points?.length) { state.handBox = null; drawHandGuide(); return; }
  const xs = points.map((point) => point.x * camera.videoWidth);
  const ys = points.map((point) => point.y * camera.videoHeight);
  const minX = Math.max(0, Math.min(...xs));
  const maxX = Math.min(camera.videoWidth, Math.max(...xs));
  const minY = Math.max(0, Math.min(...ys));
  const maxY = Math.min(camera.videoHeight, Math.max(...ys));
  const padding = Math.max(maxX - minX, maxY - minY) * 0.18;
  state.handBox = { x: Math.max(0, minX - padding), y: Math.max(0, minY - padding), width: Math.min(camera.videoWidth, maxX + padding) - Math.max(0, minX - padding), height: Math.min(camera.videoHeight, maxY + padding) - Math.max(0, minY - padding) };
  drawHandGuide();
}

async function infer() {
  if (!state.model || !state.source || state.running) return;
  if (state.source === camera && camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  state.running = true;
  try {
    await updateHandFrame();
    renderPrediction(state.model.topK(state.model.predict(preprocess(state.source)), 3));
    setStatus(state.source === camera ? "Live camera prediction updates about once per second. Keep one hand inside the guide; try model mirroring if the orientation differs." : "Prediction computed locally from the selected image.");
  } catch (error) { console.error(error); setStatus(`Unable to classify this frame: ${error.message}`); } finally { state.running = false; }
}

function stopCamera() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null; state.handBox = null; drawHandGuide();
  camera.hidden = true; stillImage.hidden = false; $("camera-controls").hidden = true;
  $("start-camera").disabled = false; $("stop-camera").disabled = true;
  if (state.source === camera) { state.source = stillImage; $("input-source").textContent = "Selected image"; infer(); }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) { setStatus("This browser does not provide camera access. Upload an image instead."); return; }
  stopCamera();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    camera.srcObject = state.stream; await camera.play(); state.source = camera;
    stillImage.hidden = true; camera.hidden = false; $("camera-controls").hidden = false;
    $("input-source").textContent = "Live camera · local browser inference";
    $("start-camera").disabled = true; $("stop-camera").disabled = false;
    await infer(); state.timer = window.setInterval(infer, 850);
  } catch (error) { console.error(error); setStatus("Camera access was not available. Upload an image instead."); stopCamera(); }
}

function selectImage(source, description) {
  stopCamera();
  stillImage.onload = () => { state.source = stillImage; $("input-source").textContent = description; infer(); };
  stillImage.src = source;
}

function selectUpload(event) {
  const [file] = event.target.files; if (!file) return;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file); selectImage(state.objectUrl, `Uploaded image · ${file.name}`);
}

async function enableAutoFrame() {
  if (!$("auto-frame").checked) { state.handBox = null; drawHandGuide(); return; }
  setStatus("Loading free on-device hand framing…");
  try { await ensureHandLandmarker(); setStatus("Auto-frame is active. It locates a hand locally, then crops it for the unchanged classifier."); }
  catch (error) { console.error(error); $("auto-frame").checked = false; setStatus("Auto-frame could not load. Use the visible guide and crop zoom instead."); }
}

function buildSampleGrid() {
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const grid = $("sample-grid");
  for (const letter of letters) {
    const button = document.createElement("button"); button.type = "button"; button.className = "sample-button"; button.setAttribute("aria-pressed", "false"); button.setAttribute("aria-label", `Load external ${letter} sample`);
    button.innerHTML = `<img src="assets/samples/${letter}.jpg" alt=""><span>${letter}</span>`;
    button.addEventListener("click", () => { grid.querySelectorAll("button").forEach((item) => item.setAttribute("aria-pressed", "false")); button.setAttribute("aria-pressed", "true"); selectImage(`assets/samples/${letter}.jpg`, `External challenge sample · true label ${letter}`); });
    grid.append(button);
  }
}

async function boot() {
  buildSampleGrid();
  try {
    $("loader-status").textContent = "Fetching model manifest and 661 KB of released weights…";
    state.model = await BrowserASLModel.load(); $("model-state").textContent = "Browser model ready";
    if (stillImage.complete) await infer(); else stillImage.addEventListener("load", infer, { once: true });
    $("loader").hidden = true; $("page").setAttribute("aria-busy", "false"); setStatus("Prediction computed locally from the selected image.");
  } catch (error) { console.error(error); $("loader-status").textContent = `Could not load the browser model: ${error.message}`; $("model-state").textContent = "Model unavailable"; setStatus("The downloadable model could not be loaded. Evaluation evidence remains below."); }
}

$("start-camera").addEventListener("click", startCamera); $("stop-camera").addEventListener("click", stopCamera); $("image-upload").addEventListener("change", selectUpload);
$("preview-mirror").addEventListener("change", (event) => $("camera-layer").classList.toggle("is-mirrored", event.target.checked));
$("model-mirror").addEventListener("change", infer); $("auto-frame").addEventListener("change", enableAutoFrame);
$("crop-zoom").addEventListener("input", (event) => { $("crop-zoom-value").textContent = `${Number(event.target.value).toFixed(2)}×`; infer(); });
$("camera-layer").classList.add("is-mirrored"); window.addEventListener("pagehide", stopCamera); boot();
