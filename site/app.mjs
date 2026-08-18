import { BrowserASLModel } from "./model.mjs";

const $ = (id) => document.getElementById(id);
const camera = $("camera");
const stillImage = $("still-image");
const modelCanvas = $("model-input");
const modelContext = modelCanvas.getContext("2d", { willReadFrequently: true });
const probabilities = $("probabilities");
const state = { model: null, stream: null, source: stillImage, timer: null, running: false, objectUrl: null };

function setStatus(message) {
  $("prediction-status").textContent = message;
}

function preprocess(source) {
  const size = state.model.size;
  const { mean, std } = state.model.manifest.normalization;
  modelContext.drawImage(source, 0, 0, size, size);
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
  probabilities.replaceChildren(...top.map(({ label, probability }) => {
    const item = document.createElement("li");
    item.className = "probability";
    const percentage = (probability * 100).toFixed(probability >= 0.1 ? 1 : 2);
    item.innerHTML = `<span class="probability__label">${label}</span><span class="probability__track"><span class="probability__fill" style="width:${Math.max(probability * 100, 0.4)}%"></span></span><span class="probability__value">${percentage}%</span>`;
    return item;
  }));
}

async function infer() {
  if (!state.model || !state.source || state.running) return;
  if (state.source === camera && camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  state.running = true;
  try {
    const top = state.model.topK(state.model.predict(preprocess(state.source)), 3);
    renderPrediction(top);
    setStatus(state.source === camera ? "Live camera prediction updates about once per second." : "Prediction computed locally from the selected image.");
  } catch (error) {
    console.error(error);
    setStatus(`Unable to classify this frame: ${error.message}`);
  } finally {
    state.running = false;
  }
}

function stopCamera() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  camera.hidden = true;
  stillImage.hidden = false;
  $("start-camera").disabled = false;
  $("stop-camera").disabled = true;
  if (state.source === camera) {
    state.source = stillImage;
    $("input-source").textContent = "Selected image";
    infer();
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("This browser does not provide camera access. Upload an image instead.");
    return;
  }
  stopCamera();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    camera.srcObject = state.stream;
    await camera.play();
    state.source = camera;
    stillImage.hidden = true;
    camera.hidden = false;
    $("input-source").textContent = "Live camera · local browser inference";
    $("start-camera").disabled = true;
    $("stop-camera").disabled = false;
    await infer();
    state.timer = window.setInterval(infer, 1000);
  } catch (error) {
    console.error(error);
    setStatus("Camera access was not available. Upload an image instead.");
    stopCamera();
  }
}

function selectUpload(event) {
  const [file] = event.target.files;
  if (!file) return;
  stopCamera();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  stillImage.onload = () => {
    state.source = stillImage;
    $("input-source").textContent = `Uploaded image · ${file.name}`;
    infer();
  };
  stillImage.src = state.objectUrl;
}

async function boot() {
  try {
    $("loader-status").textContent = "Fetching model manifest and 661 KB of released weights…";
    state.model = await BrowserASLModel.load();
    $("model-state").textContent = "Browser model ready";
    $("loader-status").textContent = "Running the default external sample…";
    if (stillImage.complete) await infer();
    else stillImage.addEventListener("load", infer, { once: true });
    $("loader").hidden = true;
    $("page").setAttribute("aria-busy", "false");
    setStatus("Prediction computed locally from the selected image.");
  } catch (error) {
    console.error(error);
    $("loader-status").textContent = `Could not load the browser model: ${error.message}`;
    $("model-state").textContent = "Model unavailable";
    setStatus("The downloadable model could not be loaded. The evaluation evidence remains available below.");
  }
}

$("start-camera").addEventListener("click", startCamera);
$("stop-camera").addEventListener("click", stopCamera);
$("image-upload").addEventListener("change", selectUpload);
window.addEventListener("pagehide", stopCamera);
boot();
