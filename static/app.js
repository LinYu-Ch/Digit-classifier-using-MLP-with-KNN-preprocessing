const pad = document.getElementById("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
const hint = document.getElementById("hint");
const digitOut = document.getElementById("digit");
const confOut = document.getElementById("confidence");
const seenImg = document.getElementById("seen");
const barsList = document.getElementById("bars");
const statusOut = document.getElementById("status");
const backendOut = document.getElementById("backend");
const predictBtn = document.getElementById("predict");
const clearBtn = document.getElementById("clear");

/* ---------- canvas ---------- */

function resetCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, pad.width, pad.height);
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 22;          // ~ MNIST stroke weight at this canvas size
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

let drawing = false;
let dirty = false;
let lastPos = null;

// Map a pointer event to canvas pixel coordinates, accounting for CSS scaling.
function positionOf(event) {
  const box = pad.getBoundingClientRect();
  return {
    x: (event.clientX - box.left) * (pad.width / box.width),
    y: (event.clientY - box.top) * (pad.height / box.height),
  };
}

function startStroke(event) {
  drawing = true;
  dirty = true;
  hint.hidden = true;
  lastPos = positionOf(event);
  // A dot, so a single tap still leaves ink.
  ctx.beginPath();
  ctx.arc(lastPos.x, lastPos.y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#000000";
  ctx.fill();
  pad.setPointerCapture(event.pointerId);
}

function extendStroke(event) {
  if (!drawing) return;
  const pos = positionOf(event);
  ctx.beginPath();
  ctx.moveTo(lastPos.x, lastPos.y);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  lastPos = pos;
}

function endStroke() {
  drawing = false;
  lastPos = null;
}

pad.addEventListener("pointerdown", startStroke);
pad.addEventListener("pointermove", extendStroke);
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);
pad.addEventListener("pointerleave", endStroke);

/* ---------- readout ---------- */

function buildBars() {
  barsList.innerHTML = "";
  for (let d = 0; d <= 9; d++) {
    const row = document.createElement("li");
    row.innerHTML =
      `<span class="key">${d}</span>` +
      `<span class="track"><span class="fill" data-fill="${d}"></span></span>` +
      `<span class="val" data-val="${d}">0.0%</span>`;
    barsList.appendChild(row);
  }
}

function renderBars(probs, top) {
  probs.forEach((p, d) => {
    barsList.children[d].dataset.top = String(d === top);
    barsList.querySelector(`[data-fill="${d}"]`).style.width = `${(p * 100).toFixed(1)}%`;
    barsList.querySelector(`[data-val="${d}"]`).textContent = `${(p * 100).toFixed(1)}%`;
  });
}

function clearReadout() {
  digitOut.textContent = "–";
  digitOut.dataset.empty = "true";
  confOut.textContent = "awaiting input";
  seenImg.removeAttribute("src");
  statusOut.textContent = "";
  renderBars(new Array(10).fill(0), -1);
}

/* ---------- actions ---------- */

clearBtn.addEventListener("click", () => {
  resetCanvas();
  dirty = false;
  hint.hidden = false;
  clearReadout();
});

predictBtn.addEventListener("click", async () => {
  if (!dirty) {
    statusOut.textContent = "Draw a digit first.";
    return;
  }

  predictBtn.disabled = true;
  statusOut.textContent = "";
  confOut.textContent = "reading…";

  try {
    const blob = await new Promise((resolve) => pad.toBlob(resolve, "image/png"));
    const form = new FormData();
    form.append("file", blob, "digit.png");

    const response = await fetch("/predict", { method: "POST", body: form });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Server returned ${response.status}.`);
    }

    const result = await response.json();
    digitOut.textContent = result.prediction;
    digitOut.dataset.empty = "false";
    confOut.textContent = `${(result.confidence * 100).toFixed(1)}% confident`;
    seenImg.src = result.processed_png;
    renderBars(result.probabilities, result.prediction);
  } catch (error) {
    confOut.textContent = "no reading";
    statusOut.textContent = error.message;
  } finally {
    predictBtn.disabled = false;
  }
});

// Keyboard shortcuts: Enter predicts, Escape clears.
document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "BUTTON") return;
  if (event.key === "Enter") predictBtn.click();
  if (event.key === "Escape") clearBtn.click();
});

/* ---------- boot ---------- */

buildBars();
resetCanvas();
clearReadout();

fetch("/health")
  .then((r) => r.json())
  .then((info) => {
    backendOut.textContent = `ConvNetwork · ${info.backend}`;
  })
  .catch(() => {
    backendOut.dataset.state = "stub";
    backendOut.textContent = "Server unreachable";
  });
