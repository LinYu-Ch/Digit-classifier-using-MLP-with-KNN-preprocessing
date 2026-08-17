"""FastAPI server: serves the drawing frontend and classifies submitted images.

    python app.py --port 8000 --model ./mnist_cnn.pt

Endpoints:
    GET  /            -> the drawing page
    POST /predict     -> multipart form field "file", or JSON {"image": "data:image/png;base64,..."}
    GET  /health      -> which backend is loaded
"""

from __future__ import annotations

import argparse
import base64
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import model_loader
from preprocess import preprocess, to_png_bytes

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = model_loader.load_model(app.state.model_path, app.state.device)
    print(f"[ok] ConvNetwork loaded on {backend}.")
    yield


app = FastAPI(title="MNIST digit recognizer", lifespan=lifespan)
app.state.model_path = None
app.state.device = None


async def _read_image(request: Request, file: UploadFile | None) -> bytes:
    """Accept either a multipart upload or a base64 data URL in JSON."""
    if file is not None:
        data = await file.read()
    else:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(400, "Send a multipart 'file' field or JSON with 'image'.")
        raw = payload.get("image", "")
        if "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except Exception:
            raise HTTPException(400, "The 'image' field is not valid base64.")

    if not data:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 5 MB limit.")
    return data


@app.post("/predict")
async def predict(request: Request, file: UploadFile | None = File(default=None)):
    image_bytes = await _read_image(request, file)

    try:
        tensor = preprocess(image_bytes)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        raise HTTPException(415, "That file could not be read as an image.")

    probs = model_loader.predict_probs(tensor)
    probs = [float(p) for p in probs]
    prediction = max(range(len(probs)), key=probs.__getitem__)

    return {
        "prediction": prediction,
        "confidence": probs[prediction],
        "probabilities": probs,
        "backend": model_loader.backend_name(),
        # The exact 28x28 the model was given, so the UI can show it.
        "processed_png": "data:image/png;base64,"
        + base64.b64encode(to_png_bytes(tensor)).decode(),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "backend": model_loader.backend_name()}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="cnn_mnist_weights.pth", help="ConvNetwork state_dict")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # app.state is lost when --reload re-imports the module, so pass these through
    # the environment too; the loader falls back to reading them.
    os.environ["MODEL_PATH"] = args.model
    if args.device:
        os.environ["MODEL_DEVICE"] = args.device
    app.state.model_path = args.model
    app.state.device = args.device
    uvicorn.run(
        "app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
