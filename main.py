import os
import uuid
import time
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
from ultralytics import YOLO
import torch

# HOW TO RUN:
# 1. pip install -r requirements.txt
# 2. uvicorn main.py:app --reload
# 3. open index.html in your browser

app = FastAPI()

# Enable CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load model once at startup
try:
    model_path = "best.pt"
    if not os.path.exists(model_path) and os.path.exists("Model/best.pt"):
        model_path = "Model/best.pt"
    
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Classes: {0: "smoke", 1: "fire"}
# OpenCV uses BGR: purple = (128, 0, 128), orange = (0, 165, 255)
COLORS = {
    0: (128, 0, 128),   # Purple (Smoke)
    1: (0, 165, 255)    # Orange (Fire)
}
CLASS_NAMES = {
    0: "smoke",
    1: "fire"
}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Ensure best.pt is present.")

    # Generate unique filename
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    if ext.lower() not in ['.mp4', '.avi', '.mov']:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload mp4, avi, or mov.")

    input_path = os.path.join("uploads", f"{file_id}{ext}")
    output_path = os.path.join("outputs", f"{file_id}_out.mp4")

    # Save uploaded video
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Process video
    try:
        start_time = time.time()
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Reassemble frames into output video (mp4, h264 codec)
        # Using 'avc1' for h264 which is widely supported in browsers
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            # conf threshold: 0.25, iou: 0.45
            device = 0 if torch.cuda.is_available() else 'cpu'
            results = model.predict(frame, conf=0.25, iou=0.45, verbose=False, device=device)
            
            # Draw boxes
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    label = f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}"
                    color = COLORS.get(cls, (0, 255, 0)) # Default green if unknown
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame)
            frames_processed += 1

        cap.release()
        out.release()
        
        # In case 'avc1' fails silently in some OpenCV builds (creating a 0KB file), 
        # check file size as a basic safeguard
        if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
            raise Exception("Output video is 0 bytes. OpenCV might be missing H264 codec support.")

        inference_time = time.time() - start_time
        
        return {
            "message": "Processing complete",
            "download_url": f"/download/{file_id}_out.mp4",
            "frames_processed": frames_processed,
            "inference_time_seconds": round(inference_time, 2)
        }
    except Exception as e:
        # Cleanup partial files if possible
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")

@app.get("/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join("outputs", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="File not found")
