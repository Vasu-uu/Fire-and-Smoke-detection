# YOLOv8 Smoke & Fire Detection Web App

A full-stack web application built with FastAPI and vanilla JavaScript to perform smoke and fire detection on uploaded videos using YOLOv8.

## Features
- Upload `.mp4`, `.avi`, or `.mov` videos via drag-and-drop.
- Frame-by-frame inference using a custom YOLOv8 model (`best.pt`).
- Highlights Smoke (Purple boxes) and Fire (Orange boxes).
- Automatically utilizes GPU (CUDA) for fast inference if available.
- Side-by-side original vs. processed video playback.
- Easily download the final processed video.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For GPU acceleration on Windows, you may need to install the CUDA-enabled PyTorch version explicitly: `pip uninstall -y torch torchvision` and then `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`)*

2. **Model File**
   Ensure your trained YOLOv8 model (`best.pt`) is placed inside the `Model/` directory.

3. **Start the Backend Server**
   ```bash
   uvicorn main.py:app --reload
   ```

4. **Launch the Frontend**
   Open `index.html` in any modern web browser.

## Directory Structure
- `main.py` - FastAPI backend.
- `index.html` - Vanilla JS & CSS frontend UI.
- `requirements.txt` - Python dependencies.
- `Model/best.pt` - The YOLOv8 model weights.
- `uploads/` - Stores uploaded videos (auto-created).
- `outputs/` - Stores processed videos (auto-created).
