# YOLOv8 Object Detection in Python

This project provides a clean, easy-to-use Python implementation for object detection using Ultralytics' **YOLOv8** (You Only Look Once v8). It supports running object detection on:
- Static images
- Video files
- Directory of images/videos
- Live webcam streams

## Features

- **Multi-source support**: Seamlessly switch between images, videos, and live webcam feeds.
- **Model selection**: Choose from different YOLOv8 model sizes (nano, small, medium, large, x-large) depending on your accuracy and speed requirements.
- **Auto-download**: The script automatically downloads pretrained weights from Ultralytics if they aren't present locally.
- **Output saving**: Detection results are saved automatically with bounding boxes and confidence scores.

---

## Getting Started

### 1. Prerequisites

Make sure you have Python 3.8 or higher installed on your system.

### 2. Installation

Clone or download this repository, navigate to the folder, and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Usage

Run the detection script `detect.py` with the `--source` argument.

#### A. Run on an Image
```bash
python detect.py --source path/to/your/image.jpg
```

#### B. Run on a Video
```bash
python detect.py --source path/to/your/video.mp4
```

#### C. Run on a Live Webcam Feed
Use `'0'` (or the appropriate webcam device index) to run detection in real-time from your webcam:
```bash
python detect.py --source 0 --show
```
*Note: Press `q` while focusing on the output window to exit the webcam stream.*

#### D. Customizing the Model and Confidence
You can customize the model size (e.g., small `s`, medium `m`, large `l`, extra large `x`) and the confidence threshold:
```bash
python detect.py --source path/to/image.jpg --model yolov8m.pt --conf 0.4
```

---

## Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str` | *Required* | Path to image, video, directory, or webcam index (e.g., `0`) |
| `--model` | `str` | `yolov8n.pt` | YOLOv8 model file (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`) |
| `--conf` | `float` | `0.25` | Confidence threshold for filtering detections (range: `0.0` to `1.0`) |
| `--show` | `flag` | `False` | Display the detection window in real-time |
| `--no-save` | `flag` | `False` | Add this flag if you do *not* want to save results to disk |
| `--output-dir`| `str` | `runs` | Directory to save prediction results |

---

## Model Sizes

The model weight file (`yolov8*.pt`) will automatically download on its first run:
- **`yolov8n.pt`** (Nano): Fastest, lowest latency, smallest size, suitable for edge devices or CPU inference.
- **`yolov8s.pt`** (Small): Good balance of speed and accuracy.
- **`yolov8m.pt`** (Medium): Higher accuracy, slightly slower.
- **`yolov8l.pt`** (Large): High accuracy, recommended for GPU environments.
- **`yolov8x.pt`** (Extra Large): Maximum accuracy, slowest inference speed.

---

## Uploading to GitHub

If you want to upload this project to your own GitHub repository, follow these steps:

1. **Initialize Git** in your project folder:
   ```bash
   git init
   ```

2. **Add all files** (this will automatically ignore large weights like `yolov8n.pt` and output folder `runs/` due to the `.gitignore`):
   ```bash
   git add .
   ```

3. **Commit the files**:
   ```bash
   git commit -m "Initial commit: YOLOv8 object detection setup"
   ```

4. **Link and push** to your GitHub repository (replace the URL with your repository's URL):
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   git push -u origin main
   ```
