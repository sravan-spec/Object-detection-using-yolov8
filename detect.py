import argparse
import os
import sys
import cv2
from ultralytics import YOLO

def run_detection(source, model_name="yolov8n.pt", confidence=0.25, show=False, save=True, output_dir="runs"):
    """
    Performs object detection using YOLOv8 on an image, video, or webcam stream.

    Args:
        source (str): Path to an image/video file, '0' (or int) for webcam, or folder.
        model_name (str): YOLOv8 model weight name (e.g., yolov8n.pt, yolov8s.pt).
        confidence (float): Confidence threshold (0.0 to 1.0) to filter weak detections.
        show (bool): If True, displays results in a window.
        save (bool): If True, saves output detections to disk.
        output_dir (str): Directory where output results will be saved.
    """
    print(f"Loading YOLOv8 model: {model_name}...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Make sure you are connected to the internet to auto-download the model, or specify a valid local path.")
        return

    # Check if the source is a webcam index
    if source.isdigit():
        source = int(source)
        # Webcams don't support pre-saving the entire stream directly via model() save flag in the same simple way,
        # we will handle webcam frame-by-frame if show or custom saving is needed, or just let ultralytics stream it.
        print(f"Starting webcam stream (index: {source}). Press 'q' to quit.")
        
        # We can use Ultralytics built-in predict stream
        results = model.predict(source=source, show=show, conf=confidence, stream=True)
        for r in results:
            # When stream=True, results is a generator. OpenCV handles window rendering if show=True.
            # We can break the loop if key 'q' is pressed in the cv2 window.
            if show:
                # cv2.waitKey(1) is needed to process window events, ultralytics' show does this internally but
                # we double check. If we want custom control, we can load cv2 capture.
                pass
    else:
        # Check if file exists (if not a URL)
        is_url = isinstance(source, str) and (source.startswith("http://") or source.startswith("https://"))
        if not is_url and not os.path.exists(source):
            print(f"Error: Source path '{source}' does not exist.")
            sys.exit(1)

        print(f"Running inference on: {source}")
        
        # Run inference
        # project and name arguments control where results are saved: output_dir/predict
        results = model(
            source=source,
            conf=confidence,
            save=save,
            show=show,
            project=output_dir,
            name="detection_results",
            exist_ok=True
        )

        if save:
            # Let's find out where the results were saved
            save_dir = os.path.join(output_dir, "detection_results")
            print(f"\nInference complete! Results saved to: {os.path.abspath(save_dir)}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection CLI")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, video, directory, or webcam index (e.g., '0' for webcam)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model weight name (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for object detection (default: 0.25)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the results in a window during inference"
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        help="Do not save the detection results to disk"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Directory to save detection results (default: 'runs')"
    )

    args = parser.parse_args()

    run_detection(
        source=args.source,
        model_name=args.model,
        confidence=args.conf,
        show=args.show,
        save=args.save,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
