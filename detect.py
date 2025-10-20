"""Simple YOLOv5 detection script with optional Intel IPEX optimization.

This refactor makes the script modular and adds a minimal CLI.
"""

from __future__ import annotations

import argparse
import sys
import os
from typing import Tuple

import numpy as np
import requests
import torch
import cv2

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except Exception:
    ipex = None
    IPEX_AVAILABLE = False


def download_image(url: str, out_path: str) -> None:
    """Download an image from `url` and save to `out_path`."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)


def load_model(model_name: str = "yolov5s", trust_repo: bool = True) -> torch.nn.Module:
    """Load a YOLOv5 model from torch.hub (uses ultralytics/yolov5).

    Args:
        model_name: model id (e.g. 'yolov5s')
        trust_repo: whether to trust the repo (affects torch.hub warning behavior)
    """
    print("Loading the YOLOv5 model...")
    model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True, trust_repo=trust_repo)
    print("Model loaded successfully!")

    # Optionally apply Intel IPEX optimizations if available
    if IPEX_AVAILABLE:
        try:
            print("Optimizing model with Intel Extension for PyTorch (IPEX)...")
            model.eval()
            model = ipex.optimize(model)
            print("Model optimized successfully!")
        except Exception as e:
            print("Warning: IPEX optimization failed, continuing without IPEX:\n", e, file=sys.stderr)

    return model


def run_detection(model: torch.nn.Module, image: np.ndarray) -> Tuple[np.ndarray, object]:
    """Run the model on `image`, draw boxes on the image, and return (annotated_image, results).

    `results` is the raw output from the YOLOv5 model (for programmatic access).
    """
    print("Running detection...")
    results = model(image)
    results.print()

    # If results.xyxy[0] is available, iterate and draw boxes.
    # Use xyxyn (normalized) if coordinates are normalized.
    if hasattr(results, "xyxyn") and len(results.xyxyn) > 0:
        h, w = image.shape[:2]
        for *box, conf, cls in results.xyxyn[0]:
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            label = f"{results.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLOv5 detection on an image (optional IPEX optimized)")
    p.add_argument("--image", default="zidane.jpg", help="Path to input image (will be downloaded if not present)")
    p.add_argument("--url", default="https://ultralytics.com/images/zidane.jpg", help="URL to download when image not found")
    p.add_argument("--output", default="detection_result.jpg", help="Path to save annotated output image")
    p.add_argument("--model", default="yolov5s", help="Model id to load from ultralytics/yolov5")
    p.add_argument("--no-ipex", action="store_true", help="Disable IPEX optimization even if available")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Ensure OpenCV is available at runtime
    if not hasattr(cv2, "imread"):
        print("Error: OpenCV (cv2) is not available.", file=sys.stderr)
        return 2

    # Download image if necessary
    if not os.path.exists(args.image):
        print(f"Downloading sample image to {args.image}...")
        try:
            download_image(args.url, args.image)
        except Exception as e:
            print(f"Failed to download image: {e}", file=sys.stderr)
            return 3

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Failed to read image '{args.image}'", file=sys.stderr)
        return 4

    # Respect --no-ipex flag
    global IPEX_AVAILABLE
    if args.no_ipex:
        IPEX_AVAILABLE = False

    model = load_model(args.model, trust_repo=True)
    annotated, results = run_detection(model, img)

    cv2.imwrite(args.output, annotated)
    print(f"Detection complete! Image saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())