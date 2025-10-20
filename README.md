# Intel-oneAPI-Lets-Try-

A small sample project that runs YOLOv5 detection with optional Intel Extension for PyTorch (IPEX) optimization.

## Files

- `detect.py` - Refactored detection script with a simple CLI. Downloads a sample image if not present and saves an annotated image.
- `requirements.txt` - Python package requirements to reproduce the environment.
- `yolov5s.pt` - (Optional) local YOLOv5 weights file if you want to avoid downloading.

## Quickstart

1. Create a virtual environment and activate it (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install requirements:

```bash
python -m pip install -r requirements.txt
```

> In containers or servers without X11, we use `opencv-python-headless` to avoid GUI system dependencies.

3. Run detection:

```bash
python detect.py
```

Optional flags:
- `--image PATH` : path to the input image (will be downloaded if missing)
- `--url URL` : URL used to download the sample image
- `--output PATH` : output path for the annotated image
- `--model MODEL` : model id to load from ultralytics/yolov5 (default: `yolov5s`)
- `--no-ipex` : disable IPEX optimization even if installed

## Notes

- The repository uses Intel Extension for PyTorch (IPEX) to optimize CPU inference when available. IPEX is version-sensitive and requires matching PyTorch versions (e.g., IPEX 2.8 works with PyTorch 2.8).
- If you encounter `libGL.so.1` import errors with OpenCV, install OS packages or use the headless OpenCV package (`opencv-python-headless`).

## License

MIT
