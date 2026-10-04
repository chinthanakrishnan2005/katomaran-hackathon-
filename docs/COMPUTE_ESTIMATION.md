# Compute Estimation

## Performance Analysis
The current implementation was tested on a local CPU using `face-demographics-walking.mp4`.

### Metrics Observed
- **Processed Frames**: 732 frames
- **Processing Time**: ~69.66 seconds
- **Throughput**: ~10.5 Frames Per Second (FPS)

*(Note: Actual throughput varies dynamically based on the number of people in the frame since tracking and InsightFace processing time scale linearly per detected track.)*

## Compute Utilization Estimates

### CPU-Only Execution (Current Baseline)
- **YOLOv8n Object Tracking**: ~30-50ms per frame.
- **InsightFace (buffalo_l)**: ~150-250ms per extracted embedding (only executed once per new track or periodically, not every frame).
- **ByteTrack Association**: <5ms per frame.
- **Estimated CPU Load**: 80-100% on 4+ cores.

### GPU Execution (Optimized)
With CUDA installed and `onnxruntime-gpu` + `torch` built with CUDA:
- **YOLOv8n**: ~5-15ms per frame.
- **InsightFace**: ~10-20ms per face.
- **Expected Throughput**: 30-60+ FPS, allowing for real-time dense crowd processing.

## Memory Requirements
- **RAM**: ~1.5 - 2.5 GB. 
  - `yolov8n.pt` footprint is very small (~10MB).
  - `buffalo_l` consumes ~300MB in memory.
  - OpenCV video buffers consume the rest.
- **VRAM (If using GPU)**: Minimum 2GB, recommended 4GB for smooth execution with multiple bounding boxes.

## Optimization Opportunities
1. **Frame Skipping**: The `config.json` allows increasing `frame_skip`. For high-framerate RTSP streams, skipping every 2-3 frames cuts processing time massively without breaking ByteTrack.
2. **GPU Acceleration**: Switch `CPUExecutionProvider` to `CUDAExecutionProvider` in `src/recognizer.py`, and install PyTorch with CUDA for Ultralytics.
3. **Gallery Size**: Right now, embeddings are searched linearly using `scipy.spatial.distance.cosine`. If unique visitor counts exceed 100,000, we should migrate the gallery to a vector database like FAISS or Milvus to maintain low latency.
