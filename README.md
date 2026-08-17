# Single-Subject Face Recognition Pipeline: RPi 5 + Hailo AI HAT Architecture

## Implementation Status Summary
- **Phase 1: Environment & System Setup** — [COMPLETED]
- **Phase 2: Project Architecture & Layout** — [COMPLETED]
- **Phase 3: Real-time Camera Feed & Hailo NPU Face Detection** — [COMPLETED]
- **Phase 4: Target Profile Enrollment (`enroll.py`)** — [PENDING]
- **Phase 5: ArcFace Embedding Integration & Cosine Matcher** — [PENDING]
- **Phase 6: End-to-End System Optimization & Benchmarking** — [PENDING]

---

## 1. System Architecture

```
+---------------------------+      +---------------------------+      +---------------------------+
| Raspberry Pi Camera       | ---> | Picamera2 Frame Stream    | ---> | Hailo NPU (SCRFD HEF)     |
| (CSI Port / rp1-cfe)      |      | (Direct RGB888 NumPy Array)|      | Face Detection & NMS      |
+---------------------------+      +---------------------------+      +---------------------------+
                                                                                    |
                                                                                    v
+---------------------------+      +---------------------------+      +---------------------------+
| Target Visual Annotation  | <--- | Centroid Cosine Similarity| <--- | Hailo NPU (ArcFace HEF)   |
| (OpenCV Frame Display)    |      | Engine (Python NumPy)     |      | Feature Embedding (512-D) |
+---------------------------+      +---------------------------+      +---------------------------+
```

### Core Components
* **Zero-Copy Capture:** `Picamera2` captures standard RGB arrays natively, avoiding V4L2/GStreamer buffer stalls.
* **NPU Acceleration:** Hailo Executable Format (`.hef`) models offload heavy inference from RPi 5 CPU:
  * **Face Detection:** `scrfd_2.5g_h8l.hef` with manual multi-stride anchor grid generation and OpenCV NMS.
  * **Feature Embedding:** `arcface_mobilefacenet_8l.hef` (outputs normalized 512-D vectors).
* **Environment:** `uv` environment created with `--system-site-packages` to inherit `python3-picamera2`, `libcamera`, and `python3-hailort`.

---

## 2. Directory & Project Structure

```text
recognizer/
├── models/
│   ├── scrfd_2.5g_h8l.hef              # [ACTIVE] SCRFD face detection model
│   └── arcface_mobilefacenet_8l.hef    # [PENDING] ArcFace feature extraction model
│
├── data/                               # [PENDING]
│   ├── raw_target/                     # Directory for target face enrollment photos (100+)
│   └── target_profile.npy              # Stored target centroid vector
│
├── src/
│   └── recognizer/
│       ├── __init__.py                 # [COMPLETED] Module entry point and live event loop
│       ├── detector.py                 # [COMPLETED] HailoFaceDetector class (SCRFD + NMS)
│       ├── enroll.py                   # [PENDING] Target photo processing & centroid generator
│       └── matcher.py                  # [PENDING] Cosine similarity evaluation engine
│
├── pyproject.toml                      # Package config & CLI command registration
├── README.md
└── uv.lock
```

---

## 3. Completed Phases

### Phase 1: System Dependencies & Environment
- Installed `hailo-all`, `python3-hailort`, `python3-picamera2`, and `python3-opencv` via `apt`.
- Configured isolated `uv` virtual environment with `--system-site-packages` access.

### Phase 2: Project Layout & Asset Isolation
- Placed binary `.hef` model files at root (`models/`) to keep package builds clean and avoid git tracking bloat.
- Configured absolute project root resolution in Python via `Path(__file__).resolve().parents[...]`.

### Phase 3: Hailo NPU Face Detection (`detector.py` & `__init__.py`)
- Built `HailoFaceDetector` class supporting Hailo PCIe Virtual Streams (`InferVStreams`).
- Implemented multi-stride anchor grid generation ($8, 16, 32$) and tensor unpacking for raw SCRFD output streams.
- Integrated `cv2.dnn.NMSBoxes` to suppress overlapping detection bounding boxes.
- Implemented continuous video stream rendering using `Picamera2` and OpenCV.

---

## 4. Pending Implementation Phases

### Phase 4: Target Profile Enrollment (`src/recognizer/enroll.py`)
1. Ingest baseline images of target subject from `data/raw_target/`.
2. Crop and align faces using bounding boxes/landmarks from `scrfd_2.5g_h8l.hef`.
3. Feed normalized face crops ($112 \times 112$) into `arcface_mobilefacenet_8l.hef` on Hailo NPU.
4. Calculate 512-dimensional feature vector $v_i$ per face.
5. Filter outliers deviating $> 0.15$ standard deviations from the mean vector.
6. Compute and save L2-normalized centroid to `data/target_profile.npy`:
   $$\mathbf{v}_{\text{target}} = \frac{\sum \mathbf{v}_i}{\left\| \sum \mathbf{v}_i \right\|_2}$$

### Phase 5: Cosine Similarity Matching (`src/recognizer/matcher.py`)
1. Create `TargetMatcher` class to evaluate live face embeddings against stored centroid:
   $$\text{similarity} = \mathbf{v}_{\text{live}} \cdot \mathbf{v}_{\text{target}}$$
2. Apply decision threshold ($\ge 0.45$):
   - **Target Match:** Draw green bounding box `(0, 255, 0)` with similarity score.
   - **Unknown Face:** Draw gray bounding box `(128, 128, 128)`.

### Phase 6: System Integration & Execution
1. Register CLI command in `pyproject.toml`:
   ```toml
   [project.scripts]
   recognizer = "recognizer:main"
   ```
2. Run full recognition stream:
   ```bash
   uv run recognizer
   ```

---

## 5. Performance Metrics & Benchmarks

| Metric | Target Benchmark | Current Status |
| :--- | :--- | :--- |
| **Video Stream Resolution** | 1280x720 @ 30 FPS | **VERIFIED (30 FPS)** |
| **Face Detection Latency** | ~5–10 ms on Hailo-8L NPU | **VERIFIED** |
| **NMS Post-Processing Overhead** | < 2 ms per frame | **VERIFIED** |
| **Feature Extraction (ArcFace)** | ~3–6 ms per crop | PENDING |
| **Overall Recognition Pipeline** | $\ge 25$ FPS continuous | PENDING |
