# Gesture Reactor

Gesture Reactor is a real-time computer vision side project that reacts to hand gestures and facial input from a webcam.

The current version recognizes selected hand gestures and displays a matching meme, while face detection is already integrated as a separate asynchronous vision branch.

The project is primarily focused on learning and experimenting with real-time vision pipelines, MediaPipe Tasks, OpenCV, asynchronous processing, gesture classification, and clean separation between detection, interpretation, reaction, and presentation.

## Current version

**v0.2.0**

### What works

- Real-time webcam capture
- Dedicated camera capture thread
- Latest-frame strategy to avoid frame backlog
- Shared frame preprocessing for MediaPipe
- Asynchronous MediaPipe Hand Landmarker
- Asynchronous MediaPipe Face Landmarker
- Parallel hand and face detection flow
- Generic `AsyncLandmarkProcessor`
- Hand pose analysis based on 3D hand landmarks
- Hand gesture classification
- Gesture stabilization
- Meme selection based on detected gesture
- Separate meme display window
- Separate camera frame display
- Optional debug landmark rendering pipeline
- Face detection pipeline prepared for future expression recognition

## Supported hand gestures

Currently recognized gestures:

- `FIST`
- `OPEN_PALM`
- `POINTING`
- `PEACE`

Unsupported or ambiguous hand poses fall back to `UNKNOWN`.

## Architecture

The application is split into a continuous video stream and asynchronous reaction flows.

```text
Camera
  ↓
VisionProcessor
  ↓
VisionDetectionPipeline
  ├── HandLandmarkerResult
  │          ↓
  │   HandReactionPipeline
  │          ↓
  │       Gesture
  │          ↓
  │     MemeReactor
  │          ↓
  │      MemeDisplay
  │
  └── FaceLandmarkerResult
             ↓
      FaceReactionPipeline
             ↓
        face detection
        (expressions later)

VisionDetectionPipeline
  ↓
DebugRenderPipeline
  ↓
FrameDisplay
```

### Detection

`VisionDetectionPipeline` submits the same prepared MediaPipe image to both hand and face landmarkers.

Each landmarker runs asynchronously and produces results independently. The pipeline does not wait for hand and face results to share the same timestamp.

### Async processing

`AsyncLandmarkProcessor` contains the common asynchronous MediaPipe mechanics:

- `detect_async()` submission
- result callback handling
- latest-result queue
- non-blocking result polling
- landmarker lifecycle management

The processor is generic and does not contain hand- or face-specific interpretation logic.

### Hand reaction flow

Raw `HandLandmarkerResult` data remains inside the hand-specific part of the application:

```text
HandLandmarkerResult
  ↓
HandPoseAnalyzer
  ↓
HandGestureClassifier
  ↓
HandGestureStabilizer
  ↓
Gesture
```

Only the resulting `Gesture` is passed to the shared reaction layer.

### Face reaction flow

Face detection already runs alongside hand detection.

At the moment, `FaceReactionPipeline` only handles face detection for development purposes. Expression analysis and classification are planned for a later version.

Raw `FaceLandmarkerResult` data is intended to stay inside the face-specific pipeline, just like hand landmark results stay inside the hand pipeline.

### Debug rendering

Landmark drawing is development-only functionality.

Detection results are dispatched to the debug renderer, which stores the latest available result. The video stream remains independent:

```text
new detection result
  ↓
update latest debug result

current frame
  ↓
draw latest available landmarks
  ↓
FrameDisplay
```

This avoids forcing exact frame/result synchronization and keeps debug visualization separate from the actual gesture reaction logic.

## Project structure

```text
src/
├── camera.py
├── main.py
├── presentation/
│   ├── frame_display.py
│   ├── hand_landmark_renderer.py
│   └── meme_display.py
├── reaction/
│   └── meme_reactor.py
└── vision/
    ├── async_landmark_processor.py
    ├── processor.py
    ├── types.py
    ├── face/
    │   ├── face_landmarker.py
    │   └── pipeline/
    │       └── face_reaction_pipeline.py
    ├── hand/
    │   ├── hand_landmarker.py
    │   ├── hand_gesture_classifier.py
    │   ├── hand_gesture_stabilizer.py
    │   ├── hand_pose_analyzer.py
    │   └── pipeline/
    │       └── hand_reaction_pipeline.py
    └── pipelines/
        ├── vision_detection_pipeline.py
        └── debug_render_pipeline.py
```

> The exact structure may evolve as the face-expression pipeline and additional reactions are implemented.

## Tech stack

- Python 3.14
- OpenCV
- MediaPipe Tasks
- NumPy
- mypy
- Black
- Flake8

## Models

The project uses MediaPipe Task model bundles stored locally:

```text
models/
├── hand_landmarker.task
└── face_landmarker.task
```

## Meme assets

Meme files are grouped by reaction:

```text
assets/
└── memes/
    ├── fist/
    ├── open_palm/
    ├── peace/
    └── pointing/
```

`MemeReactor` selects the appropriate meme path for a recognized gesture and passes it to `MemeDisplay`.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the required MediaPipe model files are available in the `models/` directory.

## Run

From the project root:

```bash
python src/main.py
```

The application will open the webcam stream and a separate meme display window.

## Roadmap

Planned next steps include:

- Face expression analysis
- Face expression classification
- Expression stabilization
- Meme reactions triggered by facial expressions
- Shared reaction handling for gestures and expressions
- Further gesture-recognition tuning
- Improved robustness for occlusion and ambiguous finger poses
- Optional performance and latency metrics for development mode

## Version history

### v0.2.0

- Added Face Landmarker
- Added parallel hand and face detection
- Added generic asynchronous landmark processor
- Added `VisionDetectionPipeline`
- Refactored hand processing into event-driven result dispatch
- Separated raw MediaPipe results from shared reaction logic
- Added development-oriented debug rendering flow
- Preserved working gesture-to-meme reactions after the architecture refactor

### v0.1.0

First working hand-gesture MVP:

- Webcam input
- Hand landmark detection
- Hand pose analysis
- Gesture classification
- Gesture stabilization
- Meme reactions
- Separate meme and camera displays

## Status

Gesture Reactor is an experimental learning project and is still under active development.