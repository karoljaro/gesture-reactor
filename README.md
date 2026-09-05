# Gesture Reactor

Gesture Reactor is a real-time computer vision project that reacts to hand gestures and facial expressions detected from a webcam by displaying matching memes.

The application uses MediaPipe Tasks for hand and face tracking, OpenCV for webcam input and display, and a small custom classification layer that turns raw landmark data into recognizable gestures and expressions.

## Version

**v1.0.0**

This version marks the first complete version of the project.

## What it does

Gesture Reactor watches the webcam feed and recognizes selected hand gestures and facial expressions in real time.

When a supported gesture or expression becomes stable, the application selects a matching meme and displays it in a separate window.

It can also react to combinations of a gesture and facial expression, for example:

- `PEACE + SMILE`
- `FIST + FROWN`
- `POINTING + SURPRISED`

The detections are stabilized before they are used, which helps prevent reactions caused by single incorrect or unstable frames.

## Supported hand gestures

- `FIST`
- `OPEN_PALM`
- `POINTING`
- `PEACE`

Unrecognized hand poses are treated as `UNKNOWN` and are not sent to the meme reaction layer.

## Supported facial expressions

- `SMILE`
- `FROWN`
- `EYES_CLOSED`
- `SURPRISED`
- `NEUTRAL`

`NEUTRAL` is used as a normal face state and does not trigger a meme by itself.

## How it works

The program continuously reads frames from the webcam and prepares them for MediaPipe.

Each frame is processed by both the hand and face landmarkers. Their results are interpreted separately:

- hand landmarks are analyzed to determine finger states and classify a gesture,
- face blendshapes are analyzed to classify a facial expression.

The detected values are then stabilized across multiple results so that short recognition errors do not immediately trigger a reaction.

Once a stable gesture or expression is available, `MemeReactor` checks whether it can form a supported hand-and-face combination. A matching combination has priority over a reaction to only the hand gesture or only the facial expression.

The selected meme is displayed in a separate OpenCV window while the webcam feed remains visible.

## Reaction examples

| Input | Reaction |
| --- | --- |
| `PEACE` | Meme assigned to the peace gesture |
| `SMILE` | Meme assigned to a smile |
| `PEACE + SMILE` | Dedicated combined meme |
| `FIST + FROWN` | Dedicated combined meme |
| `POINTING + SURPRISED` | Dedicated combined meme |

Meme paths can be changed or extended directly in the reaction configuration.

## Tech stack

- Python 3.14
- MediaPipe Tasks
- OpenCV 5
- NumPy
- mypy
- Black
- Flake8
- isort

## Models

The project uses MediaPipe Task model bundles:

```text
models/
├── hand_landmarker.task
└── face_landmarker.task
```

Make sure both model files are present before running the application.

## Meme assets

Meme files are stored under:

```text
assets/memes/
```

They can be grouped by gesture, expression, or combined reaction.

Example:

```text
assets/
└── memes/
    ├── fist/
    ├── open_palm/
    ├── peace/
    ├── pointing/
    ├── expression/
    └── combo/
        ├── peace_smile/
        ├── fist_frown/
        └── pointing_surprised/
```

## Requirements

- Python `>=3.14,<3.15`
- Working webcam
- MediaPipe model files in the `models/` directory

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Run

From the project root:

```bash
python src/main.py
```

The application opens the webcam feed and displays memes in a separate window when a supported reaction is detected.

Press:

```text
q
```

in the webcam window to stop the application.

## Development checks

Type checking:

```bash
mypy .
```

Formatting:

```bash
black .
```

Import sorting:

```bash
isort .
```

Linting:

```bash
flake8 .
```

## License

This project is licensed under the MIT License.