# Gesture Reactor

Gesture Reactor is a small real-time computer vision project that detects human gestures and facial expressions from a camera feed and reacts by displaying matching memes.

The project is primarily an experiment in computer vision, real-time event processing, and gesture recognition, with the possibility of later extending the same concepts toward robotics and human-machine interaction.

## Goals

- Detect selected hand gestures and facial expressions in real time.
- Match detected gestures or expressions with relevant meme reactions.
- Keep the project lightweight and easy to experiment with.
- Learn practical computer vision concepts through immediate visual feedback.
- Leave room for future experiments with robotics and human-machine interaction.

## Planned Direction

The first version will focus on a simple flow:

Camera input → gesture / expression detection → recognition → event → meme matching → display.

The initial scope will stay intentionally small. More advanced features such as custom model training, temporal gesture recognition, pose tracking, or robotics integration may be explored later.

## Tech

- Python 3.14
- OpenCV 5
- MediaPipe
- NumPy

## Installation

Create and activate a Python 3.14 virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install only the application dependencies:

```bash
python -m pip install -r requirements.txt
```

For development, install the application together with formatting, linting, and type-checking
tools:

```bash
python -m pip install -r requirements-dev.txt
```

## Running

Run the application from the repository root so it can find the model and meme assets:

```bash
python src/main.py
```

## Status

Early development / experimentation.
