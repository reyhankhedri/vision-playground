# Vision Playground

This is a personal project I built to learn computer vision — mostly using OpenCV and MediaPipe. The idea was to build a handful of small, self-contained webcam experiments instead of one big thing, so each one taught me something different: color masking, background subtraction, facial landmarks, hand tracking.

It's not meant to be a polished product. Some of it works better than other parts (looking at you, Invisible Object, which really depends on how much green light is in the room).

## What's in it

**Invisible Object** — hold up something green and it disappears, replaced by whatever's behind it. Classic "invisible cloak" trick using HSV color masking.

**Human Invisible** — same idea, but for a person instead of an object. Uses background subtraction (MOG2) instead of color, since a person can be wearing anything.

**Gravity Vision** — a little penguin on screen that follows your fingertip around like it's magnetically attracted to it. Uses MediaPipe to track your hand.

**Mood Camera** — guesses whether you look happy, neutral, surprised, or sad, based on measurements taken from face landmarks (how open your mouth is, how wide you're smiling, where the corners of your mouth sit).

There's also a `secret_agent.py` file in `modes/` — a face detection + HUD experiment I built while figuring out MediaPipe. It's not one of the four main experiences, just left in as a leftover from development.

## Running it

You'll need Python and a webcam.

```
git clone https://github.com/reyhankhedri/vision-playground.git
cd vision-playground

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

A couple of MediaPipe model files also need to be downloaded (they're not in the repo since they're binary and easy to regenerate):

```
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "modes\models\face_landmarker.task"

Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "modes\models\hand_landmarker.task"
```

Then just run:

```
python app.py
```

You'll get a menu in the terminal — pick a number, and ESC gets you back to it from inside any mode.

## How it's organized

```
vision-playground/
├── core/               # camera + landmark helpers shared across modes
├── modes/               # one file per experience, plus models/ and assets/
├── ui/menu.py           # the terminal menu
├── utils/effects.py     # image overlay helper (alpha blending)
├── app.py               # entry point
└── requirements.txt
```

Things like `capture_background()` or `clean_mask()` from Invisible Object get reused in Human Invisible instead of copy-pasted — that was one of the bigger lessons from this project, honestly.

## Notes to self

- Invisible Object is very lighting-dependent. If it looks bad, it's probably the room, not the code.
- Mood Camera's thresholds were tuned against my own face — they might need adjusting for someone else.
- Gravity Vision originally had gesture detection (fist, gun, sparkles, all of it) but I scrapped it to keep things simple and working.