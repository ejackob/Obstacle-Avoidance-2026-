# Developer Guide — Obstacle Avoidance App

> This document is intended for developers joining the project after the original team. It supplements the top-level `README.md` with an architectural overview, data-flow walkthrough, component descriptions, and known gotchas.

---

## Table of Contents

1. [What the App Does](#1-what-the-app-does)
2. [Tech Stack at a Glance](#2-tech-stack-at-a-glance)
3. [Project Structure](#3-project-structure)
4. [How to Set Up and Run](#4-how-to-set-up-and-run)
5. [Architecture & Data Flow](#5-architecture--data-flow)
6. [Key Components](#6-key-components)
7. [The "Corridor" System](#7-the-corridor-system)
8. [Threat Scoring & Audio Pipeline](#8-threat-scoring--audio-pipeline)
9. [ML Model Details](#9-ml-model-details)
10. [Local Storage & Preferences](#10-local-storage--preferences)
11. [Python Model Tooling](#11-python-model-tooling)
12. [Testing](#12-testing)
13. [Known Gotchas & Rough Edges](#13-known-gotchas--rough-edges)
14. [Suggested Future Work](#14-suggested-future-work)

---

## 1. What the App Does

The app uses the iPhone/iPad camera (plus LiDAR/scene depth where available) to detect nearby objects in real time and **speak warnings to the user** about what is in their path. The primary audience is people with visual impairments, so the app is designed to work well with **VoiceOver** enabled.

Core user experience:
- Open the camera tab → point the device forward.
- Objects are detected and assigned a **threat score** based on how close they are, how centered they are in the frame, and what kind of object they are.
- The highest-threat object is announced audibly (e.g. *"Person, center, 1.2 meters"*).
- A red bounding box highlights the most dangerous object on screen.
- A **corridor overlay** shows the safe walking zone, which narrows dynamically when obstacles are close.

---

## 2. Tech Stack at a Glance

| Concern | Technology |
|---|---|
| Language | Swift (app), Python (offline tooling) |
| UI framework | SwiftUI (with UIKit bridging where needed) |
| Camera & spatial data | ARKit (`ARWorldTrackingConfiguration` + scene depth) |
| Object detection | Core ML + Apple Vision framework |
| ML model | YOLOv3-tiny (`.mlpackage` bundled in app) |
| User preferences | `UserDefaults` (on-device, no backend required) |
| Unit / UI tests | XCTest + ViewInspector (SPM) |
| Python tooling | Ultralytics YOLO, PyTorch, Core ML Tools |

---

## 3. Project Structure

```
obstacle_avoidance_2026/
├── obstacle_avoidance/          # Main iOS app source
│   ├── Presentation/            # SwiftUI views + FrameHandler (camera pipeline)
│   ├── Logic/                   # Detection, threat scoring, audio logic
│   ├── Data/                    # Structs, config, local storage, ML model files
│   └── Assets.xcassets/
├── obstacle_avoidanceTests/           # Unit tests
├── obstacle_avoidanceLogicTests/      # Logic-layer unit tests
├── obstacle_avoidanceUITests/         # UI tests
├── modelTestingScript/                # Python tooling (offline, not part of iOS build)
│   └── modelPipeline/YoloPipeline.py
├── obstacle_avoidance.xcodeproj/
└── README.md
```

### Presentation layer files

| File | Role |
|---|---|
| `ObstacleAvoidanceApp.swift` | App entry point |
| `ContentView.swift` | Tab container: Home / Camera / Settings |
| `CameraView.swift` | Owns `FrameHandler`; hosts AR preview + overlays |
| `FrameView.swift` | Draws the red bounding box; polls audio queue for VoiceOver announcements |
| `CorridorOverlay.swift` | Renders the colored corridor (safe zone) on top of the camera feed |
| `FrameHandler.swift` | **Core of the camera pipeline** — see Section 5 |
| `SettingsView.swift` | User preferences (units, height, emergency contacts) |
| `ECView.swift` | Emergency contacts management screen |
| `InstructionView.swift` | Home/help screen |
| `BoundingBoxView.swift` | Helper view for drawing individual boxes |

### Logic layer files

| File | Role |
|---|---|
| `DecisionBlock.swift` | Decides whether a detection is worth announcing and at what severity |
| `AudioQueue.swift` | Priority queue for speech items; handles cooldowns and deduplication |
| `AudioPolicy.swift` / `AudioPolicyConfig.swift` | Configurable thresholds for queue behavior |
| `NMSHandler.swift` | Thin wrapper that calls into `NonMaxSuppression.swift` |
| `NonMaxSuppression.swift` | Multi-class NMS implementation |
| `DetectionUtils.swift` | Maps bounding box positions to clock-position / horizontal zones |
| `CorridorUtils.swift` | Maps bounding box + `CorridorGeometry` → Left/Center/Right/Outside string |
| `YoloDecoder.swift` | Raw YOLO tensor decoder (alternate detection path — see Section 13) |

### Data layer files

| File | Role |
|---|---|
| `Struct.swift` | `User`, `EmergencyContact`, and preference structs |
| `UserDefaultsHandler.swift` | Wrappers for all `UserDefaults` reads and writes |
| `ThreatLevelConfig.swift` / `ThreatLevelConfigV3.swift` | Per-class and per-corridor threat weights |
| `YOLOv3Tiny.mlpackage` | Bundled Core ML model (primary detector) |
| `nonchalant.mlpackage` | Alternate/experimental Core ML model |

---

## 4. How to Set Up and Run

### Prerequisites

- macOS with **Xcode 15+**
- An iPhone or iPad running **iOS 16+** (physical device strongly recommended — ARKit does not work in the simulator)

### Steps

1. **Clone the repo** and open `obstacle_avoidance.xcodeproj` in Xcode.

2. **Resolve SPM dependencies** — Xcode should do this automatically. The packages are:
   - `ViewInspector` (test only)
   - `swift-collections` (used by `AudioQueue`)

3. **Select your physical device** as the run target and press Run (`⌘R`).

4. Grant camera and motion permissions when prompted on first launch.

There is no backend or external service to configure. All user preferences are stored on-device via `UserDefaults`.

---

## 5. Architecture & Data Flow

The app follows a **linear pipeline** from camera frames to spoken output:

```
ARKit frame
    │
    ▼
FrameHandler.session(_:didUpdate:)
    │  extracts pixel buffer + scene depth map
    │
    ▼
Vision / Core ML (YOLOv3Tiny)
    │  produces VNRecognizedObjectObservation[]
    │
    ▼
NMSHandler.performNMS()
    │  removes overlapping duplicates
    │
    ▼
FrameHandler (depth + threat scoring)
    │  • samples depth map at each bbox → median depth
    │  • proximityFactor (70%) + centerednessFactor (30%) → threat score
    │  • EMA smoothing on distance for the winning box
    │  • updates @Published properties (bounding boxes, stress, corridor geometry)
    │
    ▼
DecisionBlock.computeThreatLevel()
    │  • applies class weights (ThreatLevelConfigV3)
    │  • applies corridor/position weights
    │  • applies vertical-band multiplier (upper third = higher danger)
    │  • passesHysteresis() — suppresses flicker
    │
    ▼
AudioQueue.addToHeap()
    │  priority heap keyed by severity band
    │  per-key and global cooldowns; deduplication; TTL
    │
    ▼
FrameView timer (polls every ~0.5 s)
    │  AudioQueue.popHighestPriorityObject()
    │
    ▼
UIAccessibility.post(.announcement, value: message)
    (VoiceOver speaks the message aloud)
```

**State flows through `FrameHandler`** via `@Published` properties that SwiftUI views observe:
- `boundingBoxes` — displayed by `FrameView` / `BoundingBoxView`
- `stress` — used by `CorridorOverlay` to widen/narrow the safe corridor
- `corridorGeometry` — shared between `CorridorOverlay` and `CorridorUtils`

---

## 6. Key Components

### FrameHandler

`FrameHandler.swift` is the most important file in the codebase. It:
- Acts as `ARSessionDelegate` and receives every camera frame.
- Kicks off a `VNCoreMLRequest` for object detection.
- Reads the scene depth buffer (LiDAR / scene reconstruction) and computes a median depth value per bounding box.
- Picks the single "most threatening" box and smooths its distance with an exponential moving average (EMA).
- Publishes results so views can update.
- Feeds detections into `DecisionBlock`.

If you need to change detection behavior, start here.

### DecisionBlock

`DecisionBlock.swift` is the **policy engine**. It translates a raw detection + distance into a severity band and decides if an announcement should be made. Key methods:
- `computeThreatLevel(_:distance:corridorPosition:)` — returns a 0–1 float.
- `computeSeverityBand(_:)` — maps the threat level to Low / Medium / High.
- `passesHysteresis(_:)` — prevents the same object from spamming announcements.

Tune `ThreatLevelConfigV3` to adjust per-class weights and `AudioPolicyConfig` to adjust cooldown durations.

### AudioQueue

`AudioQueue.swift` is a **priority heap** of speech items. Key behaviors:
- Items have a TTL; stale items are discarded before popping.
- Per-object and global cooldowns prevent the same thing from being said repeatedly.
- Severity bands determine priority (High > Medium > Low).
- `FrameView` pops from this queue on a timer and posts VoiceOver announcements.

---

## 7. The "Corridor" System

The corridor is the visual representation of the "safe path" in front of the user. It is rendered as a colored trapezoid overlay on the camera feed.

- **`CorridorGeometry`** is a struct holding the computed geometry (left edge, right edge, width) of the corridor.
- **`CorridorOverlay`** reads the `stress` value published by `FrameHandler` and calls `calculateCorridor(stress:)` to derive the geometry, then renders it.
- **`CorridorUtils`** uses a `CorridorGeometry` instance to classify bounding box positions as Left / Center / Right / Outside. This string is used in speech output ("obstacle in center of corridor") and in threat weighting.

> **Watch out:** There are two representations of the corridor in the code. `CorridorGeometry` (from `CorridorOverlay`) describes the visual strip, while `CorridorUtils` uses it to classify positions. If you refactor corridor layout, make sure both stay in sync.

---

## 8. Threat Scoring & Audio Pipeline

### Threat score (in FrameHandler)

Used to pick *which* box gets the bounding box highlight:

```
threatScore = 0.70 × proximityFactor + 0.30 × centerednessFactor
```

- `proximityFactor` = inverse of depth (clamped).
- `centerednessFactor` = how close the box center is to the horizontal midpoint of the frame.

### Threat level (in DecisionBlock)

Used to decide *what to say* and *how urgently*:

```
threatLevel = baseWeight(class)
            × corridorWeight(position)
            × verticalMultiplier(yPosition)
            × inverseDistance(depth)
```

All weights are defined in `ThreatLevelConfigV3.swift` and can be tuned without touching pipeline logic.

### Audio policy knobs (AudioPolicyConfig)

| Setting | Effect |
|---|---|
| `globalCooldown` | Minimum time between any two announcements |
| `perKeyCooldown` | Minimum time before the same object label is repeated |
| `perObjectCooldown` | Minimum time before the same tracked object is re-announced |
| Severity band thresholds | Float cutoffs that map threat level → Low/Medium/High |

---

## 9. ML Model Details

### Deployed model: YOLOv3-tiny

- **Input:** 416×416 RGB image.
- **Output:** Up to 80-class COCO object detections with confidence scores and bounding boxes.
- **Integration:** Loaded via `VNCoreMLModel` + `VNCoreMLRequest`; results come back as `VNRecognizedObjectObservation` objects.
- **File:** `obstacle_avoidance/Data/YOLOv3Tiny.mlpackage`

### Alternate model: nonchalant

`nonchalant.mlpackage` is a second Core ML model present in the bundle. It appears to be an experimental/alternative detector. Check `FrameHandler` for where model selection happens if you want to switch.

### Raw tensor decode path (YoloDecoder)

`YoloDecoder.swift` implements a manual YOLO output decoder (anchor boxes, sigmoid activations, etc.). This path is used if you load the model as a raw `MLModel` instead of going through the Vision framework. It is **not** the active path in the current codebase but is preserved as a reference.

### Python tooling (offline)

`modelTestingScript/modelPipeline/YoloPipeline.py` is used **offline** for:
- Running evaluation on a labeled dataset.
- Exporting a trained YOLO model to Core ML format (`.mlpackage`).
- Comparing model variants (full vs. tiny, FP32 vs. INT8).

This script is not part of the iOS build. To use it, install the Python dependencies (`ultralytics`, `torch`, `coremltools`, `pandas`, `matplotlib`) in a separate Python environment.

---

## 10. Local Storage & Preferences

The app has no backend. All persistent data is stored on-device.

**`UserDefaultsHandler`** (`UserDefaultsHandler.swift`) is the single point of contact for reading and writing preferences. It wraps `UserDefaults.standard` with typed getters/setters for:

| Key | Type | Default |
|---|---|---|
| `measurement_type` | `String` ("Feet" or "Meters") | `"Feet"` |
| `user_height` | `Double` (inches) | `60.0` |
| `haptic_feedback` | `Bool` | `false` |
| `location_sharing` | `Bool` | `false` |

SwiftUI views also use `@AppStorage` directly for some keys (e.g. `measurementType`, `userHeight`) — these read from the same `UserDefaults` store, so they stay in sync automatically.

**`Struct.swift`** still defines `EmergencyContact` (name, phone number, address). Emergency contacts are managed locally in the Settings tab via `ECView`; they are not synced to any server.

The ML model (`YOLOv3Tiny.mlpackage`) is bundled directly in the app target and loaded from the app bundle at runtime — no download or network call is needed.

---

## 11. Python Model Tooling

Located in `modelTestingScript/modelPipeline/YoloPipeline.py`.

Use this when you want to:
- **Retrain or fine-tune** a YOLO model on a custom dataset.
- **Export** the result to Core ML so it can be dropped into the iOS app.
- **Benchmark** model accuracy (mAP, precision, recall) on a test set.

Typical workflow:
1. Prepare a dataset in YOLO format (images + label `.txt` files).
2. Point the script at the dataset and configure model variant.
3. Run training / evaluation.
4. Export `.mlpackage` and replace the file in `obstacle_avoidance/Data/`.
5. If the model output shape changed, update `YoloDecoder.swift` anchors/classes or use the Vision framework path (which handles this automatically).

---

## 12. Testing

Three test targets exist:

| Target | What it covers |
|---|---|
| `obstacle_avoidanceTests` | User defaults, bounding box math, settings |
| `obstacle_avoidanceLogicTests` | NMS, DecisionBlock, AudioQueue, CorridorUtils |
| `obstacle_avoidanceUITests` | Basic UI flow tests using ViewInspector |

Run tests with `⌘U` in Xcode or via `xcodebuild test` on the command line.

Tests that touch `FrameHandler` or `ARSession` may require a physical device.

---

## 13. Known Gotchas & Rough Edges

1. **Two corridor representations.** `CorridorGeometry` (used for rendering in `CorridorOverlay`) and the position logic in `CorridorUtils` are separate. Refactoring the corridor concept requires updating both sides.

2. **Two detection paths.** The active path uses `VNCoreMLRequest` → `VNRecognizedObjectObservation`. The alternate path uses `YoloDecoder` (raw tensor decode). Both are in the codebase; only the Vision path runs at runtime. The `YoloDecoder` path exists for reference if you want to use a model that doesn't fit neatly into the Vision framework.

3. **LiDAR / scene depth is device-dependent.** Depth data is only available on devices with LiDAR (iPhone 12 Pro and later, iPad Pro 2020 and later). The app degrades gracefully (depth values default to a sentinel) but distance announcements will be absent on non-LiDAR devices.

4. **The `nonchalant.mlpackage` model.** This model is bundled but the selection logic determining when it's used vs. `YOLOv3Tiny` should be verified in `FrameHandler` before relying on it.

5. **EMA smoothing on distance.** The exponential moving average in `FrameHandler` smooths depth over frames for the "winning" box. If object identity jumps between frames (e.g. the highest-threat box switches rapidly), the EMA value can lag. This is a known trade-off.

6. **Info.plist usage strings.** The `obstacle-avoidance-Info.plist` checked into the repo appears empty. Camera and ARKit permission usage descriptions may live in Xcode target build settings — check there if you see permission crashes on a fresh install.

---

## 14. Suggested Future Work

Based on the current state of the codebase, areas the original team identified or that are natural next steps:

- **Expanded class weights.** `ThreatLevelConfigV3` has weights for common COCO classes but many are at default. Tuning these with user studies would improve the experience.
- **Better object tracking.** Currently the "highest threat" box is re-selected each frame independently. Adding a proper object tracker (e.g. IoU-based tracking between frames) would reduce announcement flicker and improve EMA accuracy.
- **Custom YOLO model.** The Python tooling exists to train a custom model on obstacle-specific data (e.g. curbs, poles, stairs). This would likely improve real-world performance over the general-purpose YOLOv3-tiny.
- **Haptic feedback.** The audio pipeline is complete; a parallel haptic channel could give additional cues without requiring VoiceOver.
- **Emergency contact persistence.** Emergency contacts currently live in memory during a session. Persisting them to `UserDefaults` (or a local JSON file) would make them survive app restarts.
- **SwiftUI navigation modernization.** Some navigation patterns use older approaches; migrating fully to `NavigationStack` (iOS 16+) would be straightforward.
- **Simulator-compatible testing.** The current test suite requires a device for ARKit-dependent tests. Abstracting `ARSession` behind a protocol would allow more tests to run in CI.
