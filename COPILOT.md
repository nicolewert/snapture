# Snapture - AI Short-Form Video Director

## Project Overview

Snapture is a short-form AI director that:
- **Coaches the user live** while they perform
- **Clips the best moment** into a shareable cut
- **Persona-driven** coaching (e.g., "Super Bowl ad," "halftime dance")

**Target:** 10–20s TikTok/Reels-style shorts with live AI coaching.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React/TS)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ VideoPreview│  │   Controls   │  │    TranscriptPanel      │ │
│  │  (camera)   │  │ (start/stop) │  │  (AI coaching text)     │ │
│  └──────┬──────┘  └──────┬───────┘  └─────────────────────────┘ │
│         │                │                                       │
│  ┌──────▼────────────────▼───────────────────────────────────┐  │
│  │                    useSnapture Hook                        │  │
│  │  - Manages connection state, streaming, moments            │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │                  WebSocket Client                          │  │
│  │  Messages: setup, audio, video, end                        │  │
│  └──────────────────────────┬────────────────────────────────┘  │
└─────────────────────────────┼───────────────────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND (FastAPI)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ClientSession                          │   │
│  │  - Handles WS messages from frontend                      │   │
│  │  - Routes to Gemini + MomentDetector                      │   │
│  └────────────┬─────────────────────────────┬───────────────┘   │
│               │                             │                    │
│  ┌────────────▼────────────┐  ┌─────────────▼────────────────┐  │
│  │   GeminiLiveClient      │  │     MomentDetector           │  │
│  │  - Streams to Gemini    │  │  - MediaPipe face/hands/pose │  │
│  │  - Returns audio coach  │  │  - Detects bookmarkable      │  │
│  │                         │  │    moments                   │  │
│  └─────────────────────────┘  └──────────────────────────────┘  │
│                                                                  │
│  models/                                                         │
│  ├── face_landmarker.task    # Face detection + blendshapes     │
│  └── gesture_recognizer.task # Hand gesture recognition         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Gemini Live API
                    ┌─────────────────────┐
                    │   Gemini 2.5 Flash  │
                    │   (Audio + Vision)  │
                    └─────────────────────┘
```

---

## Key Files

### Frontend (`src/`)
| File | Purpose |
|------|---------|
| `hooks/useSnapture.ts` | Main hook - state, streaming, moment tracking |
| `lib/websocket.ts` | WebSocket client, message types |
| `lib/media.ts` | MediaCapture (4 FPS video, 16kHz audio), AudioPlayer |
| `components/VideoPreview.tsx` | Camera preview display |
| `components/Controls.tsx` | Start/stop recording buttons |
| `components/TranscriptPanel.tsx` | Shows AI coaching text |

### Backend (`server/`)
| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, `ClientSession` handles WebSocket |
| `gemini_client.py` | `GeminiLiveClient` - streams to Gemini Live API |
| `moment_detector.py` | `MomentDetector` - MediaPipe-based detection |
| `models/` | MediaPipe task files for face/gesture recognition |

---

## Message Flow

### Client → Server
```typescript
{ type: 'setup', config: { model: string, persona?: string } }
{ type: 'audio', data: string }  // base64 PCM 16kHz
{ type: 'video', data: string }  // base64 JPEG 640x480
{ type: 'end' }
```

### Server → Client
```typescript
{ type: 'connected', sessionId: string }
{ type: 'audio', data: string }      // Gemini voice coaching
{ type: 'text', content: string }    // Coaching transcript
{ type: 'moment', data: MomentEvent } // Detected moment (TODO)
{ type: 'error', message: string }
```

---

## Team Split

| Developer A (Recording/Playback) | Developer B (AI/Detection) |
|----------------------------------|---------------------------|
| Recording timestamps | Moment detection (MediaPipe) |
| Video playback | Gemini coaching prompts |
| Clip generation | Gesture/pose recognition |
| Final export UI | Persona system |
| Bookmark markers | Confidence scoring |

---

## Moment Detection System

### Overview: Event Hints vs. Gemini Decision-Making

**Architecture:** Snapture uses a **two-layer approach** for moment detection:

1. **Local Detection Layer (Backend):** `MomentDetector` analyzes video frames with MediaPipe to identify potential moments
2. **Gemini Decision Layer:** Video/audio stream + [SYSTEM EVENT] messages sent to Gemini, which decides:
   - Whether the moment is contextually important
   - Whether to call `bookmark_moment` tool for clipping
   - What coaching to provide

**Key Principle:** We send **all detected moments as hints to Gemini** (via [SYSTEM EVENT] text messages), but Gemini decides which are actually important based on conversational context and user intent.

### Detected Moment Types (12 Total)

| Moment Type | Detection Method | Confidence Threshold | Weight | Used For |
|-------------|------------------|----------------------|--------|----------|
| **SMILE** | Face blendshape | smile_score > 60% | 0.5 | Positive reaction, celebration |
| **SURPRISE** | Face blendshape | surprise_score > 50% | 0.95 | Reaction, discovery |
| **WINK** | Face blendshape | wink_score > 60% | 0.7 | Playful, flirty moment |
| **GOOD_FRAMING** | Face centered in frame | face_center == 1.0 | 0.5 | Technical quality |
| **THUMBS_UP** | Hand gesture | GestureRecognizer 0.9+ | 1.2 | ⭐ Highest value: approval, celebration |
| **PEACE_SIGN** | Hand gesture | GestureRecognizer 0.9+ | 0.9 | Victory, peace gesture |
| **WAVE** | Hand gesture | GestureRecognizer 0.9+ | 0.8 | Greeting, farewell |
| **POINT** | Hand gesture | GestureRecognizer 0.9+ | 0.8 | Direction, emphasis |
| **ARMS_UP** | Pose landmarks | arms_up_score > 0.5 | 1.1 | Celebration, enthusiasm |
| **T_POSE** | Pose landmarks | t_pose_score > 0.7 | 0.9 | Power pose, confidence |
| **HANDS_ON_HIPS** | Pose landmarks | hands_on_hips_score > 0.7 | 0.85 | Confidence, attitude |
| **LEAN_IN** | Pose landmarks | lean_in_score > 0.6 | 0.6 | Engagement, intensity |

### How Moments Flow to Gemini

```
┌─────────────────────────────────────────────────────────┐
│ VideoFrame (4 FPS) + AudioChunk (16kHz continuous)      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MomentDetector.process_frame()                          │
│  - MediaPipe FaceLandmarker (blendshapes)               │
│  - MediaPipe GestureRecognizer (hands)                  │
│  - MediaPipe Pose (body landmarks)                      │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
       ┌─────────────────────┐
       │ Moment Detected?    │
       └────────┬────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
      YES              NO
        │               │
        ▼               ▼
    [SYSTEM EVENT]  (silence)
    Text to Gemini
    e.g., "THUMBS_UP
    detected"
        │
        ▼
  ┌──────────────────────────┐
  │ Gemini Decides:          │
  │ - Contextually relevant? │
  │ - Call bookmark_moment?  │
  │ - What coaching advice?  │
  └──────────────────────────┘
```

### Detection Logic Details

#### Face Expressions (via Blendshapes)
- **SMILE:** When `smile_score > 60%` for current frame
- **SURPRISE:** When `surprise_score > 50%` (mouth open, brows raised)
- **WINK:** When `wink_score > 60%` (one eye closed)
- **GOOD_FRAMING:** When face is centered (`face_center == 1.0`)
- **State Tracking:** Debounced to max 2.0s between state changes to avoid spamming

#### Hand Gestures (via MediaPipe GestureRecognizer)
- **THUMBS_UP** → Detected when GestureRecognizer confidence > 0.9
- **PEACE_SIGN** → V-shape hand gesture
- **WAVE** → Hand waving motion
- **POINT** → Pointing_Up gesture
- **Debouncing:** 
  - Gesture must change from last detected gesture
  - Must wait 0.8s before firing new gesture event
  - Resets after 0.3s of no gesture detected
  - Prevents repeated "thumbs up" spam from single gesture

#### Body Poses (via MediaPipe Pose Landmarks)
- **ARMS_UP:** Both arms raised above shoulders (`arms_up_score > 0.5`)
- **T_POSE:** Arms horizontal at shoulder level (`t_pose_score > 0.7`)
- **HANDS_ON_HIPS:** Both hands near hips/waist (`hands_on_hips_score > 0.7`)
- **LEAN_IN:** Shoulder Y-position forward (`lean_in_score > 0.6`)
- **State Tracking:** Each pose tracked independently; state changes trigger [SYSTEM EVENT]

### Data Sent to Gemini

When a moment is detected, ClientSession sends:
```python
# Example [SYSTEM EVENT] message to Gemini
"[SYSTEM EVENT] THUMBS_UP detected with confidence 0.95"
"[SYSTEM EVENT] SMILE state changed: now True (score: 62%)"
"[SYSTEM EVENT] ARMS_UP detected (score: 0.75)"
```

**Plus:** Raw video frame (4 FPS) + raw audio stream (16kHz continuous)

### Gemini's Response

Gemini receives all the data and decides:
- Is this moment important given the user's goal?
- Should I bookmark it? (calls `bookmark_moment` tool)
- Should I set an overlay? (calls `set_overlay` tool)
- What coaching should I give based on their performance?

### Session Summary

At end of session, `MomentDetector.get_best_moments()` returns:
- **Top N moments** ranked by `weighted_score`
- **Session summary** with `best_moments` array
- **Suggested clip** timeframe (best performing segment)

---

## Current State

- ✅ WebSocket streaming working
- ✅ Gemini Live API connected
- ✅ Audio coaching returns
- ✅ **Moment detection fully integrated** (Phase 1-5 complete)
- ✅ Face expressions detected (smile, surprise, wink, good_framing)
- ✅ Hand gestures recognized (thumbs_up, peace, wave, point)
- ✅ Body poses detected (arms_up, t_pose, hands_on_hips, lean_in)
- ✅ Moments sent to Gemini as [SYSTEM EVENT] hints
- ✅ Gemini calling bookmark_moment + set_overlay tools
- ✅ MediaPipe models loading correctly
- ⚠️ Thresholds may need tuning based on real-world testing
- ❌ Frontend UI for visualizing detected moments
- ❌ Session summary export/persistence
