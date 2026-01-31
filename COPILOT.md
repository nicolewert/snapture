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

## Current State

- ✅ WebSocket streaming working
- ✅ Gemini Live API connected
- ✅ Audio coaching returns
- ⚠️ `moment_detector.py` exists but NOT integrated
- ⚠️ Face/gesture models present but unused
- ❌ No persona selection
- ❌ No moment events to frontend
