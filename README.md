# Snapture 🎬

Real-time video coaching app powered by Gemini Live API. Create TikTok/Reels-style shorts with AI-powered live feedback.

## Architecture

```
┌─────────────┐      WebSocket      ┌─────────────┐      WebSocket      ┌─────────────┐
│   Frontend  │ ←─────────────────→ │   Backend   │ ←─────────────────→ │ Gemini Live │
│  (React/TS) │    audio/video      │  (FastAPI)  │    audio/video      │     API     │
└─────────────┘                     └─────────────┘                     └─────────────┘
```

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+
- Google API Key with Gemini access

### 1. Backend Setup

```bash
cd server
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
python main.py
```

Server runs on http://localhost:8000

### 2. Frontend Setup

```bash
# From project root
npm install
npm run dev
```

App runs on http://localhost:5173

### 3. Usage

1. Open the app in your browser
2. Click **Connect** to establish WebSocket connection
3. Click **Start Streaming** to share camera/mic
4. Speak to the camera - Gemini will provide real-time coaching!

## Environment Variables

### Frontend (`.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_PORT` | `5173` | Dev server port |
| `VITE_WS_URL` | `ws://localhost:8000/ws` | Backend WebSocket URL |

### Backend (`server/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | - | **Required** - Your Gemini API key |
| `SERVER_PORT` | `8000` | Server port |
| `SERVER_HOST` | `0.0.0.0` | Server host |

## Project Structure

```
snapture/
├── src/
│   ├── components/      # React UI components
│   ├── hooks/           # Custom React hooks
│   ├── lib/             # WebSocket & media utilities
│   └── App.tsx          # Main app component
├── server/
│   ├── main.py          # FastAPI server
│   └── gemini_client.py # Gemini Live API client
└── package.json
```

## Tech Stack

- **Frontend**: React, TypeScript, Tailwind CSS, Catppuccin Mocha theme
- **Backend**: Python, FastAPI, WebSockets
- **AI**: Gemini Live API (gemini-2.0-flash-exp)

## Development Notes

### Audio Format
- Input: 16kHz, 16-bit PCM (resampled from browser's native rate)
- Output: 24kHz, 16-bit PCM (from Gemini)

### Video Format
- JPEG frames at ~4 FPS
- 640x480 resolution

## Future Tracks

**Developer A**: Recording, timestamps, playback, final clip experience  
**Developer B**: Moment detection, tool calling, overlays/bookmarks

Merge point: Events (bookmark, overlay) + take clock
