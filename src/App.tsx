import { useRef, useCallback } from 'react'
import { useSnapture } from './hooks/useSnapture'
import { VideoPreview } from './components/VideoPreview'
import { StatusIndicator } from './components/StatusIndicator'
import { Controls } from './components/Controls'
import { TranscriptPanel } from './components/TranscriptPanel'
import { Sparkles } from 'lucide-react'

function App() {
  const {
    connectionState,
    isStreaming,
    transcript,
    connect,
    disconnect,
    startStreaming,
    stopStreaming,
  } = useSnapture()

  const videoRef = useRef<HTMLVideoElement>(null)

  const handleStartStreaming = useCallback(() => {
    if (videoRef.current) {
      startStreaming(videoRef.current)
    }
  }, [startStreaming])

  return (
    <div className="min-h-screen bg-[var(--ctp-base)] p-6">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[var(--ctp-mauve)] to-[var(--ctp-pink)] flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-[var(--ctp-base)]" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-[var(--ctp-text)]">Snapture</h1>
            <p className="text-sm text-[var(--ctp-subtext0)]">AI-powered video coaching</p>
          </div>
        </div>
        <StatusIndicator state={connectionState} />
      </header>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-180px)]">
        {/* Video preview - takes 2 columns on large screens */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <VideoPreview
            ref={videoRef}
            className="flex-1 min-h-[400px]"
          />

          {/* Controls */}
          <Controls
            connectionState={connectionState}
            isStreaming={isStreaming}
            onConnect={connect}
            onDisconnect={disconnect}
            onStartStreaming={handleStartStreaming}
            onStopStreaming={stopStreaming}
            className="justify-center"
          />
        </div>

        {/* Transcript panel */}
        <TranscriptPanel
          messages={transcript}
          className="min-h-[400px]"
        />
      </div>
    </div>
  )
}

export default App
