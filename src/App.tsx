import { useRef, useCallback } from 'react'
import { useSnapture } from './hooks/useSnapture'
import { VideoPreview } from './components/VideoPreview'
import { StatusIndicator } from './components/StatusIndicator'
import { Controls } from './components/Controls'
import { TranscriptPanel } from './components/TranscriptPanel'
import logoImg from '/public/logo.png'

function App() {
  const {
    connectionState,
    isStreaming,
    transcript,
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
          <img src={logoImg} alt="Snapture" className="w-20 h-20 object-contain" />
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
