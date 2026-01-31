import { useRef, useCallback, useState } from 'react'
import { useSnapture } from './hooks/useSnapture'
import { VideoPreview } from './components/VideoPreview'
import { StatusIndicator } from './components/StatusIndicator'
import { Controls } from './components/Controls'
import { TranscriptPanel } from './components/TranscriptPanel'
import { ClipsPanel } from './components/ClipsPanel'

const logoImg = '/logo.png'

function App() {
  const {
    connectionState,
    isStreaming,
    isClipping,
    transcript,
    clips,
    startStreaming,
    stopStreaming,
    startClip,
    stopClip,
  } = useSnapture()

  const videoRef = useRef<HTMLVideoElement>(null)
  const [activeTab, setActiveTab] = useState<'transcript' | 'clips'>('transcript')

  const handleStartStreaming = useCallback(() => {
    if (videoRef.current) {
      startStreaming(videoRef.current)
    }
  }, [startStreaming])

  return (
    <div className="min-h-screen bg-[var(--ctp-base)] overflow-x-hidden p-8 md:p-12 lg:p-16">
        {/* Header */}
        <header className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <img src={logoImg} alt="Snapture" className="w-20 h-20 object-contain" />
          </div>
          <StatusIndicator state={connectionState} />
        </header>

        {/* Main content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-10 h-[calc(100vh-200px)]">
        {/* Video preview - takes 2 columns on large screens */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <VideoPreview
            ref={videoRef}
            className="flex-1 min-h-[400px]"
          />

          {/* Controls */}
          <Controls
            connectionState={connectionState}
            isStreaming={isStreaming}
            isClipping={isClipping}
            onStartStreaming={handleStartStreaming}
            onStopStreaming={stopStreaming}
            onStartClip={startClip}
            onStopClip={stopClip}
            className="justify-center"
          />
        </div>

        {/* Right Panel with Tabs */}
        <div className="flex flex-col gap-5 min-h-[400px]">
          {/* Tab Matcher */}
          <div className="glass p-1 rounded-xl flex gap-1">
            <button
              onClick={() => setActiveTab('transcript')}
              className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${activeTab === 'transcript'
                ? 'bg-[var(--ctp-surface0)] text-[var(--ctp-text)] shadow-sm'
                : 'text-[var(--ctp-subtext0)] hover:text-[var(--ctp-text)] hover:bg-[var(--ctp-surface0)]/50'
                }`}
            >
              Transcript
            </button>
            <button
              onClick={() => setActiveTab('clips')}
              className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${activeTab === 'clips'
                ? 'bg-[var(--ctp-surface0)] text-[var(--ctp-text)] shadow-sm'
                : 'text-[var(--ctp-subtext0)] hover:text-[var(--ctp-text)] hover:bg-[var(--ctp-surface0)]/50'
                }`}
            >
              Clips
              {clips.length > 0 && (
                <span className="ml-2 px-1.5 py-0.5 text-[10px] bg-[var(--ctp-pink)] text-[var(--ctp-base)] rounded-full">
                  {clips.length}
                </span>
              )}
            </button>
          </div>

          <div className="flex-1 overflow-hidden relative">
            {activeTab === 'transcript' ? (
              <TranscriptPanel
                messages={transcript}
                className="h-full absolute inset-0"
              />
            ) : (
              <ClipsPanel
                clips={clips}
                className="h-full absolute inset-0"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
