import { cn } from '../lib/utils'
import type { ConnectionState } from '../lib/websocket'
import { Video, VideoOff, Scissors } from 'lucide-react'

interface ControlsProps {
    connectionState: ConnectionState
    isStreaming: boolean
    isClipping: boolean
    onStartStreaming: () => void
    onStopStreaming: () => void
    onStartClip: () => void
    onStopClip: () => void
    className?: string
}

export function Controls({
    connectionState,
    isStreaming,
    isClipping,
    onStartStreaming,
    onStopStreaming,
    onStartClip,
    onStopClip,
    className,
}: ControlsProps) {
    const isConnected = connectionState === 'connected'

    return (
        <div className={cn('flex items-center gap-6', className)}>
            {/* Streaming button */}
            <button
                onClick={isStreaming ? onStopStreaming : onStartStreaming}
                disabled={!isConnected}
                className={cn(
                    'flex items-center justify-center gap-3 min-w-[180px] px-8 py-4 rounded-xl font-medium text-base',
                    'transition-all duration-200 ease-out',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    isStreaming
                        ? 'bg-[var(--ctp-red)] text-[var(--ctp-base)] hover:opacity-90 glow-red'
                        : 'bg-[var(--ctp-green)] text-[var(--ctp-base)] hover:opacity-90 glow-green'
                )}
            >
                {isStreaming ? (
                    <>
                        <VideoOff className="w-5 h-5" />
                        Stop
                    </>
                ) : (
                    <>
                        <Video className="w-5 h-5" />
                        Start Recording
                    </>
                )}
            </button>

            {/* Clip button */}
            <button
                onClick={isClipping ? onStopClip : onStartClip}
                disabled={!isStreaming}
                className={cn(
                    'flex items-center justify-center gap-3 min-w-[180px] px-8 py-4 rounded-xl font-medium text-base',
                    'transition-all duration-200 ease-out',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    isClipping
                        ? 'bg-[var(--ctp-peach)] text-[var(--ctp-base)] hover:opacity-90 animate-pulse'
                        : 'bg-[var(--ctp-blue)] text-[var(--ctp-base)] hover:opacity-90 glow-blue'
                )}
            >
                <Scissors className="w-5 h-5" />
                {isClipping ? 'Stop Clip' : 'Start Clip'}
            </button>
        </div>
    )
}
