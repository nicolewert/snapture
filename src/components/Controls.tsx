import { cn } from '../lib/utils'
import type { ConnectionState } from '../lib/websocket'
import { Video, VideoOff } from 'lucide-react'

interface ControlsProps {
    connectionState: ConnectionState
    isStreaming: boolean
    onStartStreaming: () => void
    onStopStreaming: () => void
    className?: string
}

export function Controls({
    connectionState,
    isStreaming,
    onStartStreaming,
    onStopStreaming,
    className,
}: ControlsProps) {
    const isConnected = connectionState === 'connected'

    return (
        <div className={cn('flex items-center gap-4', className)}>
            {/* Streaming button */}
            <button
                onClick={isStreaming ? onStopStreaming : onStartStreaming}
                disabled={!isConnected}
                className={cn(
                    'flex items-center gap-2 px-6 py-3 rounded-xl font-medium',
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
        </div>
    )
}
