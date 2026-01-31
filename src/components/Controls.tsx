import { cn } from '../lib/utils'
import type { ConnectionState } from '../lib/websocket'
import { Video, VideoOff, Power, PowerOff } from 'lucide-react'

interface ControlsProps {
    connectionState: ConnectionState
    isStreaming: boolean
    onConnect: () => void
    onDisconnect: () => void
    onStartStreaming: () => void
    onStopStreaming: () => void
    className?: string
}

export function Controls({
    connectionState,
    isStreaming,
    onConnect,
    onDisconnect,
    onStartStreaming,
    onStopStreaming,
    className,
}: ControlsProps) {
    const isConnected = connectionState === 'connected'
    const isConnecting = connectionState === 'connecting'

    return (
        <div className={cn('flex items-center gap-4', className)}>
            {/* Connection button */}
            <button
                onClick={isConnected ? onDisconnect : onConnect}
                disabled={isConnecting}
                className={cn(
                    'flex items-center gap-2 px-6 py-3 rounded-xl font-medium',
                    'transition-all duration-200 ease-out',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    isConnected
                        ? 'bg-[var(--ctp-surface1)] text-[var(--ctp-text)] hover:bg-[var(--ctp-surface2)]'
                        : 'bg-[var(--ctp-mauve)] text-[var(--ctp-base)] hover:opacity-90 glow-mauve'
                )}
            >
                {isConnected ? (
                    <>
                        <PowerOff className="w-5 h-5" />
                        Disconnect
                    </>
                ) : (
                    <>
                        <Power className="w-5 h-5" />
                        {isConnecting ? 'Connecting...' : 'Connect'}
                    </>
                )}
            </button>

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
                        Start Streaming
                    </>
                )}
            </button>
        </div>
    )
}
