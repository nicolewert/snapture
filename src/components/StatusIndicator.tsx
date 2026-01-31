import type { ConnectionState } from '../lib/websocket'
import { cn } from '../lib/utils'
import { Loader2, AlertCircle } from 'lucide-react'

interface StatusIndicatorProps {
    state: ConnectionState
    className?: string
}

const stateConfig: Record<ConnectionState, {
    icon: typeof Loader2
    label: string
    color: string
    dotColor: string
    animate?: boolean
}> = {
    disconnected: {
        icon: Loader2,
        label: 'Reconnecting...',
        color: 'text-[var(--ctp-yellow)]',
        dotColor: 'bg-[var(--ctp-yellow)]',
        animate: true,
    },
    connecting: {
        icon: Loader2,
        label: 'Connecting...',
        color: 'text-[var(--ctp-yellow)]',
        dotColor: 'bg-[var(--ctp-yellow)]',
        animate: true,
    },
    connected: {
        icon: Loader2,
        label: 'Connected',
        color: 'text-[var(--ctp-green)]',
        dotColor: 'bg-[var(--ctp-green)]',
    },
    error: {
        icon: AlertCircle,
        label: 'Connection Error',
        color: 'text-[var(--ctp-red)]',
        dotColor: 'bg-[var(--ctp-red)]',
    },
}

export function StatusIndicator({ state, className }: StatusIndicatorProps) {
    const config = stateConfig[state]
    const Icon = config.icon

    // When connected, show minimal green dot only
    if (state === 'connected') {
        return (
            <div className={cn('flex items-center gap-2', className)}>
                <div className="w-2 h-2 rounded-full bg-[var(--ctp-green)]" title="Connected" />
            </div>
        )
    }

    // For connecting/error states, show full indicator with text
    return (
        <div className={cn('flex items-center gap-2', className)}>
            <Icon
                className={cn(
                    'w-4 h-4',
                    config.color,
                    config.animate && 'animate-spin'
                )}
            />
            <span className={cn('text-sm font-medium', config.color)}>
                {config.label}
            </span>
        </div>
    )
}
