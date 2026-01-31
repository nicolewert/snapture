import type { ConnectionState } from '../lib/websocket'
import { cn } from '../lib/utils'
import { Wifi, WifiOff, Loader2, AlertCircle } from 'lucide-react'

interface StatusIndicatorProps {
    state: ConnectionState
    className?: string
}

const stateConfig: Record<ConnectionState, {
    icon: typeof Wifi
    label: string
    color: string
    animate?: boolean
}> = {
    disconnected: {
        icon: WifiOff,
        label: 'Disconnected',
        color: 'text-[var(--ctp-overlay1)]',
    },
    connecting: {
        icon: Loader2,
        label: 'Connecting...',
        color: 'text-[var(--ctp-yellow)]',
        animate: true,
    },
    connected: {
        icon: Wifi,
        label: 'Connected',
        color: 'text-[var(--ctp-green)]',
    },
    error: {
        icon: AlertCircle,
        label: 'Error',
        color: 'text-[var(--ctp-red)]',
    },
}

export function StatusIndicator({ state, className }: StatusIndicatorProps) {
    const config = stateConfig[state]
    const Icon = config.icon

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
