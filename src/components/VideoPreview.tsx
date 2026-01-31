import { forwardRef } from 'react'
import { cn } from '../lib/utils'

interface VideoPreviewProps {
    className?: string
}

export const VideoPreview = forwardRef<HTMLVideoElement, VideoPreviewProps>(
    ({ className }, ref) => {
        return (
            <div className={cn(
                'relative overflow-hidden rounded-2xl',
                'bg-[var(--ctp-surface0)] border border-[var(--ctp-surface1)]',
                className
            )}>
                <video
                    ref={ref}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                />
                {/* Recording indicator */}
                <div className="absolute top-5 left-5 flex items-center gap-2.5">
                    <div className="w-3 h-3 rounded-full bg-[var(--ctp-red)] animate-pulse" />
                    <span className="text-xs font-medium text-[var(--ctp-text)] opacity-80">
                        LIVE
                    </span>
                </div>
            </div>
        )
    }
)

VideoPreview.displayName = 'VideoPreview'
