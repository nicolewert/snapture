import { cn } from '../lib/utils'
import { Film, Download } from 'lucide-react'

interface Clip {
    url: string
    context: string
    timestamp: number
}

interface ClipsPanelProps {
    clips: Clip[]
    className?: string
}

export function ClipsPanel({ clips, className }: ClipsPanelProps) {
    return (
        <div className={cn(
            'glass rounded-2xl p-4',
            'flex flex-col h-full',
            className
        )}>
            <div className="flex items-center gap-2 mb-4">
                <Film className="w-5 h-5 text-[var(--ctp-mauve)]" />
                <h3 className="font-semibold text-[var(--ctp-text)]">Clips</h3>
            </div>

            <div className="flex-1 overflow-y-auto space-y-4">
                {clips.length === 0 ? (
                    <p className="text-[var(--ctp-overlay1)] text-sm italic">
                        No clips generated yet...
                    </p>
                ) : (
                    clips.map((clip, i) => (
                        <div
                            key={i}
                            className="p-3 rounded-lg bg-[var(--ctp-surface0)] space-y-2"
                        >
                            <p className="text-[var(--ctp-subtext1)] text-sm font-medium">
                                {clip.context || "Generated Clip"}
                            </p>
                            <video
                                src={clip.url}
                                controls
                                className="w-full rounded-md bg-black"
                            />
                            <div className="flex justify-end">
                                <a
                                    href={clip.url}
                                    download={`clip-${clip.timestamp}.mp4`}
                                    className="flex items-center gap-2 text-xs text-[var(--ctp-blue)] hover:text-[var(--ctp-sky)] transition-colors"
                                    target="_blank"
                                    rel="noreferrer"
                                >
                                    <Download className="w-3 h-3" />
                                    Download Clip
                                </a>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    )
}
