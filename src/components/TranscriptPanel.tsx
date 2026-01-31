import { cn } from '../lib/utils'
import { MessageSquare } from 'lucide-react'

interface TranscriptPanelProps {
    messages: string[]
    className?: string
}

export function TranscriptPanel({ messages, className }: TranscriptPanelProps) {
    return (
        <div className={cn(
            'glass rounded-2xl p-5',
            'flex flex-col h-full',
            className
        )}>
            <div className="flex items-center gap-2 mb-5">
                <MessageSquare className="w-5 h-5 text-[var(--ctp-mauve)]" />
                <h3 className="font-semibold text-[var(--ctp-text)]">Transcript</h3>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3">
                {messages.length === 0 ? (
                    <p className="text-[var(--ctp-overlay1)] text-sm italic">
                        Waiting for conversation...
                    </p>
                ) : (
                    messages.map((msg, i) => (
                        <div
                            key={i}
                            className="p-3 rounded-lg bg-[var(--ctp-surface0)] text-[var(--ctp-subtext1)] text-sm"
                        >
                            {msg}
                        </div>
                    ))
                )}
            </div>
        </div>
    )
}
