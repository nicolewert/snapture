import { useState, useEffect, useCallback, useRef } from 'react'
import { wsClient } from '../lib/websocket'
import type { ConnectionState, ServerMessage } from '../lib/websocket'
import { MediaCapture, AudioPlayer } from '../lib/media'

export function useSnapture() {
    const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
    const [isStreaming, setIsStreaming] = useState(false)
    const [isClipping, setIsClipping] = useState(false)
    const [lastMessage, setLastMessage] = useState<ServerMessage | null>(null)
    const [transcript, setTranscript] = useState<string[]>([])

    const [clips, setClips] = useState<{ url: string; context: string; timestamp: number }[]>([])

    const mediaCapture = useRef<MediaCapture>(new MediaCapture())
    const audioPlayer = useRef<AudioPlayer>(new AudioPlayer())
    const videoRef = useRef<HTMLVideoElement | null>(null)
    const reconnectAttemptsRef = useRef(0)
    const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    // Auto-connect on mount
    useEffect(() => {
        const autoConnect = async () => {
            try {
                await wsClient.connect()
                wsClient.sendSetup('gemini-2.5-flash-native-audio-latest')
                reconnectAttemptsRef.current = 0
            } catch (error) {
                console.error('[Snapture] Failed to auto-connect:', error)
                scheduleReconnect()
            }
        }

        const scheduleReconnect = () => {
            const attempts = reconnectAttemptsRef.current
            const delayMs = Math.min(1000 * Math.pow(2, attempts), 30000) // exponential backoff, max 30s
            reconnectAttemptsRef.current = attempts + 1

            console.log(`[Snapture] Scheduling reconnect in ${delayMs}ms (attempt ${attempts + 1})`)
            reconnectTimeoutRef.current = setTimeout(autoConnect, delayMs)
        }

        autoConnect()

        const unsubState = wsClient.onStateChange((state) => {
            setConnectionState(state)
            // Clear any pending reconnect if we successfully connected
            if (state === 'connected' && reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current)
                reconnectTimeoutRef.current = null
            }
        })

        const unsubMessage = wsClient.onMessage((message) => {
            setLastMessage(message)

            switch (message.type) {
                case 'audio':
                    audioPlayer.current.playAudio(message.data)
                    break
                case 'text':
                    setTranscript(prev => [...prev, message.content])
                    break
                case 'clip':
                    setClips(prev => [...prev, {
                        url: message.url,
                        context: message.context,
                        timestamp: Date.now()
                    }])
                    break
                case 'interrupted':
                    audioPlayer.current.interrupt()
                    break
                case 'error':
                    console.error('[Snapture] Error:', message.message)
                    break
            }
        })

        // Handle page unload - disconnect cleanly
        const handleBeforeUnload = () => {
            stopStreaming()
            wsClient.disconnect()
            console.log('[Snapture] Page unload - disconnected')
        }

        // Handle tab visibility change
        const handleVisibilityChange = () => {
            if (document.hidden) {
                console.log('[Snapture] Tab hidden - stopping stream')
                stopStreaming() // Stop streaming but keep connection alive
            }
        }

        window.addEventListener('beforeunload', handleBeforeUnload)
        document.addEventListener('visibilitychange', handleVisibilityChange)

        return () => {
            unsubState()
            unsubMessage()
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current)
            }
            window.removeEventListener('beforeunload', handleBeforeUnload)
            document.removeEventListener('visibilitychange', handleVisibilityChange)
        }
    }, [])

    const startStreaming = useCallback(async (video: HTMLVideoElement) => {
        videoRef.current = video

        try {
            // Initialize audio player early (requires user gesture)
            await audioPlayer.current.init()
            console.log('[Snapture] Audio player initialized')
            
            const stream = await mediaCapture.current.start({
                video: true,
                audio: true,
                videoConstraints: {
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 },
                    frameRate: { ideal: 30, max: 60 }
                }
            })
            video.srcObject = stream
            await video.play()

            // Start capturing and sending to backend
            mediaCapture.current.startAudioCapture((base64) => {
                wsClient.sendAudio(base64)
            })

            mediaCapture.current.startVideoCapture(video, (base64) => {
                wsClient.sendVideo(base64)
            }, 24) // 24 fps for clip buffer (server forwards every 2nd to Gemini)

            setIsStreaming(true)
            wsClient.sendStartRecording()
        } catch (error) {
            console.error('[Snapture] Failed to start streaming:', error)
        }
    }, [])

    const stopStreaming = useCallback(() => {
        mediaCapture.current.stop()
        if (videoRef.current) {
            videoRef.current.srcObject = null
        }
        setIsStreaming(false)
        setIsClipping(false)
        wsClient.sendStopRecording()
    }, [])

    const startClip = useCallback(() => {
        wsClient.sendStartClip()
        setIsClipping(true)
    }, [])

    const stopClip = useCallback(() => {
        wsClient.sendStopClip()
        setIsClipping(false)
    }, [])

    const clearTranscript = useCallback(() => {
        setTranscript([])
    }, [])

    return {
        connectionState,
        isStreaming,
        isClipping,
        lastMessage,
        transcript,
        clips,
        startStreaming,
        stopStreaming,
        startClip,
        stopClip,
        clearTranscript,
    }
}
