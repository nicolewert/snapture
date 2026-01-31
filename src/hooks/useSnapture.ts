import { useState, useEffect, useCallback, useRef } from 'react'
import { wsClient } from '../lib/websocket'
import type { ConnectionState, ServerMessage } from '../lib/websocket'
import { MediaCapture, AudioPlayer } from '../lib/media'

export function useSnapture() {
    const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected')
    const [isStreaming, setIsStreaming] = useState(false)
    const [lastMessage, setLastMessage] = useState<ServerMessage | null>(null)
    const [transcript, setTranscript] = useState<string[]>([])

    const mediaCapture = useRef<MediaCapture>(new MediaCapture())
    const audioPlayer = useRef<AudioPlayer>(new AudioPlayer())
    const videoRef = useRef<HTMLVideoElement | null>(null)

    useEffect(() => {
        const unsubState = wsClient.onStateChange(setConnectionState)
        const unsubMessage = wsClient.onMessage((message) => {
            setLastMessage(message)

            switch (message.type) {
                case 'audio':
                    audioPlayer.current.playAudio(message.data)
                    break
                case 'text':
                    setTranscript(prev => [...prev, message.content])
                    break
                case 'interrupted':
                    audioPlayer.current.interrupt()
                    break
                case 'error':
                    console.error('[Snapture] Error:', message.message)
                    break
            }
        })

        return () => {
            unsubState()
            unsubMessage()
        }
    }, [])

    const connect = useCallback(async () => {
        try {
            await wsClient.connect()
            wsClient.sendSetup('gemini-2.5-flash-native-audio-latest')
        } catch (error) {
            console.error('[Snapture] Failed to connect:', error)
        }
    }, [])

    const disconnect = useCallback(() => {
        stopStreaming()
        wsClient.disconnect()
    }, [])

    const startStreaming = useCallback(async (video: HTMLVideoElement) => {
        videoRef.current = video

        try {
            const stream = await mediaCapture.current.start({ video: true, audio: true })
            video.srcObject = stream
            await video.play()

            // Start capturing and sending to backend
            mediaCapture.current.startAudioCapture((base64) => {
                wsClient.sendAudio(base64)
            })

            mediaCapture.current.startVideoCapture(video, (base64) => {
                wsClient.sendVideo(base64)
            }, 4) // 4 fps

            setIsStreaming(true)
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
    }, [])

    const clearTranscript = useCallback(() => {
        setTranscript([])
    }, [])

    return {
        connectionState,
        isStreaming,
        lastMessage,
        transcript,
        connect,
        disconnect,
        startStreaming,
        stopStreaming,
        clearTranscript,
    }
}
