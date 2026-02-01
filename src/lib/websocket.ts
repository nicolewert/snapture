// WebSocket client for communicating with the Python backend

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error'

export type ClientMessage =
    | { type: 'setup'; config: { model: string } }
    | { type: 'audio'; data: string }
    | { type: 'video'; data: string }
    | { type: 'start_recording' }
    | { type: 'stop_recording' }
    | { type: 'start_clip' }
    | { type: 'stop_clip' }
    | { type: 'end' }

export type ServerMessage =
    | { type: 'connected'; sessionId: string }
    | { type: 'audio'; data: string }
    | { type: 'text'; content: string }
    | { type: 'interrupted' }
    | { type: 'error'; message: string }
    | { type: 'turnComplete' }
    | { type: 'clip'; url: string; context: string }

type MessageHandler = (message: ServerMessage) => void
type StateHandler = (state: ConnectionState) => void

export class WebSocketClient {
    private ws: WebSocket | null = null
    private url: string
    private messageHandlers: Set<MessageHandler> = new Set()
    private stateHandlers: Set<StateHandler> = new Set()
    private _state: ConnectionState = 'disconnected'

    constructor(url: string = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws') {
        this.url = url
    }

    get state(): ConnectionState {
        return this._state
    }

    private setState(state: ConnectionState) {
        this._state = state
        this.stateHandlers.forEach(handler => handler(state))
    }

    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                resolve()
                return
            }

            this.setState('connecting')
            this.ws = new WebSocket(this.url)

            this.ws.onopen = () => {
                console.log('[WS] Connected to server')
                this.setState('connected')
                resolve()
            }

            this.ws.onmessage = (event) => {
                try {
                    const message: ServerMessage = JSON.parse(event.data)
                    console.log('[WS] Received:', message.type)
                    this.messageHandlers.forEach(handler => handler(message))
                } catch (e) {
                    console.error('[WS] Failed to parse message:', e)
                }
            }

            this.ws.onerror = (error) => {
                console.error('[WS] Error:', error)
                this.setState('error')
                reject(error)
            }

            this.ws.onclose = (event) => {
                console.log('[WS] Closed:', event.code, event.reason)
                if (this._state !== 'error') {
                    this.setState('disconnected')
                }
            }
        })
    }

    // Support reconnection after errors
    async reconnect(): Promise<void> {
        console.log('[WS] Attempting to reconnect...')
        this.ws = null
        this.setState('disconnected')
        return this.connect()
    }

    disconnect() {
        if (this.ws) {
            this.send({ type: 'end' })
            this.ws.close()
            this.ws = null
        }
        this.setState('disconnected')
    }

    send(message: ClientMessage) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message))
        } else {
            console.warn('[WS] Cannot send - not connected')
        }
    }

    sendSetup(model: string = 'gemini-3-preview') {
        this.send({ type: 'setup', config: { model } })
    }

    sendAudio(base64Data: string) {
        this.send({ type: 'audio', data: base64Data })
    }

    sendVideo(base64Data: string) {
        this.send({ type: 'video', data: base64Data })
    }

    sendStartRecording() {
        this.send({ type: 'start_recording' })
    }

    sendStopRecording() {
        this.send({ type: 'stop_recording' })
    }

    sendStartClip() {
        this.send({ type: 'start_clip' })
    }

    sendStopClip() {
        this.send({ type: 'stop_clip' })
    }

    onMessage(handler: MessageHandler): () => void {
        this.messageHandlers.add(handler)
        return () => this.messageHandlers.delete(handler)
    }

    onStateChange(handler: StateHandler): () => void {
        this.stateHandlers.add(handler)
        return () => this.stateHandlers.delete(handler)
    }
}

// Singleton instance
export const wsClient = new WebSocketClient()
