// Media capture utilities for camera, microphone, and audio resampling

const TARGET_SAMPLE_RATE = 16000 // 16kHz for Gemini Live API

export interface MediaConfig {
    video: boolean
    audio: boolean
    videoConstraints?: MediaTrackConstraints
    audioConstraints?: MediaTrackConstraints
}

export class MediaCapture {
    private stream: MediaStream | null = null
    private audioContext: AudioContext | null = null
    private audioProcessor: ScriptProcessorNode | null = null
    private videoCanvas: HTMLCanvasElement | null = null
    private videoContext: CanvasRenderingContext2D | null = null
    private videoElement: HTMLVideoElement | null = null
    private videoInterval: number | null = null

    private onAudioChunk: ((base64: string) => void) | null = null
    private onVideoFrame: ((base64: string) => void) | null = null

    async start(config: MediaConfig): Promise<MediaStream> {
        const constraints: MediaStreamConstraints = {
            video: config.video ? (config.videoConstraints || {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 4, max: 8 } // Low frame rate for Live API
            }) : false,
            audio: config.audio ? (config.audioConstraints || {
                sampleRate: { ideal: 48000 },
                channelCount: { exact: 1 },
                echoCancellation: true,
                noiseSuppression: true,
            }) : false,
        }

        this.stream = await navigator.mediaDevices.getUserMedia(constraints)
        return this.stream
    }

    getStream(): MediaStream | null {
        return this.stream
    }

    startAudioCapture(onChunk: (base64: string) => void) {
        if (!this.stream) {
            console.error('[Media] No stream available for audio capture')
            return
        }

        this.onAudioChunk = onChunk
        const audioTrack = this.stream.getAudioTracks()[0]
        if (!audioTrack) {
            console.error('[Media] No audio track available')
            return
        }

        // Create audio context for resampling
        this.audioContext = new AudioContext({ sampleRate: 48000 })
        const source = this.audioContext.createMediaStreamSource(this.stream)

        // Buffer size for ~40ms at 48kHz = 1920 samples, use 2048 for power of 2
        const bufferSize = 2048
        this.audioProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1)

        this.audioProcessor.onaudioprocess = (event) => {
            const inputData = event.inputBuffer.getChannelData(0)
            // Resample from source sample rate to 16kHz
            const resampledData = this.resampleAudio(inputData, this.audioContext!.sampleRate, TARGET_SAMPLE_RATE)
            const pcm16 = this.floatTo16BitPCM(resampledData)
            const base64 = this.arrayBufferToBase64(pcm16.buffer as ArrayBuffer)
            this.onAudioChunk?.(base64)
        }

        source.connect(this.audioProcessor)
        this.audioProcessor.connect(this.audioContext.destination)
        console.log('[Media] Audio capture started')
    }

    startVideoCapture(videoEl: HTMLVideoElement, onFrame: (base64: string) => void, fps: number = 4) {
        this.onVideoFrame = onFrame
        this.videoElement = videoEl
        let isProcessing = false

        // Create canvas for frame capture
        this.videoCanvas = document.createElement('canvas')
        this.videoContext = this.videoCanvas.getContext('2d')

        const intervalMs = 1000 / fps
        this.videoInterval = window.setInterval(async () => {
            if (!this.videoElement || !this.videoContext || !this.videoCanvas || isProcessing) return

            // Update canvas size to match video resolution if needed
            if (this.videoCanvas.width !== this.videoElement.videoWidth ||
                this.videoCanvas.height !== this.videoElement.videoHeight) {
                this.videoCanvas.width = this.videoElement.videoWidth
                this.videoCanvas.height = this.videoElement.videoHeight
            }

            if (this.videoCanvas.width === 0 || this.videoCanvas.height === 0) return

            isProcessing = true
            try {
                this.videoContext.drawImage(
                    this.videoElement,
                    0, 0,
                    this.videoCanvas.width,
                    this.videoCanvas.height
                )

                // Use toBlob instead of toDataURL to avoid blocking the main thread
                this.videoCanvas.toBlob(async (blob) => {
                    if (!blob) {
                        isProcessing = false
                        return
                    }

                    try {
                        const buffer = await blob.arrayBuffer()
                        const base64 = this.arrayBufferToBase64(buffer)
                        this.onVideoFrame?.(base64)
                    } catch (e) {
                        console.error('[Media] Frame conversion error:', e)
                    } finally {
                        isProcessing = false
                    }
                }, 'image/jpeg', 0.8) // Reduced quality slightly for performance
            } catch (e) {
                console.error('[Media] Capture error:', e)
                isProcessing = false
            }
        }, intervalMs)

        console.log(`[Media] Video capture started at ${fps} fps`)
    }

    stop() {
        // Stop audio processing
        if (this.audioProcessor) {
            this.audioProcessor.disconnect()
            this.audioProcessor = null
        }
        if (this.audioContext) {
            this.audioContext.close()
            this.audioContext = null
        }

        // Stop video interval
        if (this.videoInterval) {
            clearInterval(this.videoInterval)
            this.videoInterval = null
        }

        // Stop all tracks
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop())
            this.stream = null
        }

        console.log('[Media] Capture stopped')
    }

    private resampleAudio(inputData: Float32Array, inputSampleRate: number, outputSampleRate: number): Float32Array {
        if (inputSampleRate === outputSampleRate) {
            return inputData
        }

        const ratio = inputSampleRate / outputSampleRate
        const outputLength = Math.floor(inputData.length / ratio)
        const output = new Float32Array(outputLength)

        for (let i = 0; i < outputLength; i++) {
            const inputIndex = i * ratio
            const low = Math.floor(inputIndex)
            const high = Math.ceil(inputIndex)
            const fraction = inputIndex - low

            if (high >= inputData.length) {
                output[i] = inputData[low]
            } else {
                // Linear interpolation
                output[i] = inputData[low] * (1 - fraction) + inputData[high] * fraction
            }
        }

        return output
    }

    private floatTo16BitPCM(float32Array: Float32Array): Int16Array {
        const int16Array = new Int16Array(float32Array.length)
        for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]))
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
        }
        return int16Array
    }

    private arrayBufferToBase64(buffer: ArrayBuffer): string {
        const bytes = new Uint8Array(buffer)
        let binary = ''
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i])
        }
        return btoa(binary)
    }
}

// Audio playback for model responses
export class AudioPlayer {
    private audioContext: AudioContext | null = null
    private audioQueue: AudioBuffer[] = []
    private isPlaying = false
    private currentSource: AudioBufferSourceNode | null = null

    async init() {
        if (this.audioContext) {
            // Resume if suspended (common after page load without user interaction)
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume()
                console.log('[AudioPlayer] Resumed suspended audio context')
            }
            return
        }
        this.audioContext = new AudioContext({ sampleRate: 24000 }) // Gemini outputs 24kHz
        console.log('[AudioPlayer] Initialized audio context, state:', this.audioContext.state)
    }

    async playAudio(base64Data: string) {
        if (!this.audioContext) await this.init()
        
        // Ensure context is running (may be suspended on first interaction)
        if (this.audioContext!.state === 'suspended') {
            await this.audioContext!.resume()
        }

        const pcmData = this.base64ToInt16Array(base64Data)
        const floatData = this.int16ToFloat32(pcmData)

        console.log('[AudioPlayer] Received audio chunk:', pcmData.length, 'samples')
        
        const audioBuffer = this.audioContext!.createBuffer(1, floatData.length, 24000)
        audioBuffer.getChannelData(0).set(floatData)

        this.audioQueue.push(audioBuffer)
        if (!this.isPlaying) {
            this.playNext()
        }
    }

    private playNext() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false
            return
        }

        this.isPlaying = true
        const buffer = this.audioQueue.shift()!
        this.currentSource = this.audioContext!.createBufferSource()
        this.currentSource.buffer = buffer
        this.currentSource.connect(this.audioContext!.destination)
        this.currentSource.onended = () => this.playNext()
        this.currentSource.start()
    }

    interrupt() {
        if (this.currentSource) {
            this.currentSource.stop()
            this.currentSource = null
        }
        this.audioQueue = []
        this.isPlaying = false
    }

    private base64ToInt16Array(base64: string): Int16Array {
        const binary = atob(base64)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i)
        }
        return new Int16Array(bytes.buffer)
    }

    private int16ToFloat32(int16Array: Int16Array): Float32Array {
        const float32Array = new Float32Array(int16Array.length)
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / (int16Array[i] < 0 ? 0x8000 : 0x7FFF)
        }
        return float32Array
    }
}
