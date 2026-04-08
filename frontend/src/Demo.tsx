import { useCallback, useEffect, useRef, useState } from 'react'
import { findNearestColor, highlightRobustMean, type ColorResult } from './lib/colorNames'

const RETICLE_SIZE = 160

type CameraState = 'idle' | 'requesting' | 'active' | 'denied' | 'unavailable'

export default function Demo() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const [cameraState, setCameraState] = useState<CameraState>('idle')
  const [result, setResult] = useState<ColorResult | null>(null)
  const [pulse, setPulse] = useState(false) // flash the reticle on identify

  // Stop stream when component unmounts
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [])

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('unavailable')
      return
    }
    setCameraState('requesting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      })
      streamRef.current = stream
      const video = videoRef.current
      if (video) {
        video.srcObject = stream
        // Wait for metadata (videoWidth/videoHeight available) before playing
        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => resolve()
        })
        await video.play()
      }
      setCameraState('active')
    } catch {
      setCameraState('denied')
    }
  }

  const identify = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    // Guard: video must have loaded dimensions
    if (!video || !canvas || cameraState !== 'active' || !video.videoWidth) return

    // Draw current frame to hidden canvas — no mirroring needed for color sampling,
    // CSS scaleX(-1) is display-only and doesn't affect center pixel values
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(video, 0, 0)

    // Sample the 160×160 center region
    const cx = Math.floor(canvas.width / 2)
    const cy = Math.floor(canvas.height / 2)
    const half = RETICLE_SIZE / 2
    const imageData = ctx.getImageData(
      Math.max(0, cx - half),
      Math.max(0, cy - half),
      RETICLE_SIZE,
      RETICLE_SIZE,
    )

    const { r, g, b } = highlightRobustMean(imageData.data)
    const found = findNearestColor(r, g, b)
    setResult(found)

    // Reticle flash
    setPulse(true)
    setTimeout(() => setPulse(false), 180)

    // Speak via Web Speech API
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(found.name))
    }
  }, [cameraState])

  // Space key → identify (only when camera is active and focus isn't on an input)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Space' && cameraState === 'active' && !(e.target instanceof HTMLInputElement)) {
        e.preventDefault()
        identify()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cameraState, identify])

  // ── idle / error states ──────────────────────────────────────────────────

  if (cameraState === 'idle') {
    return (
      <div className="flex flex-col items-start gap-5">
        <p className="text-zinc-400 text-sm max-w-sm">
          runs the same Oklab k-NN pipeline as the desktop app — 949 XKCD colors,
          specular highlight rejection, Web Speech API output. no backend.
        </p>
        <button
          onClick={startCamera}
          className="border border-zinc-700 px-5 py-2.5 text-sm text-zinc-300 hover:border-green-600 hover:text-white transition-colors duration-150 font-mono"
        >
          start camera
        </button>
      </div>
    )
  }

  if (cameraState === 'requesting') {
    return (
      <div className="text-zinc-600 text-sm font-mono animate-pulse">
        requesting camera access...
      </div>
    )
  }

  if (cameraState === 'denied') {
    return (
      <p className="text-zinc-500 text-sm font-mono">getUserMedia: permission denied</p>
    )
  }

  if (cameraState === 'unavailable') {
    return (
      <p className="text-zinc-500 text-sm font-mono">getUserMedia unavailable</p>
    )
  }

  // ── active camera ────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-5 max-w-xl">
      {/* Camera feed with reticle — fixed aspect ratio prevents zero-height collapse */}
      <div className="relative bg-zinc-900 overflow-hidden border border-zinc-800" style={{ aspectRatio: '4/3' }}>
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
          autoPlay
          playsInline
          muted
        />
        {/* Green reticle — matches cv2.rectangle in color_detector.py */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div
            className="transition-all duration-75"
            style={{
              width: RETICLE_SIZE,
              height: RETICLE_SIZE,
              border: `1px solid ${pulse ? '#ffffff' : '#00ff00'}`,
              boxShadow: pulse ? '0 0 12px #00ff0066' : '0 0 6px #00ff0022',
            }}
          />
        </div>
      </div>

      {/* Hidden canvas for pixel sampling */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Controls row */}
      <div className="flex items-center gap-4 flex-wrap">
        <button
          onClick={identify}
          className="border border-zinc-700 px-5 py-2 text-sm font-mono text-zinc-300 hover:border-green-600 hover:text-white transition-colors duration-150 active:scale-95"
        >
          identify
        </button>
      </div>

      {/* Result */}
      {result && (
        <div className="flex items-center gap-4">
          <div
            className="w-12 h-12 border border-white/10 shrink-0"
            style={{ backgroundColor: result.hex }}
          />
          <div>
            <p className="text-white text-xl font-medium lowercase tracking-tight">
              {result.name}
            </p>
            <p className="text-zinc-600 text-xs font-mono">{result.hex}</p>
          </div>
        </div>
      )}

      <p className="text-zinc-700 text-xs">
        voice readout via Web Speech API · color matched to XKCD dataset (949 colors) in Oklab space
      </p>
    </div>
  )
}
