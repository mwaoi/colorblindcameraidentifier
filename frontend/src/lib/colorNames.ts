// Nearest-neighbor color lookup in Oklab space.
// 949 XKCD colors pre-converted at module init — O(n) linear scan is
// fast enough for 949 entries (~0.05ms per query in V8).

import XKCD from '../data/xkcdColors'
import { hexToOklab, rgbToOklab } from './oklab'

interface ColorEntry {
  name: string
  hex: string
  L: number
  a: number
  b: number
}

// Build DB once at module init
const DB: ColorEntry[] = Object.entries(XKCD).map(([name, hex]) => {
  const [L, a, b] = hexToOklab(hex)
  return { name, hex, L, a, b }
})

export interface ColorResult {
  name: string
  hex: string
}

export function findNearestColor(r: number, g: number, b: number): ColorResult {
  const [qL, qa, qb] = rgbToOklab(r, g, b)
  let bestDist = Infinity
  let best = DB[0]
  for (const entry of DB) {
    const dL = qL - entry.L
    const da = qa - entry.a
    const db = qb - entry.b
    const dist = dL * dL + da * da + db * db
    if (dist < bestDist) {
      bestDist = dist
      best = entry
    }
  }
  return { name: best.name, hex: best.hex }
}

// Highlight-robust mean — mirrors _highlight_robust_mean() in color_detector.py.
// Excludes specular highlights (very bright + near-colorless) before computing mean.
export function highlightRobustMean(data: Uint8ClampedArray): { r: number; g: number; b: number } {
  const n = data.length / 4
  let rSum = 0, gSum = 0, bSum = 0
  let included = 0

  const mask = new Uint8Array(n)
  let highlightCount = 0

  for (let i = 0; i < n; i++) {
    const r = data[i * 4]
    const g = data[i * 4 + 1]
    const b = data[i * 4 + 2]

    // HSV V and S (approximate — same thresholds as Python: V > 200, S < 40)
    const max = Math.max(r, g, b)
    const min = Math.min(r, g, b)
    const v = max
    const s = max === 0 ? 0 : ((max - min) / max) * 255

    if (v > 200 && s < 40) {
      highlightCount++
    } else {
      mask[i] = 1
    }
  }

  // If >80% masked (legitimately white object), use all pixels
  const useAll = highlightCount / n > 0.8

  for (let i = 0; i < n; i++) {
    if (useAll || mask[i]) {
      rSum += data[i * 4]
      gSum += data[i * 4 + 1]
      bSum += data[i * 4 + 2]
      included++
    }
  }

  const count = included || 1
  return {
    r: Math.round(rSum / count),
    g: Math.round(gSum / count),
    b: Math.round(bSum / count),
  }
}
