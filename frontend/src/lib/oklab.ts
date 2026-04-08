// Oklab color space conversion — ported from oklab_namer.py
// Björn Ottosson (2020): https://bottosson.github.io/posts/oklab/
//
// Pipeline: uint8 RGB → sRGB → linear RGB → XYZ (D65) → LMS → LMS^(1/3) → Oklab

type Vec3 = [number, number, number]
type Mat3 = [Vec3, Vec3, Vec3]

const M_RGB_TO_XYZ: Mat3 = [
  [0.4124564, 0.3575761, 0.1804375],
  [0.2126729, 0.7151522, 0.0721750],
  [0.0193339, 0.1191920, 0.9503041],
]

const M_XYZ_TO_LMS: Mat3 = [
  [ 0.8189330101,  0.3618667424, -0.1288597137],
  [ 0.0329845436,  0.9293118715,  0.0361456387],
  [ 0.0482003018,  0.2643662691,  0.6338517070],
]

const M_LMS_TO_LAB: Mat3 = [
  [0.2104542553,  0.7936177850, -0.0040720468],
  [1.9779984951, -2.4285922050,  0.4505937099],
  [0.0259040371,  0.7827717662, -0.8086757660],
]

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

function mulMat3Vec3(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ]
}

export function rgbToOklab(r: number, g: number, b: number): Vec3 {
  const linear: Vec3 = [
    srgbToLinear(r / 255),
    srgbToLinear(g / 255),
    srgbToLinear(b / 255),
  ]
  const xyz = mulMat3Vec3(M_RGB_TO_XYZ, linear)
  const lms = mulMat3Vec3(M_XYZ_TO_LMS, xyz)
  const lms_: Vec3 = [Math.cbrt(lms[0]), Math.cbrt(lms[1]), Math.cbrt(lms[2])]
  return mulMat3Vec3(M_LMS_TO_LAB, lms_)
}

export function hexToOklab(hex: string): Vec3 {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return rgbToOklab(r, g, b)
}
