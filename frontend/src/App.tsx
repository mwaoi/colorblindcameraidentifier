import { useState } from 'react'

// colors that the old HSV pipeline called "forest green"
// hover to reveal their actual oklab names
const PROBLEM_SWATCHES = [
  { hex: '#6e750e', correct: 'olive' },
  { hex: '#696112', correct: 'greenish brown' },
  { hex: '#5b5c37', correct: 'army green' },
  { hex: '#677a04', correct: 'olive green' },
  { hex: '#3d9114', correct: 'grass green' },
  { hex: '#06470c', correct: 'forest green' },
]

// approximate monk skin tone scale hex values, annotated with ITA range
const MONK_TONES = [
  { hex: '#f6ede4', ita: '> 55°', name: 'very light skin' },
  { hex: '#f0dfc6', ita: '41–55°', name: 'light skin' },
  { hex: '#e2c89c', ita: '28–40°', name: 'medium light skin' },
  { hex: '#c9a87c', ita: '10–27°', name: 'medium skin' },
  { hex: '#a07850', ita: '−10–9°', name: 'medium brown skin' },
  { hex: '#87614c', ita: '−30 to −11°', name: 'brown skin' },
  { hex: '#6a4a3c', ita: '−50 to −31°', name: 'dark brown skin' },
  { hex: '#3d2314', ita: '< −50°', name: 'very dark skin' },
]

const STACK = [
  ['opencv-python', 'camera feed, image processing, color space conversion'],
  ['ultralytics', 'YOLOv8n — detects objects to route identification'],
  ['scipy', 'cKDTree for O(log n) nearest-neighbor color lookup'],
  ['anthropic', 'Claude Haiku Vision API — second opinion, teacher'],
  ['pyttsx3', 'offline text-to-speech, no API required'],
  ['numpy', 'vectorized oklab conversion math'],
]

export default function App() {
  const [hovered, setHovered] = useState<number | null>(null)
  const [monkHovered, setMonkHovered] = useState<number | null>(null)

  return (
    <div className="min-h-screen bg-[#080808] text-[#c8c8c8]" style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>

      {/* ─── hero ─── */}
      <section className="px-6 sm:px-12 pt-24 pb-20 max-w-4xl mx-auto">
        <p className="font-mono text-xs text-zinc-600 mb-10 tracking-widest uppercase">
          python · opencv · oklab · yolov8 · claude
        </p>
        <h1 className="text-[clamp(3rem,10vw,7rem)] font-bold tracking-tight text-white leading-[0.9] lowercase mb-10">
          colorblind<br />
          <span className="text-zinc-500">camera</span><br />
          identifier.
        </h1>
        <div className="max-w-md">
          <p className="text-xl text-zinc-400 leading-relaxed">
            i'm severely colorblind. every morning i'd reach for what i thought
            was a blue shirt and it would be purple. so i built something that could tell me.
          </p>
        </div>
      </section>

      {/* ─── app mockup + what it does ─── */}
      <section className="px-6 sm:px-12 pb-24 max-w-4xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-start">

          {/* fake app window */}
          <div className="border border-zinc-800 overflow-hidden select-none">
            <div className="flex items-center gap-1.5 px-3 py-2.5 bg-zinc-950 border-b border-zinc-800">
              <div className="w-2.5 h-2.5 rounded-full bg-zinc-800 border border-zinc-700" />
              <div className="w-2.5 h-2.5 rounded-full bg-zinc-800 border border-zinc-700" />
              <div className="w-2.5 h-2.5 rounded-full bg-zinc-800 border border-zinc-700" />
              <span className="ml-2 text-[10px] font-mono text-zinc-700 truncate">
                Color Identifier&nbsp;&nbsp;|&nbsp;&nbsp;SPACE = identify&nbsp;&nbsp;|&nbsp;&nbsp;Y = correct&nbsp;&nbsp;|&nbsp;&nbsp;N = wrong
              </span>
            </div>

            {/* fake camera view */}
            <div className="relative bg-zinc-900 overflow-hidden" style={{ aspectRatio: '4/3' }}>
              {/* simulated blurry room */}
              <div className="absolute inset-0">
                <div className="absolute inset-0 bg-gradient-to-br from-zinc-800/60 via-zinc-900 to-zinc-950" />
                <div className="absolute top-1/4 left-1/3 w-32 h-40 bg-[#5b5c37]/20 blur-2xl rounded-full" />
                <div className="absolute bottom-1/3 right-1/4 w-24 h-24 bg-zinc-700/10 blur-xl rounded-full" />
              </div>

              {/* reticle — matches exact cv2 rectangle from color_detector.py */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div
                  className="border border-[#00ff00]"
                  style={{ width: 80, height: 80, boxShadow: '0 0 8px #00ff0022' }}
                />
              </div>

              {/* result overlay — matches _draw_text_overlay */}
              <div className="absolute bottom-3 left-3">
                <div className="bg-black/60 px-2 py-1">
                  <span className="text-white text-sm font-mono tracking-wide">army green</span>
                </div>
              </div>

              {/* volume / rate trackbars (decorative) */}
              <div className="absolute top-3 right-3 space-y-1.5 opacity-40">
                <div className="flex items-center gap-2">
                  <span className="text-[9px] font-mono text-zinc-400">VOL</span>
                  <div className="w-16 h-1 bg-zinc-700 rounded">
                    <div className="w-3/4 h-full bg-zinc-400 rounded" />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] font-mono text-zinc-400">RATE</span>
                  <div className="w-16 h-1 bg-zinc-700 rounded">
                    <div className="w-1/2 h-full bg-zinc-400 rounded" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* description */}
          <div className="space-y-6 pt-2">
            <p className="text-zinc-300 leading-relaxed">
              press <Key>space</Key> while pointing the camera at anything.
              it identifies the color and says it out loud.
            </p>
            <p className="text-zinc-400 leading-relaxed">
              clothes, walls, food, skin — whatever's in the reticle.
              the green box is 160×160 pixels. results come back in about 150ms.
            </p>
            <p className="text-zinc-500 leading-relaxed text-sm">
              a second thread simultaneously asks Claude Vision API.
              if it disagrees with the local result, it overrides and speaks again.
              that disagreement is stored as a correction — the model learns from it.
            </p>
            <p className="text-zinc-600 text-sm">
              works without internet. claude is optional.
            </p>
          </div>
        </div>
      </section>

      <Divider label="how it works" />

      {/* ─── pipeline ─── */}
      <section className="px-6 sm:px-12 py-20 max-w-4xl mx-auto">
        <p className="text-zinc-500 text-sm mb-10 max-w-lg">
          two threads start the moment you press space. the local one finishes first.
          the claude one catches up a second later and corrects if needed.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <PipelineCard
            color="green"
            label="local"
            timing="~100–200ms"
            steps={[
              'YOLOv8n detects the object in frame',
              'routes: person (≥0.65 conf) → skin detector',
              'skin: ITA formula on CIELAB → Monk scale name',
              'everything else: Oklab k-NN over 954 XKCD colors',
              'spoken immediately, no internet needed',
            ]}
          />
          <PipelineCard
            color="amber"
            label="claude haiku"
            timing="~1–2s"
            steps={[
              'frame encoded as base64 JPEG',
              'YOLO class injected: "I\'ve detected a \'cup\'..."',
              'structured prompt: object: ... / color: ...',
              'if result differs from local: override + speak',
              'correction stored in oklab memory database',
            ]}
          />
        </div>

        <p className="text-zinc-700 text-xs font-mono mt-6">
          stale results from prior space presses are discarded via press_id counter
        </p>
      </section>

      <Divider label="the forest green problem" />

      {/* ─── oklab ─── */}
      <section className="px-6 sm:px-12 py-20 max-w-4xl mx-auto">
        <div className="max-w-lg mb-10">
          <p className="text-zinc-400 leading-relaxed mb-4">
            the first version used HSV. it called my olive shirt "forest green".
            every single time. i'd put on what i thought was a safe neutral and end up
            wearing something that apparently clashed badly.
          </p>
          <p className="text-zinc-400 leading-relaxed mb-4">
            the problem: HSV is not perceptually uniform. equal HSV distances don't correspond
            to equal visual differences. olive and forest green sit close together in HSV
            even though they look nothing alike.
          </p>
          <p className="text-zinc-300 leading-relaxed">
            <span className="text-white font-medium">Oklab</span> (Björn Ottosson, 2020) is
            designed so euclidean distance = perceptual difference. these six colors all
            returned "forest green" before. hover to see what oklab calls them:
          </p>
        </div>

        {/* problem swatches */}
        <div className="flex flex-wrap gap-3 mb-3">
          {PROBLEM_SWATCHES.map((s, i) => (
            <div
              key={i}
              className="cursor-default group"
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
            >
              <div
                className="w-16 h-16 sm:w-20 sm:h-20 border border-white/5 transition-all duration-150"
                style={{
                  backgroundColor: s.hex,
                  outline: hovered === i ? '1px solid #ffffff22' : 'none',
                  outlineOffset: 2,
                }}
              />
              <div className="mt-1.5 text-xs font-mono text-center transition-colors duration-150"
                style={{ color: hovered === i ? '#e2e2e2' : '#444' }}>
                {hovered === i ? s.correct : 'forest green'}
              </div>
            </div>
          ))}
        </div>
        <p className="text-zinc-700 text-xs font-mono mb-10">hover each swatch</p>

        {/* conversion pipeline */}
        <div className="bg-[#0d0d0d] border border-zinc-800/60 p-5 font-mono text-xs overflow-x-auto">
          <p className="text-zinc-600 mb-3"># oklab conversion — runs once per identification</p>
          <div className="space-y-1 text-zinc-500">
            <p><span className="text-zinc-700">1.</span> srgb = rgb_uint8 / 255.0</p>
            <p><span className="text-zinc-700">2.</span> linear = srgb/12.92 <span className="text-zinc-700">if</span> srgb ≤ 0.04045 <span className="text-zinc-700">else</span> ((srgb + 0.055) / 1.055)^2.4</p>
            <p><span className="text-zinc-700">3.</span> xyz = M_RGB_XYZ @ linear  <span className="text-zinc-700 ml-4"># D65 illuminant</span></p>
            <p><span className="text-zinc-700">4.</span> lms = M_XYZ_LMS @ xyz   <span className="text-zinc-700 ml-4"># Ottosson M1</span></p>
            <p><span className="text-zinc-700">5.</span> lab = M_LMS_LAB @ cbrt(lms)  <span className="text-zinc-700 ml-2"># Ottosson M2</span></p>
            <p className="mt-3"><span className="text-zinc-700">6.</span> <span className="text-green-600">cKDTree</span>.query(lab) → nearest of 954 XKCD colors</p>
          </div>
        </div>
      </section>

      <Divider label="skin detection" />

      {/* ─── skin detection ─── */}
      <section className="px-6 sm:px-12 py-20 max-w-4xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-start">
          <div>
            <p className="text-zinc-400 leading-relaxed mb-5">
              when YOLO returns "person" at ≥65% confidence with ≥40% overlap of
              the center region, the frame routes to the skin pipeline instead of color naming.
            </p>
            <p className="text-zinc-400 leading-relaxed mb-5">
              tone is classified using the <span className="text-white">ITA formula</span> (Individual
              Typology Angle) on CIELAB — a medically validated 8-level scale:
            </p>

            <div className="bg-[#0d0d0d] border border-zinc-800/60 px-4 py-3 font-mono text-sm mb-5">
              <span className="text-zinc-300">ITA</span>
              <span className="text-zinc-600"> = </span>
              <span className="text-zinc-400">arctan2(L* − 50, b*)</span>
              <span className="text-zinc-600"> × </span>
              <span className="text-zinc-400">180/π</span>
            </div>

            <p className="text-zinc-500 text-sm leading-relaxed">
              OpenCV's COLOR_BGR2Lab outputs L in [0,255], not [0,100].
              rescaling is required before applying the formula.
              arctan2 avoids divide-by-zero when b* ≈ 0.
            </p>
          </div>

          {/* monk scale */}
          <div>
            <div className="flex gap-1.5 mb-3">
              {MONK_TONES.map((t, i) => (
                <div
                  key={i}
                  className="flex-1 cursor-default"
                  onMouseEnter={() => setMonkHovered(i)}
                  onMouseLeave={() => setMonkHovered(null)}
                >
                  <div
                    className="w-full transition-all duration-100 border border-white/5"
                    style={{
                      backgroundColor: t.hex,
                      height: monkHovered === i ? 72 : 56,
                    }}
                  />
                </div>
              ))}
            </div>

            <div className="text-[9px] font-mono text-zinc-700 leading-relaxed min-h-[2.5rem]">
              {monkHovered !== null ? (
                <>
                  <span className="text-zinc-400">{MONK_TONES[monkHovered].name}</span>
                  <br />
                  {MONK_TONES[monkHovered].ita}
                </>
              ) : (
                <span>hover to see tone name + ITA range</span>
              )}
            </div>

            <div className="mt-6 border border-zinc-800/60 p-4 text-sm text-zinc-500 space-y-2">
              <p className="text-zinc-600 font-mono text-xs mb-3"># is_skin_region() — three masks must agree</p>
              <p>HSV: hue 0–25°, saturation ≥28, value ≤215</p>
              <p>YCbCr: Cr 138–173, Cb 77–127</p>
              <p>brightness gate: V ≤ 215 (excludes white fabric)</p>
              <p className="text-zinc-600">all three agree on ≥22% of ROI pixels → skin</p>
            </div>
          </div>
        </div>
      </section>

      <Divider label="it learns" />

      {/* ─── memory ─── */}
      <section className="px-6 sm:px-12 py-20 max-w-4xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-start">
          <div>
            <p className="text-zinc-400 leading-relaxed mb-5">
              every correction is stored in <span className="font-mono text-zinc-300 text-sm">~/.colorblindcam/memory.json</span> —
              a k-NN database in Oklab space. claude acts as the automatic teacher.
              you act as the override.
            </p>
            <p className="text-zinc-400 leading-relaxed mb-8">
              a sample needs to be seen twice before it's trusted. after that, it skips
              the full pipeline entirely and returns directly from memory.
            </p>

            <div className="space-y-2">
              {[
                { key: 'space', label: 'identify', note: 'runs full pipeline' },
                { key: 'Y', label: 'correct', note: 'reinforce — +1 count', accent: '#22c55e' },
                { key: 'N', label: 'wrong', note: 'reject this region permanently', accent: '#ef4444' },
                { key: 'Q', label: 'quit', note: '' },
              ].map(({ key, label, note, accent }) => (
                <div key={key} className="flex items-center gap-4">
                  <kbd
                    className="border border-zinc-700 bg-zinc-900 px-2.5 py-1 text-xs font-mono min-w-[44px] text-center"
                    style={{ color: accent || '#a1a1aa' }}
                  >
                    {key}
                  </kbd>
                  <span className="text-zinc-300 text-sm">{label}</span>
                  {note && <span className="text-zinc-600 text-xs">{note}</span>}
                </div>
              ))}
            </div>
          </div>

          {/* fake terminal log */}
          <div className="bg-[#0a0a0a] border border-zinc-800/60 p-5 font-mono text-xs space-y-1 text-zinc-500 leading-relaxed">
            <p className="text-zinc-700 mb-3"># example — same shirt, two mornings</p>
            <p className="text-zinc-600">─── day 1 ───────────────────────</p>
            <p>space pressed</p>
            <p>→ YOLO: person (conf 0.43) — below 0.65 threshold</p>
            <p>→ rerouted to general</p>
            <p>→ oklab: <span className="text-green-600">olive</span></p>
            <p className="text-zinc-600">→ claude: <span className="text-amber-500">army green</span> [override]</p>
            <p className="text-zinc-600">→ correction stored (dist: 0.031)</p>
            <p className="mt-3 text-zinc-600">─── day 2 ───────────────────────</p>
            <p>space pressed</p>
            <p>→ <span className="text-green-600">memory hit</span> (count: 2, dist: 0.008)</p>
            <p>→ <span className="text-green-600">army green</span></p>
            <p className="text-zinc-700">→ 12ms<span className="animate-blink">▊</span></p>
          </div>
        </div>

        <div className="mt-10 border-l-2 border-zinc-800 pl-5 text-sm text-zinc-600 max-w-lg">
          <p>pressing N stores a rejection with negative count. that oklab region
          won't match that color again — even across sessions. if it keeps
          calling the ceiling "medium skin", one press of N teaches it permanently.</p>
        </div>
      </section>

      <Divider label="stack" />

      {/* ─── stack ─── */}
      <section className="px-6 sm:px-12 py-20 max-w-4xl mx-auto">
        <div className="max-w-2xl">
          {STACK.map(([pkg, desc]) => (
            <div
              key={pkg}
              className="grid grid-cols-[140px_1fr] gap-4 py-3 border-b border-zinc-900 last:border-0"
            >
              <span className="font-mono text-zinc-300 text-sm">{pkg}</span>
              <span className="text-zinc-600 text-sm">{desc}</span>
            </div>
          ))}
        </div>

        <div className="mt-12 text-xs font-mono text-zinc-700 space-y-1">
          <p>yolov8n.pt (~6MB) — auto-downloads on first run</p>
          <p>ANTHROPIC_API_KEY — optional, set as environment variable</p>
          <p>~/.colorblindcam/memory.json — created automatically</p>
        </div>
      </section>

      {/* ─── footer ─── */}
      <footer className="px-6 sm:px-12 py-14 max-w-4xl mx-auto border-t border-zinc-900">
        <p className="text-zinc-700 text-sm">
          built for getting dressed in the morning.
        </p>
      </footer>

    </div>
  )
}

function Divider({ label }: { label: string }) {
  return (
    <div className="px-6 sm:px-12 max-w-4xl mx-auto">
      <div className="flex items-center gap-4 border-t border-zinc-900 pt-0">
        <span className="text-[10px] font-mono text-zinc-700 uppercase tracking-widest whitespace-nowrap -mt-[9px] bg-[#080808] pr-3">
          {label}
        </span>
      </div>
    </div>
  )
}

function PipelineCard({
  color,
  label,
  timing,
  steps,
}: {
  color: 'green' | 'amber'
  label: string
  timing: string
  steps: string[]
}) {
  const dot = color === 'green' ? 'bg-green-500' : 'bg-amber-500'
  const text = color === 'green' ? 'text-green-500' : 'text-amber-500'

  return (
    <div className="border border-zinc-800 p-5">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full ${dot}`} />
          <span className={`text-sm font-mono ${text}`}>{label}</span>
        </div>
        <span className="text-xs font-mono text-zinc-700">{timing}</span>
      </div>
      <ol className="space-y-2.5">
        {steps.map((step, i) => (
          <li key={i} className="flex gap-3 text-sm text-zinc-500">
            <span className="font-mono text-zinc-700 shrink-0">{i + 1}.</span>
            <span>{step}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

function Key({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="bg-zinc-900 border border-zinc-700 px-1.5 py-0.5 rounded text-xs font-mono text-zinc-300 mx-0.5">
      {children}
    </kbd>
  )
}
