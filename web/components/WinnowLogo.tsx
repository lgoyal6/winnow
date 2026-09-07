// The "Winnow-W" mark: a stylized W where the inner V converges to a focal
// dot (the kept signal) and the outer strokes are dim with scattered
// particles drifting away (the chaff). `mono` strips the color accents.

type Props = { className?: string; mono?: boolean };

export function WinnowLogo({ className, mono = false }: Props) {
  // Color palette flips between the brand neon (mint kept + pink chaff)
  // and a clean black-and-white treatment where everything is white with
  // a subtle opacity hierarchy.
  const keep = mono ? "#ffffff" : "#36f1a3";
  const keepGrad = mono ? "#ffffff" : "#6ee7ff";
  const chaff = mono ? "#ffffff" : "#ff5fb1";

  return (
    <svg viewBox="0 0 32 32" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id={`winnow-glow-${mono ? "m" : "c"}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={keep} stopOpacity={mono ? 0.55 : 0.9} />
          <stop offset="60%" stopColor={keep} stopOpacity={mono ? 0.15 : 0.25} />
          <stop offset="100%" stopColor={keep} stopOpacity="0" />
        </radialGradient>
        <linearGradient id={`winnow-inner-${mono ? "m" : "c"}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={keepGrad} stopOpacity={mono ? 0.85 : 1} />
          <stop offset="100%" stopColor={keep} />
        </linearGradient>
      </defs>

      {/* Outer chaff strokes — leaning outward. */}
      <path d="M3.5 5 L8 26" stroke={chaff} strokeWidth="1.4" strokeOpacity={mono ? 0.3 : 0.45} strokeLinecap="round" />
      <path d="M28.5 5 L24 26" stroke={chaff} strokeWidth="1.4" strokeOpacity={mono ? 0.3 : 0.45} strokeLinecap="round" />

      {/* Chaff particles drifting outward. */}
      <circle cx="2" cy="11" r="0.7" fill={chaff} opacity={mono ? 0.4 : 0.55} />
      <circle cx="4.5" cy="21" r="0.55" fill={chaff} opacity={mono ? 0.3 : 0.4} />
      <circle cx="30" cy="11" r="0.7" fill={chaff} opacity={mono ? 0.4 : 0.55} />
      <circle cx="27.5" cy="21" r="0.55" fill={chaff} opacity={mono ? 0.3 : 0.4} />

      {/* Halo + kept-signal dot at the V apex. */}
      <circle cx="16" cy="26.5" r="5" fill={`url(#winnow-glow-${mono ? "m" : "c"})`} />

      {/* Inner V — kept signal converging. */}
      <path
        d="M10.5 5 L16 24 L21.5 5"
        stroke={`url(#winnow-inner-${mono ? "m" : "c"})`}
        strokeWidth="2.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <circle cx="16" cy="26.5" r="2" fill={keep} />
    </svg>
  );
}
