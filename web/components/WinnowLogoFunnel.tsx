// The "Compression Funnel" mark: scattered input particles up top, five
// lines fanning into a single focal point at the bottom. The center line
// is the kept stream; the outer lines fade. `mono` strips color accents.

type Props = { className?: string; mono?: boolean };

export function WinnowLogoFunnel({ className, mono = false }: Props) {
  const keep = mono ? "#ffffff" : "#36f1a3";
  const mid = mono ? "#ffffff" : "#6ee7ff";
  const chaff = mono ? "#ffffff" : "#ff5fb1";

  return (
    <svg viewBox="0 0 32 32" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id={`funnel-glow-${mono ? "m" : "c"}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={keep} stopOpacity={mono ? 0.55 : 0.9} />
          <stop offset="60%" stopColor={keep} stopOpacity={mono ? 0.15 : 0.25} />
          <stop offset="100%" stopColor={keep} stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Input particles across the top. */}
      <circle cx="4" cy="4" r="0.7" fill={chaff} opacity={mono ? 0.35 : 0.45} />
      <circle cx="10" cy="3" r="0.5" fill={chaff} opacity={mono ? 0.45 : 0.55} />
      <circle cx="16" cy="3.5" r="0.6" fill={mid} opacity={mono ? 0.6 : 0.7} />
      <circle cx="22" cy="3" r="0.5" fill={chaff} opacity={mono ? 0.45 : 0.55} />
      <circle cx="28" cy="4" r="0.7" fill={chaff} opacity={mono ? 0.35 : 0.45} />

      {/* Outer chaff lines. */}
      <path d="M4 6 L16 24" stroke={chaff} strokeWidth="1.3" strokeOpacity={mono ? 0.3 : 0.4} strokeLinecap="round" />
      <path d="M28 6 L16 24" stroke={chaff} strokeWidth="1.3" strokeOpacity={mono ? 0.3 : 0.4} strokeLinecap="round" />

      {/* Mid lines — in-between signal. */}
      <path d="M10 6 L16 24" stroke={mid} strokeWidth="1.5" strokeOpacity={mono ? 0.55 : 0.7} strokeLinecap="round" />
      <path d="M22 6 L16 24" stroke={mid} strokeWidth="1.5" strokeOpacity={mono ? 0.55 : 0.7} strokeLinecap="round" />

      {/* Center line — kept signal. */}
      <path d="M16 5 L16 24" stroke={keep} strokeWidth="2.4" strokeLinecap="round" />

      <circle cx="16" cy="26" r="5" fill={`url(#funnel-glow-${mono ? "m" : "c"})`} />
      <circle cx="16" cy="26" r="2.1" fill={keep} />
    </svg>
  );
}
