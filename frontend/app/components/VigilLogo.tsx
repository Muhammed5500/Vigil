"use client";

export default function VigilLogo({ size = 40 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Connections between honest nodes */}
      <line x1="30" y1="25" x2="70" y2="25" stroke="#22c55e" strokeWidth="2" opacity="0.5" />
      <line x1="30" y1="25" x2="50" y2="65" stroke="#22c55e" strokeWidth="2" opacity="0.5" />
      <line x1="70" y1="25" x2="50" y2="65" stroke="#22c55e" strokeWidth="2" opacity="0.5" />

      {/* Broken connection to cheater */}
      <line x1="50" y1="65" x2="80" y2="80" stroke="#ef4444" strokeWidth="2" strokeDasharray="4 3" opacity="0.4" />

      {/* Honest nodes (green) */}
      <circle cx="30" cy="25" r="8" fill="#22c55e" opacity="0.9" />
      <circle cx="70" cy="25" r="8" fill="#22c55e" opacity="0.9" />
      <circle cx="50" cy="65" r="8" fill="#22c55e" opacity="0.9" />

      {/* Cheater node (red) with X */}
      <circle cx="80" cy="80" r="8" fill="#ef4444" opacity="0.7" />
      <line x1="75" y1="75" x2="85" y2="85" stroke="#0a0a0a" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="85" y1="75" x2="75" y2="85" stroke="#0a0a0a" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}
