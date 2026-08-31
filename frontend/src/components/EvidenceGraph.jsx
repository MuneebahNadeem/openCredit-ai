import { SOURCE_TYPE_META, prettyName } from "../lib/format";

/* Lightweight SVG evidence map:
   sources (left) → evidence fields (middle) → the two assessments (right).
   Lines are drawn from real data only — counts come from the result. */
export default function EvidenceGraph({ result }) {
  const sources = (result.sources_detail || []).filter(
    (s) => s.evidence_count > 0
  );
  const evidence = result.evidence || [];
  const assessments = [
    { key: "trust", label: "Trustworthiness", level: result.trustworthiness?.level },
    {
      key: "potential",
      label: "Business potential",
      level: result.business_potential?.level,
    },
  ];
  if (sources.length === 0 || evidence.length === 0) return null;

  // Layout constants.
  const W = 720;
  const H = Math.max(320, 70 + Math.max(sources.length, evidence.length) * 52);
  const colX = { src: 120, ev: 360, out: 560 };
  const rowY = (i, n) => 40 + (i + 0.5) * ((H - 80) / Math.max(n, 1));

  const srcY = sources.map((_, i) => rowY(i, sources.length));
  const evY = evidence.map((_, i) => rowY(i, evidence.length));
  const outY = assessments.map((_, i) => rowY(i, assessments.length) + 40);

  // Link evidence to the source it came from.
  const srcIndexByUrl = new Map(sources.map((s, i) => [s.url, i]));
  const evLinks = evidence.map((ev) => {
    const url = ev.source_url ? String(ev.source_url) : null;
    return url !== null && srcIndexByUrl.has(url)
      ? srcIndexByUrl.get(url)
      : null; // self-reported / no URL → no line
  });

  return (
    <div className="evidence-graph-wrap" role="img" aria-label="Evidence map linking sources to evidence and assessments">
      <svg viewBox={`0 0 ${W} ${H}`} className="evidence-graph">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(17,82,83,0.35)" />
          </marker>
        </defs>
        <text x={colX.src} y={18} textAnchor="middle" className="eg-col-title">SOURCES</text>
        <text x={colX.ev} y={18} textAnchor="middle" className="eg-col-title">EVIDENCE</text>
        <text x={colX.out} y={18} textAnchor="middle" className="eg-col-title">ASSESSMENTS</text>

        {evLinks.map((srcIdx, i) =>
          srcIdx === null ? null : (
            <path
              key={`l-${i}`}
              d={`M ${colX.src + 66} ${srcY[srcIdx]} C ${colX.src + 130} ${srcY[srcIdx]}, ${colX.ev - 130} ${evY[i]}, ${colX.ev - 96} ${evY[i]}`}
              fill="none"
              stroke="rgba(17,82,83,0.25)"
              strokeWidth="1.4"
              markerEnd="url(#arrow)"
            />
          )
        )}
        {assessments.map((a, i) => (
          <path
            key={`lo-${i}`}
            d={`M ${colX.ev + 96} ${evY[Math.floor(evidence.length / 2)]} C ${colX.ev + 170} ${evY[Math.floor(evidence.length / 2)]}, ${colX.out - 170} ${outY[i]}, ${colX.out - 86} ${outY[i]}`}
            fill="none"
            stroke="rgba(17,82,83,0.25)"
            strokeWidth="1.6"
            markerEnd="url(#arrow)"
          />
        ))}

        {sources.map((s, i) => {
          const meta = SOURCE_TYPE_META[s.type] || SOURCE_TYPE_META.web;
          return (
            <g key={s.url} transform={`translate(${colX.src - 66}, ${srcY[i] - 14})`}>
              <rect width="132" height="28" rx="14" className={`eg-node eg-src`} />
              <text x="14" y="18" className="eg-icon">{meta.icon}</text>
              <text x="30" y="18" className="eg-label">{truncate(s.name || s.url, 15)}</text>
            </g>
          );
        })}

        {evidence.map((ev, i) => (
          <g key={`${ev.field_name}-${i}`} transform={`translate(${colX.ev - 96}, ${evY[i] - 14})`}>
            <rect
              width="192"
              height="28"
              rx="14"
              className={`eg-node eg-ev ev-${ev.evidence_type}`}
            />
            <text x="14" y="18" className="eg-label">
              {truncate(prettyName(ev.field_name), 22)}
            </text>
          </g>
        ))}

        {assessments.map((a, i) => (
          <g key={a.key} transform={`translate(${colX.out - 86}, ${outY[i] - 16})`}>
            <rect width="172" height="32" rx="16" className={`eg-node eg-out level-fill-${a.level}`} />
            <text x="86" y="20" textAnchor="middle" className="eg-out-label">
              {a.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function truncate(text, n) {
  return text.length > n ? `${text.slice(0, n - 1)}…` : text;
}
