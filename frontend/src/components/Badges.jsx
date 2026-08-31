import { titleCase, RECOMMENDATION_META } from "../lib/format";

export function LevelBadge({ level }) {
  if (!level) return null;
  return (
    <span className={`badge badge-level badge-${level}`}>
      <span className="badge-dot" aria-hidden="true" />
      {titleCase(level)}
    </span>
  );
}

export function RecommendationBadge({ recommendation }) {
  if (!recommendation) return null;
  const meta = RECOMMENDATION_META[recommendation] || {
    label: titleCase(recommendation),
  };
  return (
    <span
      className={`badge badge-recommendation rec-${recommendation}`}
      title={meta.description}
    >
      {meta.label}
    </span>
  );
}

export function TypeBadge({ value }) {
  if (!value) return null;
  const cls = {
    corroborated: "type-corroborated",
    observed: "type-observed",
    inference: "type-inference",
    unknown: "type-unknown",
  }[value] || "type-unknown";
  return <span className={`badge badge-type ${cls}`}>{titleCase(value)}</span>;
}

export function ReliabilityBadge({ value }) {
  if (!value) return null;
  return (
    <span className={`badge badge-reliability rel-${value}`}>
      {titleCase(value)} reliability
    </span>
  );
}
