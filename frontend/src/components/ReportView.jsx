import LevelGauge from "./LevelGauge";
import SignalCard from "./SignalCard";
import EvidenceTable from "./EvidenceTable";
import SourceList from "./SourceList";
import CredibilityPanel from "./CredibilityPanel";
import SentimentBar from "./SentimentBar";
import EvidenceGraph from "./EvidenceGraph";
import AskPanel from "./AskPanel";
import { LevelBadge, RecommendationBadge } from "./Badges";
import { formatDateTime, titleCase, prettyName } from "../lib/format";

/* The complete investigation report. Renders live API results and the
   labeled demo example identically — one source of truth for the layout. */
export default function ReportView({
  report,
  business,
  demo = false,
  investigationId = null,
}) {
  const trust = report.trustworthiness || {};
  const potential = report.business_potential || {};
  const features = report.features || [];

  return (
    <article className="report">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="report-header">
        <div className="report-header-main">
          {demo && <span className="chip chip-demo">Example — demo data</span>}
          <h1>{business?.name || "Business"}</h1>
          <p className="report-sub">
            {[business?.location, business?.category].filter(Boolean).join(" · ")}
          </p>
          <div className="report-meta">
            <span className="chip">
              Investigated {formatDateTime(report.investigated_at)}
            </span>
            <span className="chip">
              {report.searches_performed ?? 0} searches ·{" "}
              {(report.sources_detail || []).length} sources ·{" "}
              {(report.evidence || []).length} evidence items
            </span>
            {report.agent_status && report.agent_status !== "complete" && (
              <span className="chip">{titleCase(report.agent_status)} run</span>
            )}
          </div>
        </div>
        <div className="report-recommendation">
          <span className="recommendation-label">Recommendation</span>
          <RecommendationBadge recommendation={report.recommendation} />
          <p className="muted recommendation-note">
            Decision support for lenders — not a lending decision.
          </p>
        </div>
      </header>

      {/* ── The two independent assessments ────────────────────── */}
      <section className="report-section" aria-labelledby="assessments-h">
        <div className="section-head">
          <h2 id="assessments-h">The two assessments</h2>
          <p className="muted">
            Kept deliberately separate — a business can be trustworthy and still
            have low potential, or the reverse.
          </p>
        </div>
        <div className="gauge-grid">
          <div className="card card-pad gauge-card">
            <h3>Trustworthiness</h3>
            <LevelGauge
              score={trust.score}
              level={trust.level}
              label="Trustworthiness"
            />
            <p className="gauge-explanation">{trust.explanation}</p>
            <p className="muted gauge-count">
              Backed by {trust.evidence_count ?? (report.evidence || []).length}{" "}
              evidence items
            </p>
          </div>
          <div className="card card-pad gauge-card">
            <h3>Business potential</h3>
            <LevelGauge
              score={potential.score}
              level={potential.level}
              label="Business potential"
            />
            <p className="gauge-explanation">{potential.explanation}</p>
            <p className="muted gauge-count">
              Backed by {potential.evidence_count ?? (report.evidence || []).length}{" "}
              evidence items
            </p>
          </div>
        </div>
        {report.justification && (
          <div className="justification" role="note">
            <span className="justification-label">Summary</span>
            <p>{report.justification}</p>
          </div>
        )}
      </section>

      {/* ── Signals ────────────────────────────────────────────── */}
      <section className="report-section" aria-labelledby="signals-h">
        <div className="section-head">
          <h2 id="signals-h">Signals</h2>
        </div>
        <div className="signals-grid">
          <div className="signals-col">
            <h3 className="signals-title signals-positive-title">
              Positive signals
            </h3>
            {(report.positive_signals || []).length > 0 ? (
              <ul className="signal-list">
                {(report.positive_signals || []).map((s, i) => (
                  <SignalCard key={i} signal={s} kind="positive" />
                ))}
              </ul>
            ) : (
              <p className="empty-note">No positive signals were found.</p>
            )}
          </div>
          <div className="signals-col">
            <h3 className="signals-title signals-risk-title">Risk signals</h3>
            {(report.risk_signals || []).length > 0 ? (
              <ul className="signal-list">
                {(report.risk_signals || []).map((s, i) => (
                  <SignalCard key={i} signal={s} kind="risk" />
                ))}
              </ul>
            ) : (
              <p className="empty-note">No risk signals were found.</p>
            )}
          </div>
        </div>
      </section>

      {/* ── Evidence table ─────────────────────────────────────── */}
      <section className="report-section" aria-labelledby="evidence-h">
        <div className="section-head">
          <h2 id="evidence-h">Evidence</h2>
          <p className="muted">
            Every number in this report traces back to a row below. Inferences
            are never silently promoted to verified facts.
          </p>
        </div>
        <EvidenceTable evidence={report.evidence} />
      </section>

      {/* ── Evidence map ───────────────────────────────────────── */}
      {features.length > 0 || (report.sources_detail || []).length > 0 ? (
        <section className="report-section" aria-labelledby="map-h">
          <div className="section-head">
            <h2 id="map-h">How the evidence connects</h2>
            <p className="muted">
              Sources feed evidence items; evidence feeds the two assessments.
            </p>
          </div>
          <EvidenceGraph result={report} />
        </section>
      ) : null}

      {/* ── What we couldn't find ──────────────────────────────── */}
      {(report.missing_information || []).length > 0 && (
        <section className="report-section" aria-labelledby="missing-h">
          <div className="section-head">
            <h2 id="missing-h">What we couldn’t find</h2>
            <p className="muted">
              Missing information is a gap — not proof of fraud.
            </p>
          </div>
          <ul className="missing-list">
            {(report.missing_information || []).map((item, i) => (
              <li key={i} className="missing-item">
                <span aria-hidden="true">◌</span> {item}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ── Discovered features ────────────────────────────────── */}
      {features.length > 0 && (
        <section className="report-section" aria-labelledby="features-h">
          <div className="section-head">
            <h2 id="features-h">Discovered business features</h2>
          </div>
          <ul className="feature-grid">
            {features.map((f, i) => (
              <li key={i} className="feature-card card">
                <div className="feature-head">
                  <span className="feature-name">{f.name}</span>
                  <span className="chip">{titleCase(f.category)}</span>
                </div>
                {f.value && <div className="feature-value">{f.value}</div>}
                {f.reason && <p className="feature-reason muted">{f.reason}</p>}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ── Sources ────────────────────────────────────────────── */}
      <section className="report-section" aria-labelledby="sources-h">
        <div className="section-head">
          <h2 id="sources-h">Sources</h2>
          <p className="muted">
            Every page the investigation read, with how much it contributed.
          </p>
        </div>
        <SourceList sources={report.sources_detail} />
      </section>

      {/* ── Analysis detail ────────────────────────────────────── */}
      {report.analysis_context &&
        (report.analysis_context.credibility || report.analysis_context.sentiment) && (
          <section className="report-section" aria-labelledby="analysis-h">
            <div className="section-head">
              <h2 id="analysis-h">Under the assessments</h2>
              <p className="muted">
                How the ML layer scored the quality of this evidence.
              </p>
            </div>
            <div className="analysis-grid">
              <CredibilityPanel credibility={report.analysis_context.credibility} />
              <SentimentBar sentiment={report.analysis_context.sentiment} />
            </div>
          </section>
        )}

      {/* ── Ask ────────────────────────────────────────────────── */}
      <section className="report-section" aria-labelledby="ask-h">
        <AskPanel investigationId={investigationId} demo={demo} />
      </section>
    </article>
  );
}
