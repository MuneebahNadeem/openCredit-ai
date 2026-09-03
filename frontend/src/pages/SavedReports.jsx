import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Nav from "../components/Nav";
import { LevelBadge } from "../components/Badges";
import { listInvestigations, setSaved } from "../lib/api";
import { formatDateTime, formatScore } from "../lib/format";

export default function SavedReports() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const all = await listInvestigations();
        if (active) setRecords((all || []).filter((r) => r.saved));
      } catch (err) {
        if (active) setError(err.message || "Could not load saved reports.");
      } finally {
        if (active) setLoading(false);
      }
    }

    load();
    return () => {
      active = false;
    };
  }, []);

  async function unsave(id) {
    try {
      await setSaved(id, false);
      setRecords((rows) => rows.filter((r) => r.id !== id));
    } catch {
      /* non-critical; keep item so user can retry */
    }
  }

  if (loading) {
    return (
      <div className="page-saved">
        <Nav dark />
        <main className="container report-container">
          <div className="report-loading" role="status" aria-live="polite">
            <span className="loading-ring" aria-hidden="true" />
            <p className="muted">Loading saved reports…</p>
          </div>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-saved">
        <Nav dark />
        <main className="container report-container">
          <div className="card card-pad report-state">
            <h1>Could not load saved reports</h1>
            <p className="muted">{error}</p>
            <div className="room-actions">
              <Link to="/" className="btn btn-secondary">
                Back to home
              </Link>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="page-saved">
      <Nav dark />
      <main className="container report-container">
        <div className="report-toolbar">
          <h1>Saved reports</h1>
          <Link to="/" className="report-back">
            ← Back to home
          </Link>
        </div>

        {records.length === 0 ? (
          <div className="card card-pad report-state">
            <h2>No saved reports yet</h2>
            <p className="muted">
              When you save a completed report, it will appear here.
            </p>
            <div className="room-actions">
              <Link to="/new" className="btn btn-primary">
                Start an investigation
              </Link>
            </div>
          </div>
        ) : (
          <ul className="saved-grid">
            {records.map((record) => (
              <SavedCard key={record.id} record={record} onUnsave={unsave} />
            ))}
          </ul>
        )}
      </main>
    </div>
  );
}

function SavedCard({ record, onUnsave }) {
  const business = record.business || {};
  const trust = record.trustworthiness;
  const potential = record.business_potential;

  return (
    <li className="saved-card">
      <div className="saved-card-main">
        <div className="saved-card-head">
          <Link to={`/report/${record.id}`} className="saved-card-title">
            {business.name || "Untitled investigation"}
          </Link>
          <LevelBadge level={trust?.level} />
        </div>

        <div className="saved-card-meta">
          <span>{formatDateTime(record.completed_at || record.created_at)}</span>
          <span className="saved-card-sep">·</span>
          <span>{record.phase_label || statusLabel(record.status)}</span>
          <span className="saved-card-sep">·</span>
          <span className="mono">{record.id}</span>
        </div>

        {(trust || potential) && (
          <div className="saved-card-scores">
            {trust && (
              <MiniBar
                label="Trustworthiness"
                score={trust.score}
                level={trust.level}
              />
            )}
            {potential && (
              <MiniBar
                label="Business potential"
                score={potential.score}
                level={potential.level}
              />
            )}
          </div>
        )}

        {record.justification && (
          <p className="saved-card-note">{record.justification}</p>
        )}
      </div>

      <div className="saved-card-actions">
        <Link to={`/report/${record.id}`} className="btn btn-secondary">
          View report
        </Link>
        <button
          type="button"
          className="btn btn-secondary report-save is-saved"
          onClick={() => onUnsave(record.id)}
        >
          ★ Saved
        </button>
      </div>
    </li>
  );
}

function MiniBar({ label, score, level }) {
  const pct = Math.max(0, Math.min(100, Math.round((score || 0) * 100)));
  return (
    <div className="hc-row saved-card-score">
      <span className="hc-label">{label}</span>
      <div className="hc-bar">
        <span
          className={`hc-fill ${fillClass(level)}`}
          style={{ width: `${pct}%` }}
          aria-hidden="true"
        />
      </div>
      <span className="hc-val mono">{formatScore(score)}</span>
    </div>
  );
}

function fillClass(level) {
  if (level === "high") return "hc-high";
  if (level === "moderate") return "hc-mod";
  if (level === "low") return "hc-low";
  return "hc-insufficient";
}

function statusLabel(status) {
  if (!status) return "Unknown";
  return status.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
