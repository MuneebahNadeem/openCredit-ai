import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import Nav from "../components/Nav";
import ReportView from "../components/ReportView";
import { getInvestigation, getResult, setSaved } from "../lib/api";

/* The live report page. Fetches the completed investigation and renders the
   same ReportView the demo uses — one layout, two data sources. */
export default function ReportPage() {
  const { id } = useParams();
  const [record, setRecord] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [stillRunning, setStillRunning] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let active = true;

    async function load() {
      let rec = null;
      try {
        rec = await getInvestigation(id);
      } catch (err) {
        if (active) setError(err);
        return;
      }
      if (!active) return;
      setRecord(rec);
      if (rec.status === "failed") {
        setError({
          status: 422,
          message:
            rec.error ||
            "This investigation could not be completed. No report was produced.",
        });
        return;
      }
      try {
        const res = await getResult(id);
        if (active) setResult(res);
      } catch (err) {
        if (!active) return;
        if (err.status === 409) setStillRunning(true);
        else setError(err);
      }
    }

    load();
    return () => {
      active = false;
    };
  }, [id]);

  async function toggleSave() {
    if (!record || saving) return;
    const next = !record.saved;
    setSaving(true);
    try {
      const updated = await setSaved(id, next);
      setRecord((r) => ({ ...r, saved: updated.saved }));
    } catch {
      /* keep current state; the toggle is non-critical */
    } finally {
      setSaving(false);
    }
  }

  if (error) {
    return (
      <div className="page-report">
        <Nav dark />
        <main className="container report-container">
          <div className="card card-pad report-state">
            <h1>
              {error.status === 404
                ? "We couldn’t find that investigation."
                : "No report available"}
            </h1>
            <p className="muted">{error.message}</p>
            <div className="room-actions">
              <Link to="/new" className="btn btn-primary">
                Start a new investigation
              </Link>
              <Link to="/" className="btn btn-secondary">
                Back to home
              </Link>
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (stillRunning) {
    return (
      <div className="page-report">
        <Nav dark />
        <main className="container report-container">
          <div className="card card-pad report-state">
            <h1>The report isn’t ready yet.</h1>
            <p className="muted">
              This investigation is still running. We’ll take you back to the
              room where you can follow its real progress.
            </p>
            <div className="room-actions">
              <Link to={`/investigation/${id}`} className="btn btn-primary">
                Back to the investigation room
              </Link>
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="page-report">
        <Nav dark />
        <main className="container report-container">
          <div className="report-loading" role="status" aria-live="polite">
            <span className="loading-ring" aria-hidden="true" />
            <p className="muted">Loading report…</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="page-report">
      <Nav dark />
      <main className="container report-container">
        <div className="report-toolbar">
          <Link to="/" className="report-back">
            ← New investigation
          </Link>
          <div className="report-toolbar-actions">
            <button
              type="button"
              className={`btn btn-secondary report-save ${
                record?.saved ? "is-saved" : ""
              }`}
              onClick={toggleSave}
              disabled={saving || !record}
              aria-pressed={record ? Boolean(record.saved) : undefined}
            >
              {record?.saved ? "★ Saved" : "☆ Save report"}
            </button>
          </div>
        </div>
        <ReportView
          report={result}
          business={record?.business}
          investigationId={id}
        />
      </main>
    </div>
  );
}
