import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Nav from "../components/Nav";
import { createInvestigation } from "../lib/api";

const CATEGORY_OPTIONS = [
  "",
  "Clothing & fashion",
  "Food & home kitchen",
  "Beauty & cosmetics",
  "Jewellery & accessories",
  "Home goods & furniture",
  "Electronics & repair",
  "Services",
  "Other",
];

const EXAMPLE_HINTS = [
  "Monthly revenue Rs 250k",
  "10 years in business",
  "4 tailors working",
  "Takes orders via WhatsApp and Instagram",
];

const ALLOWED_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "text/plain",
  "text/csv",
  "image/jpeg",
  "image/png",
];

const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".txt", ".csv", ".jpg", ".jpeg", ".png"];
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

export default function NewInvestigation() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: "",
    location: "",
    category: "",
    website: "",
    socialLinks: [""],
    marketplaceLinks: [""],
    description: "",
    additionalInfo: "",
  });
  const [documents, setDocuments] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  function set(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
    if (errors[field]) setErrors((e) => ({ ...e, [field]: null }));
  }

  function setLink(field, index, value) {
    setForm((f) => ({
      ...f,
      [field]: f[field].map((l, i) => (i === index ? value : l)),
    }));
  }

  function addLink(field) {
    setForm((f) => ({ ...f, [field]: [...f[field], ""] }));
  }

  function removeLink(field, index) {
    setForm((f) => ({
      ...f,
      [field]: f[field].filter((_, i) => i !== index),
    }));
  }

  function cleanUrl(value, label) {
    const v = (value || "").trim();
    if (!v) return null;
    const withScheme = /^https?:\/\//i.test(v) ? v : `https://${v}`;
    if (/\s/.test(withScheme) || !/\./.test(hostOf(withScheme))) {
      throw new Error(`Please enter a valid ${label} URL.`);
    }
    return withScheme;
  }

  function fileIsAllowed(file) {
    if (ALLOWED_TYPES.includes(file.type)) return true;
    const name = file.name.toLowerCase();
    return ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext));
  }

  function addFiles(fileList) {
    const incoming = Array.from(fileList || []);
    const next = [...documents];
    const fileErrors = [];
    incoming.forEach((file) => {
      if (!fileIsAllowed(file)) {
        fileErrors.push(`${file.name}: file type not allowed.`);
        return;
      }
      if (file.size > MAX_FILE_SIZE_BYTES) {
        fileErrors.push(`${file.name}: exceeds ${MAX_FILE_SIZE_MB} MB limit.`);
        return;
      }
      if (!next.some((d) => d.name === file.name && d.size === file.size)) {
        next.push(file);
      }
    });
    setDocuments(next);
    if (fileErrors.length) {
      setSubmitError(fileErrors.join(" "));
    } else {
      setSubmitError(null);
    }
  }

  function removeFile(index) {
    setDocuments((d) => d.filter((_, i) => i !== index));
  }

  function validate() {
    const next = {};
    if (!form.name.trim()) {
      next.name = "Business name is required to begin an investigation.";
    }
    try {
      cleanUrl(form.website, "website");
    } catch (err) {
      next.website = err.message;
    }
    ["socialLinks", "marketplaceLinks"].forEach((field) => {
      form[field].forEach((link) => {
        if (!link.trim()) return;
        try {
          cleanUrl(link, field === "socialLinks" ? "social media" : "marketplace");
        } catch (err) {
          next[field] = err.message;
        }
      });
    });
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  async function submit(event) {
    event.preventDefault();
    setSubmitError(null);
    if (!validate()) return;
    setSubmitting(true);
    try {
      const payload = {
        name: form.name.trim(),
        location: form.location.trim() || null,
        category: form.category || null,
        website: cleanUrl(form.website) || null,
        social_links: form.socialLinks.map((l) => l.trim()).filter(Boolean),
        marketplace_links: form.marketplaceLinks
          .map((l) => l.trim())
          .filter(Boolean),
        description: form.description.trim() || null,
        additional_info: form.additionalInfo.trim() || null,
      };
      const record = await createInvestigation(payload, documents);
      navigate(`/investigation/${record.id}`);
    } catch (err) {
      setSubmitError(err.message || "Could not start the investigation.");
      setSubmitting(false);
    }
  }

  return (
    <div className="page-new">
      <Nav dark />
      <main className="container form-page">
        <div className="form-intro">
          <h1>Start an investigation</h1>
          <p className="muted">
            A business name is all we require. Anything else you add — links,
            description, documents, revenue notes — gives the agent more to work
            with.
          </p>
        </div>

        <form className="card card-pad form-card" onSubmit={submit} noValidate>
          <div className="field-row">
            <div className={`field ${errors.name ? "field-error" : ""}`}>
              <label htmlFor="f-name">
                Business name <span className="req">*</span>
              </label>
              <input
                id="f-name"
                type="text"
                value={form.name}
                onChange={(e) => set("name", e.target.value)}
                placeholder="e.g. Karachi Threads"
                aria-required="true"
                aria-invalid={Boolean(errors.name)}
              />
              {errors.name && (
                <p className="field-msg" role="alert">
                  {errors.name}
                </p>
              )}
            </div>
            <div className="field">
              <label htmlFor="f-location">Location</label>
              <input
                id="f-location"
                type="text"
                value={form.location}
                onChange={(e) => set("location", e.target.value)}
                placeholder="e.g. Karachi, Pakistan"
              />
            </div>
          </div>

          <div className="field-row">
            <div className="field">
              <label htmlFor="f-category">Category</label>
              <select
                id="f-category"
                value={form.category}
                onChange={(e) => set("category", e.target.value)}
              >
                {CATEGORY_OPTIONS.map((c) => (
                  <option key={c} value={c}>
                    {c || "Select a category (optional)"}
                  </option>
                ))}
              </select>
            </div>
            <div className={`field ${errors.website ? "field-error" : ""}`}>
              <label htmlFor="f-website">Website</label>
              <input
                id="f-website"
                type="text"
                value={form.website}
                onChange={(e) => set("website", e.target.value)}
                placeholder="karachithreads.com"
                aria-invalid={Boolean(errors.website)}
              />
              {errors.website && (
                <p className="field-msg" role="alert">
                  {errors.website}
                </p>
              )}
            </div>
          </div>

          <LinkField
            id="f-social"
            label="Social media links"
            hint="Instagram, Facebook, TikTok — one per field."
            values={form.socialLinks}
            error={errors.socialLinks}
            onChange={(i, v) => setLink("socialLinks", i, v)}
            onAdd={() => addLink("socialLinks")}
            onRemove={(i) => removeLink("socialLinks", i)}
            placeholder="instagram.com/…"
          />
          <LinkField
            id="f-market"
            label="Marketplace links"
            hint="Daraz shop, OLX listings, Amazon store."
            values={form.marketplaceLinks}
            error={errors.marketplaceLinks}
            onChange={(i, v) => setLink("marketplaceLinks", i, v)}
            onAdd={() => addLink("marketplaceLinks")}
            onRemove={(i) => removeLink("marketplaceLinks", i)}
            placeholder="daraz.pk/shop/…"
          />

          <div className="field">
            <label htmlFor="f-desc">Short description</label>
            <textarea
              id="f-desc"
              rows={2}
              value={form.description}
              onChange={(e) => set("description", e.target.value)}
              placeholder="Home-based boutique specialising in embroidered lawn suits…"
            />
          </div>

          <div className="field">
            <label htmlFor="f-info">Anything else you know</label>
            <textarea
              id="f-info"
              rows={4}
              value={form.additionalInfo}
              onChange={(e) => set("additionalInfo", e.target.value)}
              placeholder={`e.g. ${EXAMPLE_HINTS.join(" · ")}`}
            />
            <p className="field-hint">
              Self-reported figures are used, but always labeled as inference —
              never as verified facts.
            </p>
          </div>

          <div className="field">
            <label>Attach documents (optional)</label>
            <div
              className={`dropzone ${dragActive ? "dropzone-active" : ""}`}
              onDragEnter={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDragActive(true);
              }}
              onDragOver={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDragActive(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDragActive(false);
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDragActive(false);
                addFiles(e.dataTransfer.files);
              }}
            >
              <input
                type="file"
                multiple
                accept={ALLOWED_EXTENSIONS.join(",")}
                onChange={(e) => addFiles(e.target.files)}
                aria-label="Upload documents"
              />
              <div className="dropzone-icon" aria-hidden="true">
                📎
              </div>
              <p className="dropzone-title">Drag files here or click to upload</p>
              <p className="dropzone-hint">
                PDF, DOCX, TXT, JPG, PNG, CSV, XLSX · up to {MAX_FILE_SIZE_MB} MB
                each
              </p>
            </div>
            {documents.length > 0 && (
              <ul className="file-list" aria-label="Selected files">
                {documents.map((file, i) => (
                  <li className="file-item" key={`${file.name}-${i}`}>
                    <span className="file-icon" aria-hidden="true">
                      {fileIcon(file.name)}
                    </span>
                    <span className="file-info">
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">{formatBytes(file.size)}</span>
                    </span>
                    <button
                      type="button"
                      className="file-remove"
                      onClick={() => removeFile(i)}
                      aria-label={`Remove ${file.name}`}
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {submitError && (
            <p className="submit-error" role="alert">
              {submitError}
            </p>
          )}

          <div className="form-actions">
            <Link to="/" className="btn btn-secondary">
              Cancel
            </Link>
            <button
              type="submit"
              className="btn btn-primary btn-lg"
              disabled={submitting}
            >
              {submitting ? "Starting…" : "Investigate business"}
            </button>
          </div>
          <p className="form-note muted">
            The agent searches public sources — no accounts are accessed, and
            nothing is posted anywhere.
          </p>
        </form>
      </main>
    </div>
  );
}

function LinkField({
  id,
  label,
  hint,
  values,
  error,
  onChange,
  onAdd,
  onRemove,
  placeholder,
}) {
  return (
    <div className={`field ${error ? "field-error" : ""}`}>
      <div className="field-label-row">
        <label htmlFor={id}>{label}</label>
        <button
          type="button"
          className="link-add"
          onClick={onAdd}
          aria-label={`Add another ${label.toLowerCase()} field`}
        >
          + Add
        </button>
      </div>
      {hint && <p className="field-hint">{hint}</p>}
      {values.map((value, i) => (
        <div className="link-row" key={i}>
          <input
            id={i === 0 ? id : `${id}-${i}`}
            type="text"
            value={value}
            onChange={(e) => onChange(i, e.target.value)}
            placeholder={placeholder}
          />
          {values.length > 1 && (
            <button
              type="button"
              className="link-remove"
              onClick={() => onRemove(i)}
              aria-label={`Remove ${label.toLowerCase()} field ${i + 1}`}
            >
              ✕
            </button>
          )}
        </div>
      ))}
      {error && (
        <p className="field-msg" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

function hostOf(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return url.replace(/^https?:\/\//i, "").split("/")[0];
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileIcon(name) {
  const lower = name.toLowerCase();
  if (lower.endsWith(".pdf")) return "📄";
  if (lower.endsWith(".docx") || lower.endsWith(".doc")) return "📝";
  if (lower.endsWith(".xlsx") || lower.endsWith(".xls") || lower.endsWith(".csv"))
    return "📊";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png"))
    return "🖼";
  return "📁";
}
