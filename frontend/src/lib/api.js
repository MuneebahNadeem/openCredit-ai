const JSON_HEADERS = { "Content-Type": "application/json" };

async function parseError(response) {
  let detail = `Request failed (${response.status}).`;
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") detail = body.detail;
  } catch {
    /* non-JSON error body */
  }
  const error = new Error(detail);
  error.status = response.status;
  return error;
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export function createInvestigation(payload, documents = []) {
  const formData = new FormData();
  formData.append("payload", JSON.stringify(payload));
  documents.forEach((file) => formData.append("documents", file));

  return request("/api/investigations", {
    method: "POST",
    body: formData,
  });
}

export function getStatus(id) {
  return request(`/api/investigations/${id}/status`);
}

export function getResult(id) {
  return request(`/api/investigations/${id}/result`);
}

export function getInvestigation(id) {
  return request(`/api/investigations/${id}`);
}

export function listInvestigations() {
  return request("/api/investigations");
}

export function setSaved(id, saved) {
  return request(`/api/investigations/${id}/save`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ saved }),
  });
}

export function setTrustworthinessOverride(id, enabled) {
  return request(`/api/investigations/${id}/trustworthiness-override`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ enabled }),
  });
}

export function askQuestion(id, question) {
  return request(`/api/investigations/${id}/ask`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ question }),
  });
}

export function getHealth() {
  return request("/api/health");
}
