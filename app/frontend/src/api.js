export async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const detail = payload.detail;
    const message = typeof detail === "string" ? detail : "The request failed.";
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return payload;
}
