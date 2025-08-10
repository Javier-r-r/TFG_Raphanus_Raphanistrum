export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const file = formData.get("mask") as File | null
    const endpointOverride = (formData.get("endpoint") as string) || ""
    // Build base: if INFERENCE_URL is .../predict, strip trailing /predict for base.
    const configured = (process.env.INFERENCE_URL as string) || "http://localhost:8000/predict"
    const base = (endpointOverride || configured).replace(/\/+$/, "")
    const serverBase = base.endsWith("/predict") ? base.slice(0, -"/predict".length) : base
    const endpoint = `${serverBase}/metrics/from-mask`

    if (!file) {
      return new Response(JSON.stringify({ error: "No mask provided" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      })
    }

    const upstream = new FormData()
    upstream.append("mask", file, file.name)

    const res = await fetch(endpoint, { method: "POST", body: upstream })
    const text = await res.text()

    return new Response(text, {
      status: res.status,
      headers: { "Content-Type": res.headers.get("Content-Type") || "application/json" },
    })
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || "Proxy failed" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    })
  }
}
