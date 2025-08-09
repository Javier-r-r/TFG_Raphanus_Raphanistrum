export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const file = formData.get("image") as File | null
    const threshold = (formData.get("threshold") as string) ?? "0.5"
    const endpointOverride = (formData.get("endpoint") as string) || ""
    const endpoint = endpointOverride.trim() || (process.env.INFERENCE_URL as string) || "http://localhost:8000/predict"

    if (!file) {
      return new Response(JSON.stringify({ error: "No image provided" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      })
    }

    // Forward the request to the Python server
    const upstream = new FormData()
    upstream.append("image", file, file.name)
    upstream.append("threshold", threshold)

    const res = await fetch(endpoint, {
      method: "POST",
      body: upstream,
    })

    if (!res.ok) {
      const text = await res.text().catch(() => "")
      return new Response(JSON.stringify({ error: `Upstream error: ${text || res.statusText}` }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      })
    }

    // Pass-through the PNG mask
    const buf = await res.arrayBuffer()
    return new Response(buf, {
      status: 200,
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "no-store",
      },
    })
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err?.message || "Proxy failed" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    })
  }
}
