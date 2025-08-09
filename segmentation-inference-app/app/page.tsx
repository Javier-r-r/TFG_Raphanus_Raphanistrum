"use client"

import { useState, useEffect, useMemo, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Upload, ImageIcon, Layers, LinkIcon, CircleAlert, Download } from "lucide-react"

type MaskView = "overlay" | "mask-only" | "input-only"

export default function Page() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [maskUrl, setMaskUrl] = useState<string | null>(null)
  const [threshold, setThreshold] = useState<number>(0.5)
  const [alpha, setAlpha] = useState<number>(0.5)
  const [endpoint, setEndpoint] = useState<string>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("inference-endpoint") || "http://localhost:8000/predict"
    }
    return "http://localhost:8000/predict"
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [maskView, setMaskView] = useState<MaskView>("overlay")

  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile)
      setImageUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setImageUrl(null)
    }
  }, [imageFile])

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("inference-endpoint", endpoint)
    }
  }, [endpoint])

  // Draw overlay if we have both image and mask
  useEffect(() => {
    if (!imageUrl || !maskUrl || maskView !== "overlay") return
    const canvas = canvasRef.current
    if (!canvas) return

    const inputImg = new window.Image()
    const maskImg = new window.Image()
    inputImg.crossOrigin = "anonymous"
    maskImg.crossOrigin = "anonymous"
    let cancelled = false

    inputImg.onload = () => {
      if (cancelled) return
      canvas.width = inputImg.naturalWidth
      canvas.height = inputImg.naturalHeight
      const ctx = canvas.getContext("2d")
      if (!ctx) return
      // Draw the input image first
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(inputImg, 0, 0, canvas.width, canvas.height)

      // Draw mask tinted in green with alpha
      maskImg.onload = () => {
        if (cancelled) return
        // Create an offscreen canvas to colorize the mask
        const off = document.createElement("canvas")
        off.width = canvas.width
        off.height = canvas.height
        const octx = off.getContext("2d")
        if (!octx) return
        octx.drawImage(maskImg, 0, 0, off.width, off.height)

        // Get mask image data
        const imgData = octx.getImageData(0, 0, off.width, off.height)
        const data = imgData.data
        // The mask is grayscale (0 or 255). We’ll paint green where > 0.
        for (let i = 0; i < data.length; i += 4) {
          const v = data[i] // R channel in grayscale mask
          if (v > 0) {
            // green color with alpha
            data[i] = 0 // R
            data[i + 1] = 255 // G
            data[i + 2] = 0 // B
            data[i + 3] = Math.round(alpha * 255) // A
          } else {
            data[i + 3] = 0 // transparent for background
          }
        }
        octx.putImageData(imgData, 0, 0)

        // Draw colored mask onto main canvas
        ctx.drawImage(off, 0, 0)
      }
      maskImg.src = maskUrl
    }
    inputImg.src = imageUrl

    return () => {
      cancelled = true
    }
  }, [imageUrl, maskUrl, alpha, maskView])

  const canSegment = useMemo(() => !!imageFile && !!endpoint, [imageFile, endpoint])

  const handleSubmit = async () => {
    if (!imageFile) return
    setError(null)
    setLoading(true)
    setMaskUrl(null)

    try {
      const form = new FormData()
      form.append("image", imageFile)
      form.append("threshold", String(threshold))
      // Allow overriding backend endpoint via proxy
      form.append("endpoint", endpoint)

      const res = await fetch("/api/predict", {
        method: "POST",
        body: form,
      })

      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || "Prediction failed")
      }

      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setMaskUrl(url)
      setMaskView("overlay")
    } catch (e: any) {
      setError(e?.message || "Failed to get prediction")
    } finally {
      setLoading(false)
    }
  }

  const downloadBlob = (url: string | null, filename: string) => {
    if (!url) return
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  // Generate an overlay image URL from canvas for download
  const getOverlayDataUrl = (): string | null => {
    if (!canvasRef.current) return null
    return canvasRef.current.toDataURL("image/png")
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-8">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">Segmentation Mask Inference</h1>
        <p className="text-sm text-muted-foreground">
          Upload an image to generate a binary segmentation mask with your trained model.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Input
            </CardTitle>
            <CardDescription>Select an image and configure inference</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="image">Image</Label>
              <Input
                id="image"
                type="file"
                accept="image/*"
                onChange={(e) => {
                  setMaskUrl(null)
                  setImageFile(e.target.files?.[0] ?? null)
                }}
              />
              <p className="text-xs text-muted-foreground">
                Supported: JPG, PNG, TIFF (browsers often convert TIFF to raster; prefer PNG/JPG here).
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="endpoint" className="flex items-center gap-2">
                <LinkIcon className="h-4 w-4" />
                Inference server endpoint
              </Label>
              <Input
                id="endpoint"
                value={endpoint}
                onChange={(e) => setEndpoint(e.target.value)}
                placeholder="http://localhost:8000/predict"
              />
              <p className="text-xs text-muted-foreground">
                Default points to the local FastAPI server you’ll run below.
              </p>
            </div>

            <Separator />

            <div className="space-y-2">
              <Label className="flex items-center justify-between">
                <span>Threshold</span>
                <Badge variant="secondary">{threshold.toFixed(2)}</Badge>
              </Label>
              <Slider value={[threshold]} step={0.01} min={0} max={1} onValueChange={(v) => setThreshold(v[0])} />
              <p className="text-xs text-muted-foreground">
                Pixels with probability ≥ threshold are marked as foreground.
              </p>
            </div>

            <div className="space-y-2">
              <Label className="flex items-center justify-between">
                <span>Overlay alpha</span>
                <Badge variant="secondary">{alpha.toFixed(2)}</Badge>
              </Label>
              <Slider value={[alpha]} step={0.01} min={0} max={1} onValueChange={(v) => setAlpha(v[0])} />
              <p className="text-xs text-muted-foreground">Controls intensity of the green overlay.</p>
            </div>

            <div className="flex gap-3">
              <Button onClick={handleSubmit} disabled={!canSegment || loading}>
                {loading ? "Segmenting…" : "Segment"}
              </Button>
              <Button
                variant="secondary"
                onClick={() => {
                  setMaskUrl(null)
                  setError(null)
                }}
                disabled={loading}
              >
                Reset
              </Button>
            </div>

            <Alert variant="default">
              <CircleAlert className="h-4 w-4" />
              <AlertTitle>Backend required</AlertTitle>
              <AlertDescription>
                Start the Python server (see instructions below). The UI will call: {endpoint}
              </AlertDescription>
            </Alert>

            {error && (
              <Alert variant="destructive">
                <CircleAlert className="h-4 w-4" />
                <AlertTitle>Inference failed</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers className="h-5 w-5" />
              Preview
            </CardTitle>
            <CardDescription>Input, mask, and overlay</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs
              defaultValue="overlay"
              value={maskView}
              onValueChange={(v) => setMaskView(v as MaskView)}
              className="w-full"
            >
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="input-only" className="flex items-center gap-2">
                  <ImageIcon className="h-4 w-4" />
                  Input
                </TabsTrigger>
                <TabsTrigger value="mask-only" className="flex items-center gap-2">
                  <Layers className="h-4 w-4" />
                  Mask
                </TabsTrigger>
                <TabsTrigger value="overlay" className="flex items-center gap-2">
                  <Layers className="h-4 w-4" />
                  Overlay
                </TabsTrigger>
              </TabsList>

              <div className="mt-4">
                {maskView === "input-only" && (
                  <div className="relative aspect-[4/3] w-full overflow-hidden rounded-md border">
                    {imageUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={imageUrl || "/placeholder.svg"} alt="Input" className="h-full w-full object-contain" />
                    ) : (
                      <EmptyPreview />
                    )}
                  </div>
                )}

                {maskView === "mask-only" && (
                  <div className="relative aspect-[4/3] w-full overflow-hidden rounded-md border">
                    {maskUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={maskUrl || "/placeholder.svg"} alt="Mask" className="h-full w-full object-contain" />
                    ) : imageUrl ? (
                      <div className="flex h-full w-full items-center justify-center text-sm text-muted-foreground">
                        Run segmentation to see mask
                      </div>
                    ) : (
                      <EmptyPreview />
                    )}
                  </div>
                )}

                {maskView === "overlay" && (
                  <div className="relative aspect-[4/3] w-full overflow-hidden rounded-md border">
                    {imageUrl ? <canvas ref={canvasRef} className="h-full w-full object-contain" /> : <EmptyPreview />}
                  </div>
                )}
              </div>
            </Tabs>

            <div className="mt-4 flex flex-wrap gap-2">
              <Button variant="outline" size="sm" onClick={() => downloadBlob(maskUrl, "mask.png")} disabled={!maskUrl}>
                <Download className="mr-2 h-4 w-4" />
                Download mask
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const dataUrl = getOverlayDataUrl()
                  if (dataUrl) {
                    const link = document.createElement("a")
                    link.href = dataUrl
                    link.download = "overlay.png"
                    link.click()
                  }
                }}
                disabled={!maskUrl || maskView !== "overlay"}
              >
                <Download className="mr-2 h-4 w-4" />
                Download overlay
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      <section className="mt-10">
        <Card>
          <CardHeader>
            <CardTitle>How to run the Python inference server</CardTitle>
            <CardDescription>FastAPI server that loads your .pth and returns a PNG mask</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <ol className="list-decimal space-y-2 pl-5 text-sm">
              <li>Download this project (top right “Download Code”).</li>
              <li>
                Ensure your trained files are available:
                <ul className="list-disc pl-6">
                  <li>best_model.pth</li>
                  <li>dataset_mean.npy and dataset_std.npy (if you saved them; otherwise defaults to ImageNet)</li>
                  <li>Optionally config.json (saved by your trainer) with arch and encoder</li>
                </ul>
              </li>
              <li>
                In a terminal, create a Python env and install server deps:
                <code className="block rounded bg-muted px-2 py-1 mt-2 text-xs">
                  {
                    "cd scripts\npython -m venv .venv && source .venv/bin/activate  # On Windows: .venv\\\\Scripts\\\\activate\npip install -r requirements.txt"
                  }
                </code>
              </li>
              <li>
                Start the server (adjust paths as needed):
                <code className="block rounded bg-muted px-2 py-1 mt-2 text-xs">
                  {
                    "python inference_server.py --weights /absolute/path/to/output/best_model.pth --device auto --port 8000"
                  }
                </code>
              </li>
              <li>Keep the server running. Return here, upload an image, and click Segment.</li>
            </ol>
            <p className="text-xs text-muted-foreground">
              Tip: The UI proxies to /api/predict which forwards to your FastAPI server (default
              http://localhost:8000/predict). You can change the endpoint above.
            </p>
          </CardContent>
        </Card>
      </section>
    </main>
  )

  function EmptyPreview() {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
        <ImageIcon className="h-8 w-8 opacity-60" />
        <span>Upload an image to preview</span>
      </div>
    )
  }
}
