"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Upload, ImageIcon, Layers, LinkIcon, CircleAlert, Download, Calculator, FileImage, Info } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

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

  // Metrics
  const [metrics, setMetrics] = useState<Record<string, number | string> | null>(null)
  const [metricsLoading, setMetricsLoading] = useState(false)
  const [metricsError, setMetricsError] = useState<string | null>(null)
  const [customMaskFile, setCustomMaskFile] = useState<File | null>(null)

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
            data[i] = 0 // R
            data[i + 1] = 255 // G
            data[i + 2] = 0 // B
            data[i + 3] = Math.round(alpha * 255) // A
          } else {
            data[i + 3] = 0 // transparent
          }
        }
        octx.putImageData(imgData, 0, 0)
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
    setMetrics(null)
    setMetricsError(null)

    try {
      const form = new FormData()
      form.append("image", imageFile)
      form.append("threshold", String(threshold))
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

  // Metrics: compute from current mask blob
  const computeMetricsFromCurrentMask = async () => {
    if (!maskUrl) return
    setMetricsError(null)
    setMetrics(null)
    setMetricsLoading(true)
    try {
      const blob = await (await fetch(maskUrl)).blob()
      const file = new File([blob], "mask.png", { type: "image/png" })

      const form = new FormData()
      form.append("mask", file)
      form.append("endpoint", endpoint) // use same backend base
      // bin_thresh left default (128) on the server

      const res = await fetch("/api/metrics-from-mask", {
        method: "POST",
        body: form,
      })
      const text = await res.text()
      if (!res.ok) {
        throw new Error(text || "Metrics failed")
      }
      const json = JSON.parse(text)
      setMetrics(json)
    } catch (err: any) {
      setMetricsError(err?.message || "Failed to compute metrics")
    } finally {
      setMetricsLoading(false)
    }
  }

  // Metrics: compute from uploaded mask file
  const computeMetricsFromUploadedMask = async () => {
    if (!customMaskFile) return
    setMetricsError(null)
    setMetrics(null)
    setMetricsLoading(true)
    try {
      const form = new FormData()
      form.append("mask", customMaskFile)
      form.append("endpoint", endpoint)

      const res = await fetch("/api/metrics-from-mask", {
        method: "POST",
        body: form,
      })
      const text = await res.text()
      if (!res.ok) {
        throw new Error(text || "Metrics failed")
      }
      const json = JSON.parse(text)
      setMetrics(json)
    } catch (err: any) {
      setMetricsError(err?.message || "Failed to compute metrics")
    } finally {
      setMetricsLoading(false)
    }
  }

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-8 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Segmentation Mask Inference</h1>
          <p className="text-sm text-muted-foreground">
            Upload an image to generate a binary segmentation mask with your trained model. Then compute metrics from
            the mask.
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-md border px-3 py-1 text-xs text-muted-foreground">
          <Info className="h-4 w-4" />
          <span>Project v1.3.0</span>
        </div>
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
                  setMetrics(null)
                  setError(null)
                  setImageFile(e.target.files?.[0] ?? null)
                }}
              />
              <p className="text-xs text-muted-foreground">Supported: JPG, PNG (preferred in browser).</p>
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
                Default points to the local FastAPI server. You can override it here.
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
                  setMetrics(null)
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

      <section className="mt-8 grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calculator className="h-5 w-5" />
              Metrics (from mask)
            </CardTitle>
            <CardDescription>
              Compute personalized metrics directly from a binary mask (0/255). Use the predicted mask or upload one.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={computeMetricsFromCurrentMask} disabled={!maskUrl || metricsLoading}>
                {metricsLoading ? "Computing…" : "Use current mask"}
              </Button>
              <div className="flex items-center gap-2">
                <Label htmlFor="custom-mask" className="flex items-center gap-2 text-sm">
                  <FileImage className="h-4 w-4" />
                  Upload mask
                </Label>
                <Input
                  id="custom-mask"
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={(e) => {
                    setMetrics(null)
                    setMetricsError(null)
                    setCustomMaskFile(e.target.files?.[0] ?? null)
                  }}
                  className="max-w-xs"
                />
                <Button
                  variant="secondary"
                  onClick={computeMetricsFromUploadedMask}
                  disabled={!customMaskFile || metricsLoading}
                >
                  {metricsLoading ? "Computing…" : "Compute from file"}
                </Button>
              </div>
            </div>

            {metricsError && (
              <Alert variant="destructive">
                <CircleAlert className="h-4 w-4" />
                <AlertTitle>Metrics failed</AlertTitle>
                <AlertDescription>{metricsError}</AlertDescription>
              </Alert>
            )}

            {metrics && (
              <div className="mt-2">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Metric</TableHead>
                      <TableHead className="text-right">Value</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(metrics).map(([k, v]) => (
                      <TableRow key={k}>
                        <TableCell className="whitespace-pre-wrap">{k}</TableCell>
                        <TableCell className="text-right">{typeof v === "number" ? v.toFixed(2) : String(v)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>How to run the Python inference server</CardTitle>
            <CardDescription>FastAPI server that loads your .pth and returns a PNG mask</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <ol className="list-decimal space-y-2 pl-5 text-sm">
              <li>Start your Python env and install dependencies (see scripts/requirements.txt).</li>
              <li>
                Run the server:
                <code className="block rounded bg-muted px-2 py-1 mt-2 text-xs">
                  {
                    "python scripts/inference_server.py --weights C:\\\\path\\\\to\\\\best_model.pth --default-resize 640 --force-imagenet-stats --port 8000"
                  }
                </code>
              </li>
              <li>Point the endpoint field above to http://localhost:8000/predict (the UI will proxy).</li>
            </ol>
            <p className="text-xs text-muted-foreground">
              The metrics are computed from the mask image, matching your metrics code expectations.
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
