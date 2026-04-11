"use client";

import { useCallback, useState, useRef } from "react";
import {
  Upload,
  FileText,
  ImageIcon,
  CheckCircle2,
  AlertCircle,
  Loader2,
  X,
  Eye,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

// ============================================================================
// Types
// ============================================================================
type UploadStatus = "idle" | "uploading" | "processing" | "success" | "error";

interface UploadedFile {
  name: string;
  size: number;
  type: "pdf" | "image";
  status: UploadStatus;
  progress: number;
  message?: string;
  chunks?: number; // For PDFs — number of chunks ingested
  description?: string; // For images — vision description
}

interface UploadPanelProps {
  sessionId: string | null;
  onPdfUpload: (file: File) => Promise<void>;
  onImageUpload: (file: File) => Promise<void>;
  uploadedPdf: UploadedFile | null;
  uploadedImage: UploadedFile | null;
  imagePreviewUrl: string | null;
}

// ============================================================================
// Component
// ============================================================================
export function UploadPanel({
  sessionId,
  onPdfUpload,
  onImageUpload,
  uploadedPdf,
  uploadedImage,
  imagePreviewUrl,
}: UploadPanelProps) {
  const [pdfDragActive, setPdfDragActive] = useState(false);
  const [imgDragActive, setImgDragActive] = useState(false);
  const [showDescription, setShowDescription] = useState(false);
  const pdfInputRef = useRef<HTMLInputElement>(null);
  const imgInputRef = useRef<HTMLInputElement>(null);

  // --- Drag handlers (shared factory) ---
  const createDragHandlers = (
    setActive: (v: boolean) => void,
    onDrop: (file: File) => Promise<void>,
    accept: string[]
  ) => ({
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setActive(true);
    },
    onDragLeave: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setActive(false);
    },
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setActive(false);
      const file = e.dataTransfer.files?.[0];
      if (file && accept.some((ext) => file.name.toLowerCase().endsWith(ext))) {
        onDrop(file);
      }
    },
  });

  const pdfDragHandlers = createDragHandlers(setPdfDragActive, onPdfUpload, [
    ".pdf",
  ]);
  const imgDragHandlers = createDragHandlers(setImgDragActive, onImageUpload, [
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
    ".webp",
  ]);

  const isDisabled = !sessionId;

  return (
    <div className="flex flex-col h-full gap-4 p-4 overflow-y-auto">
      {/* ── Header ──────────────────────────────── */}
      <div>
        <h2 className="text-lg font-semibold text-foreground tracking-tight">
          Data Context
        </h2>
        <p className="text-xs text-muted-foreground mt-0.5">
          Upload patient documents and histopathology images
        </p>
      </div>

      {/* ── No Session Warning ───────────────────── */}
      {isDisabled && (
        <div className="flex items-center gap-2 rounded-lg border border-dashed border-muted-foreground/30 bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
          <AlertCircle className="w-4 h-4 shrink-0" />
          Create or select a session to upload files.
        </div>
      )}

      {/* ── PDF Upload Zone ──────────────────────── */}
      <Card
        className={cn(
          "border-dashed transition-all duration-300 overflow-hidden",
          isDisabled && "opacity-40 pointer-events-none",
          pdfDragActive && "border-primary/60 bg-primary/5 glow-primary",
          uploadedPdf?.status === "success" && "border-chart-2/30",
          !pdfDragActive &&
            !uploadedPdf &&
            "upload-zone-idle hover:border-primary/40 hover:bg-primary/[0.02]"
        )}
      >
        <CardHeader className="pb-2 pt-4 px-4">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <FileText className="w-4 h-4 text-chart-2" />
            Patient History (PDF)
          </CardTitle>
        </CardHeader>

        <CardContent className="px-4 pb-4">
          {!uploadedPdf || uploadedPdf.status === "error" ? (
            <div
              {...pdfDragHandlers}
              onClick={() => pdfInputRef.current?.click()}
              className="flex flex-col items-center justify-center py-8 cursor-pointer rounded-lg"
            >
              <div className="w-12 h-12 rounded-full bg-chart-2/10 flex items-center justify-center mb-3">
                <Upload className="w-5 h-5 text-chart-2" />
              </div>
              <p className="text-sm text-muted-foreground">
                Drag & drop or{" "}
                <span className="text-chart-2 font-medium">browse</span>
              </p>
              <p className="text-[10px] text-muted-foreground/60 mt-1">
                PDF up to 50 MB
              </p>
              {uploadedPdf?.status === "error" && (
                <p className="text-xs text-destructive mt-2">
                  {uploadedPdf.message}
                </p>
              )}
              <input
                ref={pdfInputRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) onPdfUpload(file);
                }}
              />
            </div>
          ) : (
            <div className="space-y-3">
              {/* Uploading / Processing state */}
              {(uploadedPdf.status === "uploading" ||
                uploadedPdf.status === "processing") && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-primary" />
                    <span className="text-sm text-muted-foreground">
                      {uploadedPdf.status === "uploading"
                        ? "Uploading…"
                        : "Extracting & embedding chunks…"}
                    </span>
                  </div>
                  <Progress value={uploadedPdf.progress} className="h-1.5" />
                </div>
              )}

              {/* Success state */}
              {uploadedPdf.status === "success" && (
                <div className="flex items-start gap-3 animate-fade-in-up">
                  <div className="w-10 h-10 rounded-lg bg-chart-2/10 flex items-center justify-center shrink-0">
                    <CheckCircle2 className="w-5 h-5 text-chart-2" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">
                      {uploadedPdf.name}
                    </p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {(uploadedPdf.size / 1024 / 1024).toFixed(1)} MB •{" "}
                      {uploadedPdf.chunks} chunks indexed
                    </p>
                  </div>
                  <Badge
                    variant="secondary"
                    className="bg-chart-2/10 text-chart-2 border-chart-2/20 text-[10px] shrink-0"
                  >
                    Indexed ✓
                  </Badge>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── Image Upload Zone ────────────────────── */}
      <Card
        className={cn(
          "border-dashed transition-all duration-300 overflow-hidden flex-1",
          isDisabled && "opacity-40 pointer-events-none",
          imgDragActive && "border-primary/60 bg-primary/5 glow-primary",
          uploadedImage?.status === "success" && "border-chart-5/30",
          !imgDragActive &&
            !uploadedImage &&
            "upload-zone-idle hover:border-primary/40 hover:bg-primary/[0.02]"
        )}
      >
        <CardHeader className="pb-2 pt-4 px-4">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-chart-5" />
            Histopathology Slide
          </CardTitle>
        </CardHeader>

        <CardContent className="px-4 pb-4">
          {!uploadedImage || uploadedImage.status === "error" ? (
            <div
              {...imgDragHandlers}
              onClick={() => imgInputRef.current?.click()}
              className="flex flex-col items-center justify-center py-8 cursor-pointer rounded-lg"
            >
              <div className="w-12 h-12 rounded-full bg-chart-5/10 flex items-center justify-center mb-3">
                <Upload className="w-5 h-5 text-chart-5" />
              </div>
              <p className="text-sm text-muted-foreground">
                Drag & drop or{" "}
                <span className="text-chart-5 font-medium">browse</span>
              </p>
              <p className="text-[10px] text-muted-foreground/60 mt-1">
                PNG, JPG, TIFF, BMP, WebP up to 20 MB
              </p>
              {uploadedImage?.status === "error" && (
                <p className="text-xs text-destructive mt-2">
                  {uploadedImage.message}
                </p>
              )}
              <input
                ref={imgInputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.tiff,.bmp,.webp"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) onImageUpload(file);
                }}
              />
            </div>
          ) : (
            <div className="space-y-3">
              {/* Uploading / Processing */}
              {(uploadedImage.status === "uploading" ||
                uploadedImage.status === "processing") && (
                <div className="space-y-3">
                  {/* Show preview even while processing */}
                  {imagePreviewUrl && (
                    <div className="relative rounded-lg overflow-hidden border border-border/50">
                      <img
                        src={imagePreviewUrl}
                        alt="Histopathology slide"
                        className="w-full h-32 object-cover opacity-60"
                      />
                      <div className="absolute inset-0 flex items-center justify-center bg-background/60 backdrop-blur-sm">
                        <div className="flex flex-col items-center gap-2">
                          <Loader2 className="w-6 h-6 animate-spin text-chart-5" />
                          <span className="text-xs text-chart-5 font-medium">
                            {uploadedImage.message || "Analyzing with AI…"}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  <Progress value={uploadedImage.progress} className="h-1.5" />
                </div>
              )}

              {/* Success state with preview */}
              {uploadedImage.status === "success" && (
                <div className="space-y-3 animate-fade-in-up">
                  {imagePreviewUrl && (
                    <div className="relative rounded-lg overflow-hidden border border-chart-5/20 group">
                      <img
                        src={imagePreviewUrl}
                        alt="Histopathology slide"
                        className="w-full h-36 object-cover"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent" />
                      <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
                        <span className="text-[10px] text-foreground/80 font-medium">
                          {uploadedImage.name}
                        </span>
                        <Badge className="bg-chart-5/20 text-chart-5 border-chart-5/30 text-[10px]">
                          Analyzed ✓
                        </Badge>
                      </div>
                    </div>
                  )}

                  {/* Toggle description */}
                  {uploadedImage.description && (
                    <div>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="w-full text-xs text-muted-foreground hover:text-foreground gap-1.5"
                        onClick={() => setShowDescription(!showDescription)}
                      >
                        <Eye className="w-3.5 h-3.5" />
                        {showDescription
                          ? "Hide AI Description"
                          : "View AI Description"}
                      </Button>
                      {showDescription && (
                        <div className="mt-2 p-3 rounded-lg bg-muted/50 border border-border/50 text-xs text-muted-foreground leading-relaxed max-h-40 overflow-y-auto animate-fade-in-up">
                          {uploadedImage.description}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
