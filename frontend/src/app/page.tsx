"use client";

/**
 * =============================================================================
 *  Patho-Assist AI — Main Dashboard Page
 * =============================================================================
 *  Orchestrates the entire UI:
 *    • Sidebar (session management)
 *    • Upload Panel (PDF + image)
 *    • Chat Panel (cross-modal Q&A)
 *
 *  All state lives here and flows down as props.
 *  API calls are made through the centralised api client.
 * =============================================================================
 */

import { useState, useEffect, useCallback } from "react";
import { Sidebar, type Session } from "@/components/sidebar";
import { UploadPanel } from "@/components/upload-panel";
import { ChatPanel, type ChatMessage } from "@/components/chat-panel";
import { api } from "@/lib/api";

// ============================================================================
// Types
// ============================================================================
interface UploadedFile {
  name: string;
  size: number;
  type: "pdf" | "image";
  status: "idle" | "uploading" | "processing" | "success" | "error";
  progress: number;
  message?: string;
  chunks?: number;
  description?: string;
}

interface SessionData {
  messages: ChatMessage[];
  uploadedPdf: UploadedFile | null;
  uploadedImage: UploadedFile | null;
  imagePreviewUrl: string | null;
}

// ============================================================================
// Main Page Component
// ============================================================================
export default function DashboardPage() {
  // -- Global state --
  const [isConnected, setIsConnected] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // -- Per-session data store --
  const [sessionData, setSessionData] = useState<Record<string, SessionData>>(
    {}
  );

  // Derived state for the active session
  const activeData: SessionData = activeSessionId
    ? sessionData[activeSessionId] || {
        messages: [],
        uploadedPdf: null,
        uploadedImage: null,
        imagePreviewUrl: null,
      }
    : { messages: [], uploadedPdf: null, uploadedImage: null, imagePreviewUrl: null };

  // =========================================================================
  // Helpers
  // =========================================================================
  const updateSessionData = useCallback(
    (sessionId: string, updater: (prev: SessionData) => SessionData) => {
      setSessionData((prev) => ({
        ...prev,
        [sessionId]: updater(
          prev[sessionId] || {
            messages: [],
            uploadedPdf: null,
            uploadedImage: null,
            imagePreviewUrl: null,
          }
        ),
      }));
    },
    []
  );

  const updateSessionMeta = useCallback(
    (sessionId: string, updates: Partial<Session>) => {
      setSessions((prev) =>
        prev.map((s) => (s.id === sessionId ? { ...s, ...updates } : s))
      );
    },
    []
  );

  // =========================================================================
  // Health Check (runs on mount)
  // =========================================================================
  useEffect(() => {
    let interval: NodeJS.Timeout;

    const check = async () => {
      try {
        await api.checkHealth();
        setIsConnected(true);
      } catch {
        setIsConnected(false);
      }
    };

    check();
    interval = setInterval(check, 15000); // Re-check every 15s
    return () => clearInterval(interval);
  }, []);

  // =========================================================================
  // Session Management
  // =========================================================================
  const handleCreateSession = useCallback(async () => {
    try {
      const { session_id } = await api.createSession();
      const newSession: Session = {
        id: session_id,
        label: `Analysis ${sessions.length + 1}`,
        hasPdf: false,
        hasImage: false,
        chatTurns: 0,
        createdAt: new Date(),
      };

      setSessions((prev) => [newSession, ...prev]);
      setActiveSessionId(session_id);
      setSessionData((prev) => ({
        ...prev,
        [session_id]: {
          messages: [],
          uploadedPdf: null,
          uploadedImage: null,
          imagePreviewUrl: null,
        },
      }));
    } catch (err) {
      console.error("Failed to create session:", err);
    }
  }, [sessions.length]);

  const handleSelectSession = useCallback((id: string) => {
    setActiveSessionId(id);
  }, []);

  const handleDeleteSession = useCallback(
    async (id: string) => {
      try {
        await api.deleteSession(id);
      } catch {
        // Session may not exist on backend — still remove locally
      }

      setSessions((prev) => prev.filter((s) => s.id !== id));
      setSessionData((prev) => {
        const next = { ...prev };
        // Revoke image preview URL to prevent memory leaks
        if (next[id]?.imagePreviewUrl) {
          URL.revokeObjectURL(next[id].imagePreviewUrl!);
        }
        delete next[id];
        return next;
      });

      if (activeSessionId === id) {
        setActiveSessionId(null);
      }
    },
    [activeSessionId]
  );

  // =========================================================================
  // PDF Upload
  // =========================================================================
  const handlePdfUpload = useCallback(
    async (file: File) => {
      if (!activeSessionId) return;
      const sid = activeSessionId;

      // Set uploading state
      updateSessionData(sid, (prev) => ({
        ...prev,
        uploadedPdf: {
          name: file.name,
          size: file.size,
          type: "pdf",
          status: "uploading",
          progress: 30,
        },
      }));

      try {
        // Progress simulation (actual upload is a single POST)
        updateSessionData(sid, (prev) => ({
          ...prev,
          uploadedPdf: { ...prev.uploadedPdf!, status: "processing", progress: 60 },
        }));

        const result = await api.ingestPdf(file, sid);

        updateSessionData(sid, (prev) => ({
          ...prev,
          uploadedPdf: {
            ...prev.uploadedPdf!,
            status: "success",
            progress: 100,
            chunks: result.num_chunks,
            message: result.message,
          },
        }));

        updateSessionMeta(sid, { hasPdf: true });
      } catch (err: any) {
        updateSessionData(sid, (prev) => ({
          ...prev,
          uploadedPdf: {
            ...prev.uploadedPdf!,
            status: "error",
            progress: 0,
            message: err.message || "Upload failed",
          },
        }));
      }
    },
    [activeSessionId, updateSessionData, updateSessionMeta]
  );

  // =========================================================================
  // Image Upload + Vision Analysis
  // =========================================================================
  const handleImageUpload = useCallback(
    async (file: File) => {
      if (!activeSessionId) return;
      const sid = activeSessionId;

      // Create preview URL
      const previewUrl = URL.createObjectURL(file);

      // Set uploading state
      updateSessionData(sid, (prev) => ({
        ...prev,
        imagePreviewUrl: previewUrl,
        uploadedImage: {
          name: file.name,
          size: file.size,
          type: "image",
          status: "uploading",
          progress: 20,
          message: "Uploading image…",
        },
      }));

      try {
        // Update to processing (vision inference)
        updateSessionData(sid, (prev) => ({
          ...prev,
          uploadedImage: {
            ...prev.uploadedImage!,
            status: "processing",
            progress: 50,
            message: "Running vision AI analysis…",
          },
        }));

        const result = await api.analyzeImage(file, sid);

        updateSessionData(sid, (prev) => ({
          ...prev,
          uploadedImage: {
            ...prev.uploadedImage!,
            status: "success",
            progress: 100,
            description: result.description,
            message: result.message,
          },
        }));

        updateSessionMeta(sid, { hasImage: true });
      } catch (err: any) {
        updateSessionData(sid, (prev) => ({
          ...prev,
          uploadedImage: {
            ...prev.uploadedImage!,
            status: "error",
            progress: 0,
            message: err.message || "Image analysis failed",
          },
        }));
      }
    },
    [activeSessionId, updateSessionData, updateSessionMeta]
  );

  // =========================================================================
  // Cross-Modal Chat
  // =========================================================================
  const handleSendMessage = useCallback(
    async (content: string) => {
      if (!activeSessionId) return;
      const sid = activeSessionId;

      // Add user message immediately
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content,
        timestamp: new Date(),
      };

      updateSessionData(sid, (prev) => ({
        ...prev,
        messages: [...prev.messages, userMsg],
      }));

      setIsLoading(true);

      try {
        const result = await api.chat(sid, content);

        const assistantMsg: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: result.answer,
          timestamp: new Date(),
          sources: result.sources,
          usedImageContext: result.used_image_context,
        };

        updateSessionData(sid, (prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMsg],
        }));

        // Update chat turn count in sidebar
        updateSessionMeta(sid, {
          chatTurns:
            (sessions.find((s) => s.id === sid)?.chatTurns || 0) + 1,
        });
      } catch (err: any) {
        const errorMsg: ChatMessage = {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: `⚠️ Error: ${err.message || "Failed to get response"}. Please ensure Ollama is running with the required models pulled.`,
          timestamp: new Date(),
        };

        updateSessionData(sid, (prev) => ({
          ...prev,
          messages: [...prev.messages, errorMsg],
        }));
      } finally {
        setIsLoading(false);
      }
    },
    [activeSessionId, sessions, updateSessionData, updateSessionMeta]
  );

  // =========================================================================
  // Render
  // =========================================================================
  return (
    <div className="flex h-full bg-background">
      {/* ── Sidebar ────────────────────────────── */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        isConnected={isConnected}
        onCreateSession={handleCreateSession}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
      />

      {/* ── Main Workspace (Split View) ──────── */}
      <main className="flex-1 flex min-w-0">
        {/* Left Panel: Upload Zone */}
        <div className="w-[380px] shrink-0 border-r border-border/50 bg-card/30">
          <UploadPanel
            sessionId={activeSessionId}
            onPdfUpload={handlePdfUpload}
            onImageUpload={handleImageUpload}
            uploadedPdf={activeData.uploadedPdf}
            uploadedImage={activeData.uploadedImage}
            imagePreviewUrl={activeData.imagePreviewUrl}
          />
        </div>

        {/* Right Panel: Cross-Modal Chat */}
        <div className="flex-1 min-w-0 bg-background bg-grid">
          <ChatPanel
            sessionId={activeSessionId}
            messages={activeData.messages}
            isLoading={isLoading}
            onSendMessage={handleSendMessage}
            hasPdf={activeData.uploadedPdf?.status === "success"}
            hasImage={activeData.uploadedImage?.status === "success"}
          />
        </div>
      </main>
    </div>
  );
}
