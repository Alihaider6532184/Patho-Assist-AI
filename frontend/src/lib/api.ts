/**
 * Patho-Assist AI — Backend API Client
 * 
 * Centralised HTTP layer for all FastAPI communication.
 * Ensures consistent error handling and type safety.
 * 
 * Base URL points to the local FastAPI server.
 * No data ever leaves the host machine.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ============================================================================
// Types
// ============================================================================

export interface HealthResponse {
  status: string;
  version: string;
  ollama_url: string;
  models: {
    vision: string;
    text: string;
  };
  models_available: {
    vision: boolean;
    text: boolean;
  };
}

export interface SessionResponse {
  session_id: string;
}

export interface SessionInfo {
  session_id: string;
  has_image: boolean;
  image_filename: string | null;
  chat_turns: number;
}

export interface IngestResponse {
  session_id: string;
  filename: string;
  num_chunks: number;
  message: string;
}

export interface AnalyzeImageResponse {
  session_id: string;
  filename: string;
  description: string;
  message: string;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  sources: Array<{
    text: string;
    page: number;
    chunk_index: number;
    distance: number;
  }>;
  used_image_context: boolean;
}

// ============================================================================
// API Client
// ============================================================================

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  /**
   * Generic fetcher with error handling.
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(
        errorBody.detail || `API Error: ${response.status} ${response.statusText}`
      );
    }

    return response.json();
  }

  // --- Health Check -------------------------------------------------------
  async checkHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  // --- Session Management -------------------------------------------------
  async createSession(): Promise<SessionResponse> {
    return this.request<SessionResponse>("/session", { method: "POST" });
  }

  async getSession(sessionId: string): Promise<SessionInfo> {
    return this.request<SessionInfo>(`/session/${sessionId}`);
  }

  async deleteSession(sessionId: string): Promise<void> {
    return this.request(`/session/${sessionId}`, { method: "DELETE" });
  }

  // --- PDF Ingestion ------------------------------------------------------
  async ingestPdf(file: File, sessionId: string): Promise<IngestResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sessionId);

    return this.request<IngestResponse>("/ingest-pdf", {
      method: "POST",
      body: formData,
      // Note: Don't set Content-Type — browser sets it with boundary for multipart
    });
  }

  // --- Vision Analysis ----------------------------------------------------
  async analyzeImage(
    file: File,
    sessionId: string
  ): Promise<AnalyzeImageResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sessionId);

    return this.request<AnalyzeImageResponse>("/analyze-image", {
      method: "POST",
      body: formData,
    });
  }

  // --- Cross-Modal Chat ---------------------------------------------------
  async chat(sessionId: string, question: string): Promise<ChatResponse> {
    return this.request<ChatResponse>("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question }),
    });
  }
}

// Singleton instance
export const api = new ApiClient(API_BASE);
