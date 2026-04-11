"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send,
  Bot,
  User,
  Sparkles,
  BookOpen,
  AlertCircle,
  Microscope,
  Brain,
  Database,
  Zap,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ============================================================================
// Types
// ============================================================================
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: Array<{
    text: string;
    page: number;
    chunk_index: number;
    distance: number;
  }>;
  usedImageContext?: boolean;
}

interface ChatPanelProps {
  sessionId: string | null;
  messages: ChatMessage[];
  isLoading: boolean;
  onSendMessage: (message: string) => Promise<void>;
  hasPdf: boolean;
  hasImage: boolean;
}

// ============================================================================
// Loading Stage Messages (UX: Masking Model Swap Latency)
// ============================================================================
const LOADING_STAGES = [
  { message: "Querying patient document vectors…", icon: Database, delay: 0 },
  { message: "Retrieving relevant context…", icon: BookOpen, delay: 1500 },
  { message: "Swapping to reasoning engine…", icon: Brain, delay: 3500 },
  { message: "Loading medical reasoning model…", icon: Zap, delay: 5500 },
  { message: "Analyzing cross-modal context…", icon: Microscope, delay: 8000 },
  { message: "Generating clinical insights…", icon: Sparkles, delay: 11000 },
];

const IMAGE_LOADING_STAGES = [
  { message: "Loading medical vision engine…", icon: Microscope, delay: 0 },
  { message: "Analyzing histopathology slide…", icon: Brain, delay: 2000 },
  { message: "Identifying cellular anomalies…", icon: Zap, delay: 5000 },
  { message: "Generating detailed description…", icon: Sparkles, delay: 8000 },
];

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Multi-stage loading indicator that cycles through professional messages
 * to mask the 2–10s model swap latency.
 */
function LoadingIndicator({ hasImage }: { hasImage: boolean }) {
  const [currentStage, setCurrentStage] = useState(0);
  const stages = LOADING_STAGES;

  useEffect(() => {
    const timers: NodeJS.Timeout[] = [];

    stages.forEach((stage, index) => {
      if (index === 0) return; // First stage shown immediately
      const timer = setTimeout(() => setCurrentStage(index), stage.delay);
      timers.push(timer);
    });

    return () => timers.forEach(clearTimeout);
  }, []);

  const CurrentIcon = stages[currentStage].icon;

  return (
    <div className="flex items-start gap-3 animate-fade-in-up">
      {/* AI Avatar */}
      <div className="w-8 h-8 rounded-lg bg-primary/15 flex items-center justify-center shrink-0 mt-0.5">
        <Bot className="w-4 h-4 text-primary" />
      </div>

      <div className="flex-1 space-y-2">
        {/* Animated stage message */}
        <div className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-muted/50 border border-border/50">
          <CurrentIcon className="w-4 h-4 text-primary animate-pulse" />
          <span className="text-sm text-muted-foreground animate-fade-in-up" key={currentStage}>
            {stages[currentStage].message}
          </span>
        </div>

        {/* Progress dots */}
        <div className="flex items-center gap-1.5 px-1">
          {stages.map((_, i) => (
            <div
              key={i}
              className={cn(
                "w-1.5 h-1.5 rounded-full transition-all duration-500",
                i <= currentStage ? "bg-primary" : "bg-muted-foreground/20",
                i === currentStage && "w-3"
              )}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * Individual chat message bubble.
 */
function MessageBubble({ message }: { message: ChatMessage }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex items-start gap-3 animate-fade-in-up",
        isUser && "flex-row-reverse"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5",
          isUser ? "bg-secondary" : "bg-primary/15"
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-secondary-foreground" />
        ) : (
          <Bot className="w-4 h-4 text-primary" />
        )}
      </div>

      {/* Message content */}
      <div
        className={cn(
          "flex-1 max-w-[85%] space-y-2",
          isUser && "flex flex-col items-end"
        )}
      >
        <div
          className={cn(
            "px-4 py-3 rounded-2xl text-sm leading-relaxed",
            isUser
              ? "bg-primary text-primary-foreground rounded-tr-md"
              : "bg-muted/50 border border-border/50 text-foreground rounded-tl-md"
          )}
        >
          {/* Render message with basic paragraph support */}
          {message.content.split("\n").map((line, i) => (
            <p key={i} className={i > 0 ? "mt-2" : ""}>
              {line}
            </p>
          ))}
        </div>

        {/* Context badges + sources toggle */}
        {!isUser && (
          <div className="flex items-center gap-2 flex-wrap">
            {message.usedImageContext && (
              <Badge
                variant="secondary"
                className="text-[10px] bg-chart-5/10 text-chart-5 border-chart-5/20"
              >
                <Microscope className="w-2.5 h-2.5 mr-1" />
                Image context used
              </Badge>
            )}

            {message.sources && message.sources.length > 0 && (
              <button
                onClick={() => setShowSources(!showSources)}
                className="text-[10px] text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
              >
                <BookOpen className="w-2.5 h-2.5" />
                {showSources ? "Hide" : "Show"} {message.sources.length} sources
              </button>
            )}

            <span className="text-[10px] text-muted-foreground/40">
              {message.timestamp.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </div>
        )}

        {/* Expandable sources */}
        {showSources && message.sources && (
          <div className="space-y-1.5 animate-fade-in-up">
            {message.sources.map((source, i) => (
              <div
                key={i}
                className="px-3 py-2 rounded-lg bg-muted/30 border border-border/30 text-xs"
              >
                <div className="flex items-center gap-1.5 mb-1">
                  <FileText className="w-3 h-3 text-chart-2" />
                  <span className="text-chart-2 font-medium">
                    Page {source.page}
                  </span>
                  <span className="text-muted-foreground/50">•</span>
                  <span className="text-muted-foreground/50">
                    Relevance: {((1 - source.distance) * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-muted-foreground leading-relaxed line-clamp-3">
                  {source.text}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================
export function ChatPanel({
  sessionId,
  messages,
  isLoading,
  onSendMessage,
  hasPdf,
  hasImage,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  const handleSubmit = async () => {
    if (!input.trim() || isLoading || !sessionId) return;
    const msg = input.trim();
    setInput("");
    await onSendMessage(msg);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isDisabled = !sessionId;

  return (
    <div className="flex flex-col h-full">
      {/* ── Header ────────────────────────────── */}
      <div className="flex items-center justify-between px-5 h-14 border-b border-border/50 shrink-0">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          <h2 className="text-sm font-semibold text-foreground">
            Cross-Modal Chat
          </h2>
        </div>
        <div className="flex items-center gap-1.5">
          {hasPdf && (
            <Badge
              variant="secondary"
              className="text-[10px] bg-chart-2/10 text-chart-2 border-chart-2/20"
            >
              PDF Active
            </Badge>
          )}
          {hasImage && (
            <Badge
              variant="secondary"
              className="text-[10px] bg-chart-5/10 text-chart-5 border-chart-5/20"
            >
              Image Active
            </Badge>
          )}
        </div>
      </div>

      {/* ── Messages Area ─────────────────────── */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-4">
        {messages.length === 0 && !isLoading ? (
          // Empty state
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
              <Sparkles className="w-7 h-7 text-primary" />
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-1">
              Ready to Assist
            </h3>
            <p className="text-sm text-muted-foreground max-w-xs leading-relaxed">
              {isDisabled
                ? "Create a session and upload data to begin cross-modal analysis."
                : "Upload a PDF and/or image, then ask questions. I'll correlate findings across all your data."}
            </p>

            {/* Suggested prompts */}
            {!isDisabled && (hasPdf || hasImage) && (
              <div className="mt-6 space-y-2 w-full max-w-sm">
                <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">
                  Try asking
                </p>
                {[
                  "What are the key findings in this case?",
                  "Describe the cellular morphology observed.",
                  "Is there evidence of malignancy?",
                ].map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => setInput(prompt)}
                    className="w-full text-left px-3 py-2 rounded-lg border border-border/50 text-xs text-muted-foreground hover:text-foreground hover:border-primary/30 hover:bg-primary/5 transition-all duration-200"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-5">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {/* Loading indicator with multi-stage messages */}
            {isLoading && <LoadingIndicator hasImage={hasImage} />}
          </div>
        )}
      </div>

      {/* ── Input Area ────────────────────────── */}
      <div className="shrink-0 px-5 pb-4 pt-2">
        <div
          className={cn(
            "flex items-end gap-2 rounded-xl border border-border/50 bg-muted/30 px-3 py-2 transition-all duration-200",
            !isDisabled && "focus-within:border-primary/40 focus-within:bg-muted/50 focus-within:glow-primary"
          )}
        >
          <textarea
            ref={inputRef}
            id="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isDisabled
                ? "Select a session to start chatting…"
                : "Ask about your patient data…"
            }
            disabled={isDisabled || isLoading}
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none disabled:opacity-50 py-1"
          />
          <Button
            id="btn-send-message"
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading || isDisabled}
            size="icon"
            className="h-8 w-8 rounded-lg bg-primary hover:bg-primary/90 shrink-0 transition-all duration-200 disabled:opacity-30"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>

        <p className="text-[10px] text-muted-foreground/40 text-center mt-2">
          All analysis runs locally. No data leaves this machine.
        </p>
      </div>
    </div>
  );
}
