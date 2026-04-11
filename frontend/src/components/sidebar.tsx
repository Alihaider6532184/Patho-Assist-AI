"use client";

import { useState } from "react";
import {
  Microscope,
  Plus,
  ChevronLeft,
  ChevronRight,
  FileText,
  Image,
  MessageSquare,
  Trash2,
  Shield,
  Wifi,
  WifiOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ============================================================================
// Types
// ============================================================================
export interface Session {
  id: string;
  label: string;
  hasPdf: boolean;
  hasImage: boolean;
  chatTurns: number;
  createdAt: Date;
}

interface SidebarProps {
  sessions: Session[];
  activeSessionId: string | null;
  isConnected: boolean;
  onCreateSession: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
}

// ============================================================================
// Component
// ============================================================================
export function Sidebar({
  sessions,
  activeSessionId,
  isConnected,
  onCreateSession,
  onSelectSession,
  onDeleteSession,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "relative flex flex-col h-full border-r border-border/50 bg-sidebar transition-all duration-300 ease-in-out",
        collapsed ? "w-[68px]" : "w-[280px]"
      )}
    >
      {/* ── Header ────────────────────────────────── */}
      <div className="flex items-center gap-3 px-4 h-16 shrink-0">
        <div className="relative flex items-center justify-center w-9 h-9 rounded-lg bg-primary/15 shrink-0">
          <Microscope className="w-5 h-5 text-primary" />
          <span className="absolute inset-0 rounded-lg border border-primary/20 animate-pulse" />
        </div>
        {!collapsed && (
          <div className="flex flex-col overflow-hidden animate-fade-in-up">
            <span className="text-sm font-semibold text-foreground tracking-tight truncate">
              Patho-Assist
            </span>
            <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest">
              AI Research Lab
            </span>
          </div>
        )}
      </div>

      <Separator className="opacity-50" />

      {/* ── New Session Button ─────────────────────── */}
      <div className="px-3 pt-3 pb-1">
        <Button
          id="btn-new-session"
          onClick={onCreateSession}
          variant="outline"
          className={cn(
            "w-full border-dashed border-primary/30 hover:border-primary/60 hover:bg-primary/5 transition-all duration-200",
            collapsed ? "px-0 justify-center" : "justify-start gap-2"
          )}
          size={collapsed ? "icon" : "default"}
        >
          <Plus className="w-4 h-4 text-primary" />
          {!collapsed && (
            <span className="text-sm text-muted-foreground">
              New Analysis
            </span>
          )}
        </Button>
      </div>

      {/* ── Session List ──────────────────────────── */}
      <ScrollArea className="flex-1 px-3 py-2">
        <div className="space-y-1">
          {sessions.length === 0 && !collapsed && (
            <p className="text-xs text-muted-foreground/60 text-center py-8 px-2">
              No sessions yet.
              <br />
              Start a new analysis above.
            </p>
          )}

          {sessions.map((session) => (
            <div
              key={session.id}
              id={`session-${session.id}`}
              role="button"
              tabIndex={0}
              onClick={() => onSelectSession(session.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectSession(session.id);
                }
              }}
              className={cn(
                "group w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-all duration-200 cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-primary/50",
                activeSessionId === session.id
                  ? "bg-primary/10 border border-primary/20 glow-border"
                  : "hover:bg-accent/50 border border-transparent"
              )}
            >
              {/* Session icon */}
              <div
                className={cn(
                  "flex items-center justify-center w-8 h-8 rounded-md shrink-0 transition-colors",
                  activeSessionId === session.id
                    ? "bg-primary/20 text-primary"
                    : "bg-muted text-muted-foreground group-hover:bg-accent"
                )}
              >
                <FileText className="w-4 h-4" />
              </div>

              {!collapsed && (
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {session.label}
                  </p>
                  {/* Status indicators */}
                  <div className="flex items-center gap-1.5 mt-1">
                    {session.hasPdf && (
                      <Badge
                        variant="secondary"
                        className="h-4 px-1 text-[9px] bg-chart-2/15 text-chart-2 border-0"
                      >
                        <FileText className="w-2.5 h-2.5 mr-0.5" />
                        PDF
                      </Badge>
                    )}
                    {session.hasImage && (
                      <Badge
                        variant="secondary"
                        className="h-4 px-1 text-[9px] bg-chart-5/15 text-chart-5 border-0"
                      >
                        <Image className="w-2.5 h-2.5 mr-0.5" />
                        IMG
                      </Badge>
                    )}
                    {session.chatTurns > 0 && (
                      <Badge
                        variant="secondary"
                        className="h-4 px-1 text-[9px] bg-primary/15 text-primary border-0"
                      >
                        <MessageSquare className="w-2.5 h-2.5 mr-0.5" />
                        {session.chatTurns}
                      </Badge>
                    )}
                  </div>
                </div>
              )}

              {/* Delete — parent is a <div>, so this nested <button> is valid HTML */}
              {!collapsed && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(session.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-destructive/10 hover:text-destructive"
                  aria-label="Delete session"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* ── Footer: Status & Collapse Toggle ───────── */}
      <Separator className="opacity-50" />
      <div className="px-3 py-3 space-y-2 shrink-0">
        {/* Connection status */}
        <div
          className={cn(
            "flex items-center gap-2 rounded-md px-2.5 py-2 text-xs",
            isConnected
              ? "bg-chart-2/8 text-chart-2"
              : "bg-destructive/8 text-destructive"
          )}
        >
          {isConnected ? (
            <Wifi className="w-3.5 h-3.5 shrink-0" />
          ) : (
            <WifiOff className="w-3.5 h-3.5 shrink-0" />
          )}
          {!collapsed && (
            <span className="truncate">
              {isConnected ? "Backend Connected" : "Backend Offline"}
            </span>
          )}
        </div>

        {/* Security badge */}
        {!collapsed && (
          <div className="flex items-center gap-2 px-2.5 py-1.5 text-[10px] text-muted-foreground/60">
            <Shield className="w-3 h-3" />
            <span>100% Local • No Cloud</span>
          </div>
        )}

        {/* Collapse toggle */}
        <Button
          id="btn-toggle-sidebar"
          variant="ghost"
          size="sm"
          onClick={() => setCollapsed(!collapsed)}
          className="w-full justify-center text-muted-foreground hover:text-foreground"
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </Button>
      </div>
    </aside>
  );
}
