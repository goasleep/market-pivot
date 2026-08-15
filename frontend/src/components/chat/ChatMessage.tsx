import {
  A2UIRenderer,
  createMarkdownSurface,
  type A2UIAction,
  type A2UIMessage,
} from "./A2UIRenderer";
import { WidgetRenderer } from "./WidgetRenderer";
import { cn } from "@/lib/utils";
import { Bot, Loader2, Pencil, User } from "lucide-react";

export interface ChatMessagePart {
  type: "text" | "a2ui" | "widget";
  content: string | A2UIMessage | A2UIMessage[];
  widgetType?: string;
}

export interface ChatMessageData {
  id?: string;
  role: "user" | "assistant";
  parts: ChatMessagePart[];
  loading?: boolean;
  status?:
    | "pending"
    | "running"
    | "completed"
    | "cancelled"
    | "failed"
    | "interrupted";
  taskId?: string;
}

interface ChatMessageProps {
  message: ChatMessageData;
  editable?: boolean;
  onEdit?: () => void;
  onAction?: (action: A2UIAction) => void;
}

/**
 * Renders a single chat message with mixed text and widget parts.
 * User messages are right-aligned, assistant messages are left-aligned.
 */
export function ChatMessage({
  message,
  editable = false,
  onEdit,
  onAction,
}: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 px-4 py-3",
        isUser ? "flex-row-reverse" : "flex-row",
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground",
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* Message body */}
      <div
        className={cn(
          "group flex max-w-[85%] flex-col gap-2",
          isUser ? "items-end" : "items-start",
        )}
      >
        <div className="flex items-start gap-2">
          <div className="flex flex-col items-end gap-2">
            {message.parts.map((part, i) => {
              if (part.type === "text") {
                if (!isUser) {
                  return (
                    <div key={i} className="w-full overflow-hidden rounded-lg">
                      <A2UIRenderer
                        messages={createMarkdownSurface(
                          String(part.content),
                          `legacy-markdown-${message.id || "message"}-${i}`,
                        )}
                        onAction={onAction}
                      />
                    </div>
                  );
                }
                return (
                  <div
                    key={i}
                    className={cn(
                      "rounded-lg px-4 py-2 text-sm leading-relaxed",
                      isUser
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-foreground",
                    )}
                  >
                    <MarkdownText text={String(part.content)} />
                  </div>
                );
              }
              if (part.type === "widget") {
                return (
                  <div key={i} className="w-full overflow-hidden rounded-lg">
                    <WidgetRenderer html={String(part.content)} />
                  </div>
                );
              }
              if (part.type === "a2ui") {
                const messages = Array.isArray(part.content)
                  ? part.content
                  : [part.content];
                return (
                  <div key={i} className="w-full overflow-hidden rounded-lg">
                    <A2UIRenderer
                      messages={messages as A2UIMessage[]}
                      onAction={onAction}
                    />
                  </div>
                );
              }
              return null;
            })}

            {message.loading && (
              <div className="flex animate-in items-center gap-2 rounded-lg bg-muted px-4 py-2 fade-in duration-300">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm text-muted-foreground">Agent 正在处理</span>
                <span className="flex gap-0.5" aria-label="处理中">
                  <span className="h-1 w-1 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]" />
                  <span className="h-1 w-1 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
                  <span className="h-1 w-1 animate-bounce rounded-full bg-muted-foreground" />
                </span>
              </div>
            )}
          </div>

          {editable && onEdit && (
            <button
              type="button"
              onClick={onEdit}
              aria-label="编辑最后一条消息"
              title="编辑消息"
              className="mt-1 rounded-md p-1.5 text-muted-foreground opacity-0 transition-opacity hover:bg-accent hover:text-foreground group-hover:opacity-100 focus-visible:opacity-100"
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Simple inline markdown renderer.
 * Handles **bold** and line breaks without external deps.
 */
function MarkdownText({ text }: { text: string }) {
  const lines = text.split("\n");

  return (
    <>
      {lines.map((line, i) => {
        const parts = line.split(/(\*\*[^*]+\*\*)/g);
        return (
          <span key={i}>
            {parts.map((part, j) => {
              if (part.startsWith("**") && part.endsWith("**")) {
                return (
                  <strong key={j} className="font-semibold">
                    {part.slice(2, -2)}
                  </strong>
                );
              }
              return <span key={j}>{part}</span>;
            })}
            {i < lines.length - 1 && <br />}
          </span>
        );
      })}
    </>
  );
}
