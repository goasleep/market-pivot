import { WidgetRenderer } from "./WidgetRenderer";
import { cn } from "@/lib/utils";
import { Loader2, User, Bot } from "lucide-react";

export interface ChatMessagePart {
  type: "text" | "widget";
  content: string;
  widgetType?: string;
}

export interface ChatMessageData {
  role: "user" | "assistant";
  parts: ChatMessagePart[];
  loading?: boolean;
}

interface ChatMessageProps {
  message: ChatMessageData;
}

/**
 * Renders a single chat message with mixed text and widget parts.
 * User messages are right-aligned, assistant messages are left-aligned.
 */
export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 px-4 py-3",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* Message body */}
      <div
        className={cn(
          "flex max-w-[85%] flex-col gap-2",
          isUser ? "items-end" : "items-start"
        )}
      >
        {message.parts.map((part, i) => {
          if (part.type === "text") {
            return (
              <div
                key={i}
                className={cn(
                  "rounded-lg px-4 py-2 text-sm leading-relaxed",
                  isUser
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-foreground"
                )}
              >
                <MarkdownText text={part.content} />
              </div>
            );
          }
          if (part.type === "widget") {
            return (
              <div key={i} className="w-full overflow-hidden rounded-lg">
                <WidgetRenderer html={part.content} />
              </div>
            );
          }
          return null;
        })}

        {message.loading && (
          <div className="flex items-center gap-2 rounded-lg bg-muted px-4 py-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm text-muted-foreground">Analyzing...</span>
          </div>
        )}
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
