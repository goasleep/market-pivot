import { useState } from "react";
import {
  A2UIRenderer,
  createMarkdownSurface,
  MarkdownInline,
  type A2UIAction,
  type A2UIMessage,
} from "./A2UIRenderer";
import { WidgetRenderer } from "./WidgetRenderer";
import { LazyArtifactCard } from "./LazyArtifactCard";
import type { Artifact } from "@/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Check,
  Copy,
  ExternalLink,
  GitFork,
  Loader2,
  Pencil,
  RefreshCw,
  User,
  Bot,
} from "lucide-react";

export interface ChatMessagePart {
  type: "text" | "a2ui" | "widget" | "artifact" | "interaction";
  content: string | A2UIMessage | A2UIMessage[] | Artifact | ChatInteraction;
  widgetType?: string;
}

export interface ChatInteractionOption {
  id: string;
  label: string;
}

export interface ChatInteraction {
  interaction_id: string;
  task_id: string;
  kind: "intent_clarification" | "tool_confirmation" | string;
  question: string;
  options: ChatInteractionOption[];
  status: "pending" | "answered" | "cancelled";
  selected_option?: string | null;
  tool?: { tool_name?: string; args?: Record<string, unknown> };
}

export interface ChatReference {
  title: string;
  url?: string;
  source?: string;
  snippet?: string;
  date?: string;
}

export interface TaskOutcomeData {
  outcome: "satisfied" | "partial" | "needs_input" | "data_unavailable" | "failed";
  satisfied: boolean;
  terminal: boolean;
  missing?: string[];
  reason?: string;
}

export interface ChatMessageData {
  id?: string;
  role: "user" | "assistant";
  parts: ChatMessagePart[];
  createdAt?: string;
  references?: ChatReference[];
  loading?: boolean;
  status?:
    | "pending"
    | "running"
    | "completed"
    | "cancelled"
    | "failed"
    | "interrupted"
    | "waiting_user"
    | "superseded";
  taskId?: string;
  outcome?: TaskOutcomeData;
}

interface ChatMessageProps {
  message: ChatMessageData;
  editable?: boolean;
  onEdit?: () => void;
  onRegenerate?: () => void;
  onBranch?: () => void;
  branching?: boolean;
  onOpenReferences?: (references: ChatReference[]) => void;
  onAction?: (action: A2UIAction) => void;
  onInteraction?: (interaction: ChatInteraction) => void;
}

function messageText(message: ChatMessageData): string {
  const text = message.parts
    .filter((part) => part.type === "text")
    .map((part) => String(part.content))
    .join("\n\n")
    .trim();
  const references = (message.references || []).filter(
    (reference) => reference.url,
  );
  if (!references.length) return text;
  return [
    text,
    "\n参考来源：",
    ...references.map((item) => `${item.title} · ${item.url || item.source}`),
  ]
    .filter(Boolean)
    .join("\n");
}

async function copyPlainText(content: string): Promise<void> {
  let clipboardError: unknown;
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(content);
      return;
    } catch (error) {
      clipboardError = error;
    }
  }

  const textarea = document.createElement("textarea");
  const previouslyFocused = document.activeElement;
  textarea.value = content;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.inset = "0";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);
  try {
    if (!document.execCommand("copy")) {
      throw clipboardError instanceof Error
        ? clipboardError
        : new Error("浏览器未允许访问剪贴板");
    }
  } finally {
    textarea.remove();
    if (previouslyFocused instanceof HTMLElement) previouslyFocused.focus();
  }
}

function formatMessageTime(value?: string) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (number: number) => String(number).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

/**
 * Renders a single chat message with mixed text and widget parts.
 * User messages are right-aligned, assistant messages are left-aligned.
 */
export function ChatMessage({
  message,
  editable = false,
  onEdit,
  onRegenerate,
  onBranch,
  branching = false,
  onOpenReferences,
  onAction,
  onInteraction,
}: ChatMessageProps) {
  const isUser = message.role === "user";
  const [copyStatus, setCopyStatus] = useState<
    "idle" | "copied" | "failed"
  >("idle");
  const references = (message.references || []).filter(
    (reference) => reference.url,
  );
  const timestamp = formatMessageTime(message.createdAt);
  const copyContent = messageText(message);

  const copyMessage = async () => {
    if (!copyContent) return;
    try {
      await copyPlainText(copyContent);
      setCopyStatus("copied");
    } catch {
      setCopyStatus("failed");
    }
    window.setTimeout(() => setCopyStatus("idle"), 1600);
  };

  const copyLabel =
    copyStatus === "copied"
      ? "已复制"
      : copyStatus === "failed"
        ? "复制失败"
        : isUser
          ? "复制消息"
          : "复制回复";

  return (
    <div
      className={cn(
        "flex min-w-0 gap-3 px-4 py-3",
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
          "group flex w-full min-w-0 flex-col gap-2",
          isUser
            ? "max-w-[85%] items-end"
            : "max-w-[calc(100%-2.75rem)] items-start",
        )}
      >
        <div className="flex w-full min-w-0 max-w-full items-start gap-2">
          <div
            className={cn(
              "flex w-full min-w-0 max-w-full flex-col gap-2",
              isUser ? "items-end" : "items-start",
            )}
          >
            {message.parts.map((part, i) => {
              if (part.type === "text") {
                if (!isUser) {
                  return (
                    <div
                      key={i}
                      className="w-full min-w-0 max-w-full overflow-hidden rounded-lg"
                    >
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
                    <MarkdownInline text={String(part.content)} />
                  </div>
                );
              }
              if (part.type === "widget") {
                return (
                  <div
                    key={i}
                    className="w-full min-w-0 max-w-full overflow-hidden rounded-lg"
                  >
                    <WidgetRenderer html={String(part.content)} />
                  </div>
                );
              }
              if (part.type === "a2ui") {
                const messages = Array.isArray(part.content)
                  ? part.content
                  : [part.content];
                return (
                  <div
                    key={i}
                    className="w-full min-w-0 max-w-full overflow-hidden rounded-lg"
                  >
                    <A2UIRenderer
                      messages={messages as A2UIMessage[]}
                      onAction={onAction}
                    />
                  </div>
                );
              }
              if (part.type === "artifact") {
                return (
                  <div
                    key={i}
                    className="w-full min-w-0 max-w-full overflow-hidden rounded-lg"
                  >
                    <LazyArtifactCard artifact={part.content as Artifact} />
                  </div>
                );
              }
              if (part.type === "interaction") {
                const interaction = part.content as ChatInteraction;
                const answered = interaction.status !== "pending";
                return (
                  <div
                    key={i}
                    className="w-full rounded-xl border border-primary/30 bg-primary/5 p-4"
                  >
                    <div className="text-sm font-medium">
                      {interaction.question}
                    </div>
                    {interaction.kind === "tool_confirmation" &&
                      interaction.tool?.tool_name && (
                        <div className="mt-2 rounded-md bg-background/70 px-3 py-2 text-xs text-muted-foreground">
                          工具：{interaction.tool.tool_name}
                          {interaction.tool.args && (
                            <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-[11px]">
                              {JSON.stringify(interaction.tool.args, null, 2)}
                            </pre>
                          )}
                        </div>
                      )}
                    <div className="mt-3 flex flex-wrap gap-2">
                      {interaction.options.map((option) => (
                        <Button
                          key={option.id}
                          size="sm"
                          variant={
                            interaction.selected_option === option.id
                              ? "default"
                              : "outline"
                          }
                          disabled={answered}
                          onClick={() =>
                            onInteraction?.({
                              ...interaction,
                              selected_option: option.id,
                            })
                          }
                        >
                          {option.label}
                        </Button>
                      ))}
                    </div>
                    {answered && (
                      <p className="mt-2 text-xs text-muted-foreground">
                        已提交选择，Agent 将继续执行。
                      </p>
                    )}
                  </div>
                );
              }
              return null;
            })}

            {message.loading && (
              <div className="flex animate-in items-center gap-2 rounded-lg bg-muted px-4 py-2 fade-in duration-300">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm text-muted-foreground">
                  Agent 正在处理
                </span>
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
        <div
          className={cn(
            "flex items-center gap-1.5 text-xs text-muted-foreground",
            isUser && "justify-end",
          )}
        >
          {timestamp && (!message.loading || isUser) && (
            <span className="mr-1 tabular-nums">{timestamp}</span>
          )}
          {!isUser && !message.loading && message.outcome && (
            <span className="mr-1 rounded bg-muted px-1.5 py-0.5">
              {message.outcome.outcome === "satisfied"
                ? "任务已满足"
                : message.outcome.outcome === "partial"
                  ? "部分完成"
                  : message.outcome.outcome === "needs_input"
                    ? "等待补充信息"
                    : message.outcome.outcome === "data_unavailable"
                      ? "公开数据不可得"
                      : "任务未完成"}
            </span>
          )}
          {isUser && copyContent && (
            <button
              type="button"
              onClick={() => void copyMessage()}
              aria-label={copyLabel}
              title={copyLabel}
              className="rounded-md p-1.5 transition-colors hover:bg-accent hover:text-foreground"
            >
              {copyStatus === "copied" ? (
                <Check className="h-3.5 w-3.5 text-emerald-600" />
              ) : (
                <Copy
                  className={cn(
                    "h-3.5 w-3.5",
                    copyStatus === "failed" && "text-destructive",
                  )}
                />
              )}
            </button>
          )}
        </div>
        {!isUser && !message.loading && message.parts.length > 0 && (
          <div className="flex items-center gap-1 text-muted-foreground">
            {copyContent && (
              <button
                type="button"
                onClick={() => void copyMessage()}
                aria-label={copyLabel}
                title={copyLabel}
                className="rounded-md p-1.5 transition-colors hover:bg-accent hover:text-foreground"
              >
                {copyStatus === "copied" ? (
                  <Check className="h-4 w-4 text-emerald-600" />
                ) : (
                  <Copy
                    className={cn(
                      "h-4 w-4",
                      copyStatus === "failed" && "text-destructive",
                    )}
                  />
                )}
              </button>
            )}
            <button
              type="button"
              onClick={onRegenerate}
              disabled={!onRegenerate}
              aria-label="重新生成"
              title="重新生成"
              className="rounded-md p-1.5 transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
            {onBranch && (
              <button
                type="button"
                onClick={onBranch}
                disabled={branching}
                aria-label="在新对话中分支"
                title="在新对话中分支"
                className="rounded-md p-1.5 transition-colors hover:bg-accent hover:text-foreground disabled:cursor-wait disabled:opacity-50"
              >
                {branching ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <GitFork className="h-4 w-4" />
                )}
              </button>
            )}
            {references.length > 0 && (
              <>
                <span className="mx-1 h-4 w-px bg-border" aria-hidden="true" />
                <button
                  type="button"
                  onClick={() => onOpenReferences?.(references)}
                  className="flex items-center gap-1.5 rounded-full border border-border/80 bg-background/60 px-2.5 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-primary"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Reference
                  <span className="rounded-full bg-primary/10 px-1.5 text-[10px]">
                    {references.length}
                  </span>
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
