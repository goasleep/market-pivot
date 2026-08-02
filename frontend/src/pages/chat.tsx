import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatMessage, type ChatMessageData } from "@/components/chat/ChatMessage";
import { Send, Sparkles } from "lucide-react";

const SUGGESTIONS = [
  { label: "分析 000737", text: "分析 000737" },
  { label: "查询 600519 行情", text: "查询 600519 行情" },
  { label: "查看 000858 新闻", text: "查看 000858 新闻" },
  { label: "股票 Agent 能做什么？", text: "股票 Agent 能做什么？" },
];

export function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [sending, setSending] = useState(false);
  const conversationId = useRef(crypto.randomUUID());
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = useCallback(
    (text: string) => {
      if (!text.trim() || sending) return;
      setSending(true);

      const history = messages.slice(-12).map((message) => ({
        role: message.role,
        content: message.parts
          .filter((part) => part.type === "text")
          .map((part) => part.content)
          .join("\n"),
      }));

      // Add user message
      const userMsg: ChatMessageData = {
        role: "user",
        parts: [{ type: "text", content: text }],
      };

      // Add placeholder assistant message
      const assistantMsg: ChatMessageData = {
        role: "assistant",
        parts: [],
        loading: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInput("");

      // SSE request via fetch + ReadableStream
      const url = "/api/chat/send";
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history,
          conversation_id: conversationId.current,
        }),
      })
        .then(async (res) => {
          if (!res.ok) throw new Error(`Chat failed: ${res.statusText}`);
          const reader = res.body?.getReader();
          if (!reader) throw new Error("No response body");

          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) continue;

              // Parse SSE format: event: xxx\ndata: xxx
              if (line.startsWith("event:")) {
                // Find the corresponding data line
                continue;
              }

              if (line.startsWith("data:")) {
                const dataStr = line.slice(5).trim();
                if (!dataStr) continue;

                try {
                  const data = JSON.parse(dataStr);

                  // Determine event type from data structure
                  if (data.text !== undefined) {
                    // Text event
                    setMessages((prev) => {
                      const updated = [...prev];
                      const lastMsg = updated[updated.length - 1];
                      if (lastMsg && lastMsg.role === "assistant") {
                        lastMsg.loading = false;
                        // Append text to last text part or create new
                        const lastPart = lastMsg.parts[lastMsg.parts.length - 1];
                        if (lastPart && lastPart.type === "text") {
                          lastPart.content += data.text;
                        } else {
                          lastMsg.parts.push({ type: "text", content: data.text });
                        }
                      }
                      return updated;
                    });
                  } else if (data.html !== undefined) {
                    // Widget event
                    setMessages((prev) => {
                      const updated = [...prev];
                      const lastMsg = updated[updated.length - 1];
                      if (lastMsg && lastMsg.role === "assistant") {
                        lastMsg.loading = false;
                        lastMsg.parts.push({
                          type: "widget",
                          content: data.html,
                          widgetType: data.type,
                        });
                      }
                      return updated;
                    });
                  }
                } catch {
                  // skip non-JSON lines
                }
              }
            }
          }

          // Mark loading as done
          setMessages((prev) => {
            const updated = [...prev];
            const lastMsg = updated[updated.length - 1];
            if (lastMsg && lastMsg.role === "assistant") {
              lastMsg.loading = false;
            }
            return updated;
          });
          setSending(false);
        })
        .catch((err) => {
          setMessages((prev) => {
            const updated = [...prev];
            const lastMsg = updated[updated.length - 1];
            if (lastMsg && lastMsg.role === "assistant") {
              lastMsg.loading = false;
              lastMsg.parts.push({
                type: "text",
                content: `Error: ${err.message}`,
              });
            }
            return updated;
          });
          setSending(false);
        });
    },
    [messages, sending]
  );

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b px-6 py-4">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Stock Agent</h1>
          <span className="text-xs text-muted-foreground">
            A 股研究与交易辅助
          </span>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto"
      >
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4">
            <div className="text-center">
              <BotIcon />
              <p className="mt-3 text-sm text-muted-foreground">
                告诉我股票代码和你想做的事情
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s.label}
                  onClick={() => sendMessage(s.text)}
                  className="rounded-full border px-4 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="py-4">
            {messages.map((msg, i) => (
              <ChatMessage key={i} message={msg} />
            ))}
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t p-4">
        <div className="flex items-end gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage(input);
              }
            }}
            placeholder="例如：分析 000737、查询 600519 行情、查看 000858 新闻"
            disabled={sending}
            className="flex-1"
          />
          <Button
            onClick={() => sendMessage(input)}
            disabled={sending || !input.trim()}
            size="icon"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

function BotIcon() {
  return (
    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
      <Sparkles className="h-6 w-6 text-primary" />
    </div>
  );
}
