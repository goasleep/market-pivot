import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChatMessage, type ChatMessageData } from "@/components/chat/ChatMessage";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Archive,
  ChevronLeft,
  ChevronRight,
  Menu,
  MessageSquare,
  Pencil,
  Plus,
  Search,
  Send,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";

const STORAGE_KEY = "a-share-agent:chat:conversations";
const LEGACY_CURRENT_KEY = "a-share-agent:chat:current";
const LEGACY_HISTORY_KEY = "a-share-agent:chat:history";

interface Conversation {
  conversationId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessageData[];
}

interface ChatStore {
  activeId: string;
  conversations: Conversation[];
}

const SUGGESTIONS = [
  { label: "分析 ETF 510300", text: "分析 ETF 510300，适合我的短中期交易吗？" },
  { label: "查询实时行情", text: "查询 600519 的实时行情" },
  { label: "对比两个基金", text: "对比 ETF 510300 和 ETF 159915 的近期走势" },
  { label: "Agent 能做什么？", text: "你能帮我做哪些事情？" },
];

function newConversation(): Conversation {
  const now = new Date().toISOString();
  return {
    conversationId: crypto.randomUUID(),
    title: "新对话",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function titleFromMessages(messages: ChatMessageData[]) {
  const firstUser = messages.find((message) => message.role === "user");
  const text = firstUser?.parts
    .filter((part) => part.type === "text")
    .map((part) => part.content)
    .join(" ")
    .trim();
  if (!text) return "新对话";
  return text.length > 30 ? `${text.slice(0, 30)}…` : text;
}

function loadStore(): ChatStore {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as ChatStore;
      if (parsed.conversations?.length) return parsed;
    }

    // Migrate the previous split current/history format once.
    const current = JSON.parse(localStorage.getItem(LEGACY_CURRENT_KEY) || "null");
    const history = JSON.parse(localStorage.getItem(LEGACY_HISTORY_KEY) || "[]");
    const migrated = [...(Array.isArray(history) ? history : [])];
    if (current?.messages?.length && !migrated.some((item) => item.conversationId === current.conversationId)) {
      migrated.unshift({
        conversationId: current.conversationId || crypto.randomUUID(),
        title: titleFromMessages(current.messages),
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        messages: current.messages,
      });
    }
    if (migrated.length) {
      const conversations = migrated.slice(0, 50).map((item) => ({
        ...item,
        createdAt: item.createdAt || item.updatedAt || new Date().toISOString(),
      }));
      return { activeId: conversations[0].conversationId, conversations };
    }
  } catch {
    // Fall through to a clean local store.
  }
  const first = newConversation();
  return { activeId: first.conversationId, conversations: [first] };
}

function saveStore(store: ChatStore) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch {
    // Chat remains usable when browser storage is unavailable.
  }
}

export function ChatPage() {
  const [store, setStore] = useState<ChatStore>(() => loadStore());
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const activeConversation = useMemo(
    () => store.conversations.find((item) => item.conversationId === store.activeId) || store.conversations[0],
    [store]
  );

  const visibleConversations = useMemo(() => {
    const query = search.trim().toLowerCase();
    return store.conversations
      .filter((item) => !query || item.title.toLowerCase().includes(query))
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }, [search, store.conversations]);

  useEffect(() => saveStore(store), [store]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [activeConversation?.messages]);

  const updateConversation = useCallback((conversationId: string, update: (item: Conversation) => Conversation) => {
    setStore((current) => ({
      ...current,
      conversations: current.conversations.map((item) =>
        item.conversationId === conversationId ? update(item) : item
      ),
    }));
  }, []);

  const startNewConversation = () => {
    if (sending) return;
    const conversation = newConversation();
    setStore((current) => ({
      activeId: conversation.conversationId,
      conversations: [conversation, ...current.conversations],
    }));
    setInput("");
    setSidebarOpen(true);
  };

  const openConversation = (conversationId: string) => {
    if (sending) return;
    setStore((current) => ({ ...current, activeId: conversationId }));
    setSidebarOpen(false);
  };

  const removeConversation = (conversationId: string) => {
    if (sending) return;
    if (!window.confirm("确定删除这条历史会话吗？删除后无法恢复。")) return;
    setStore((current) => {
      const remaining = current.conversations.filter((item) => item.conversationId !== conversationId);
      const next = remaining.length ? remaining : [newConversation()];
      return {
        activeId: current.activeId === conversationId ? next[0].conversationId : current.activeId,
        conversations: next,
      };
    });
  };

  const beginRename = (conversation: Conversation) => {
    setRenamingId(conversation.conversationId);
    setRenameValue(conversation.title);
  };

  const commitRename = () => {
    if (!renamingId) return;
    const title = renameValue.trim() || "新对话";
    updateConversation(renamingId, (item) => ({ ...item, title, updatedAt: new Date().toISOString() }));
    setRenamingId(null);
  };

  const appendToAssistant = (conversationId: string, part: { type: "text" | "widget"; content: string; widgetType?: string }) => {
    updateConversation(conversationId, (item) => {
      const messages = [...item.messages];
      const last = messages[messages.length - 1];
      if (!last || last.role !== "assistant") return item;
      last.loading = false;
      if (part.type === "text" && last.parts[last.parts.length - 1]?.type === "text") {
        last.parts[last.parts.length - 1].content += part.content;
      } else {
        last.parts.push(part);
      }
      return { ...item, messages, updatedAt: new Date().toISOString() };
    });
  };

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || sending || !activeConversation) return;
    const conversationId = activeConversation.conversationId;
    const history = activeConversation.messages.slice(-12).map((message) => ({
      role: message.role,
      content: message.parts.filter((part) => part.type === "text").map((part) => part.content).join("\n"),
    }));
    const userMessage: ChatMessageData = { role: "user", parts: [{ type: "text", content: text.trim() }] };
    const assistantMessage: ChatMessageData = { role: "assistant", parts: [], loading: true };
    setSending(true);
    setInput("");
    updateConversation(conversationId, (item) => ({
      ...item,
      title: item.title === "新对话" ? titleFromMessages([userMessage]) : item.title,
      messages: [...item.messages, userMessage, assistantMessage],
      updatedAt: new Date().toISOString(),
    }));

    try {
      const response = await fetch("/api/chat/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text.trim(), history, conversation_id: conversationId }),
      });
      if (!response.ok) throw new Error(`请求失败（${response.status}）`);
      const reader = response.body?.getReader();
      if (!reader) throw new Error("没有收到流式响应");
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          try {
            const data = JSON.parse(line.slice(5).trim());
            if (data.text !== undefined) appendToAssistant(conversationId, { type: "text", content: data.text });
            if (data.html !== undefined) appendToAssistant(conversationId, { type: "widget", content: data.html, widgetType: data.type });
          } catch {
            // Ignore incomplete SSE frames.
          }
        }
      }
    } catch (error) {
      appendToAssistant(conversationId, {
        type: "text",
        content: `请求失败：${error instanceof Error ? error.message : "未知错误"}`,
      });
    } finally {
      updateConversation(conversationId, (item) => ({
        ...item,
        messages: item.messages.map((message, index) =>
          index === item.messages.length - 1 && message.role === "assistant" ? { ...message, loading: false } : message
        ),
      }));
      setSending(false);
    }
  }, [activeConversation, appendToAssistant, sending, updateConversation]);

  const messages = activeConversation?.messages || [];

  return (
    <div className="flex h-full min-h-0 bg-background">
      {sidebarOpen && <div className="fixed inset-0 z-20 bg-black/20 md:hidden" onClick={() => setSidebarOpen(false)} />}
      <aside className={`absolute inset-y-0 left-0 z-30 flex w-[292px] shrink-0 flex-col border-r bg-card transition-transform md:relative md:translate-x-0 ${sidebarOpen ? "translate-x-0" : "-translate-x-full md:-translate-x-full"}`}>
        <div className="flex h-16 items-center justify-between border-b px-4">
          <div className="flex items-center gap-2"><div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground"><Sparkles className="h-4 w-4" /></div><div><p className="text-sm font-semibold">Agent 对话</p><p className="text-[11px] text-muted-foreground">研究与模拟交易</p></div></div>
          <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setSidebarOpen(false)}><X className="h-4 w-4" /></Button>
        </div>
        <div className="p-3"><Button className="w-full justify-start gap-2" onClick={startNewConversation} disabled={sending}><Plus className="h-4 w-4" />新对话</Button></div>
        <div className="px-3 pb-3"><div className="relative"><Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" /><Input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索历史会话" className="pl-9" /></div></div>
        <div className="flex items-center justify-between px-4 pb-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground"><span>最近会话</span><span>{store.conversations.length}</span></div>
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
          {visibleConversations.length === 0 ? <p className="px-3 py-6 text-center text-xs text-muted-foreground">没有匹配的会话</p> : visibleConversations.map((conversation) => (
            <div key={conversation.conversationId} className={`group mb-1 flex items-center gap-1 rounded-lg px-2 py-2 ${conversation.conversationId === store.activeId ? "bg-primary/10 text-primary" : "hover:bg-accent"}`}>
              {renamingId === conversation.conversationId ? <Input autoFocus value={renameValue} onChange={(event) => setRenameValue(event.target.value)} onBlur={commitRename} onKeyDown={(event) => { if (event.key === "Enter") commitRename(); if (event.key === "Escape") setRenamingId(null); }} className="h-8 min-w-0 text-sm" /> : <button className="min-w-0 flex-1 text-left" onClick={() => openConversation(conversation.conversationId)}><span className="block truncate text-sm">{conversation.title}</span><span className="block text-[11px] text-muted-foreground">{formatRelativeTime(conversation.updatedAt)}</span></button>}
              <div className="hidden shrink-0 gap-0.5 group-hover:flex"><Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => beginRename(conversation)} aria-label="重命名"><Pencil className="h-3.5 w-3.5" /></Button><Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-destructive" onClick={() => removeConversation(conversation.conversationId)} aria-label="删除"><Trash2 className="h-3.5 w-3.5" /></Button></div>
            </div>
          ))}
        </div>
        <div className="border-t px-4 py-3 text-[11px] leading-relaxed text-muted-foreground"><div className="flex items-center gap-2"><Archive className="h-3.5 w-3.5" />会话自动保存在本机浏览器</div><p className="mt-1">不会同步到服务器，也不会作为投资承诺。</p></div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center gap-3 border-b px-4 md:px-6"><Button variant="ghost" size="icon" className="md:hidden" onClick={() => setSidebarOpen(true)}><Menu className="h-5 w-5" /></Button><div className="min-w-0 flex-1"><div className="flex items-center gap-2"><h1 className="truncate text-base font-semibold">{activeConversation?.title || "新对话"}</h1><Badge variant="secondary" className="hidden sm:inline-flex">短中期基金研究</Badge></div><p className="text-xs text-muted-foreground">{messages.length ? `${Math.ceil(messages.length / 2)} 轮对话` : "从一个问题开始"}</p></div><Button variant="outline" size="sm" onClick={startNewConversation} disabled={sending} className="hidden gap-1.5 sm:flex"><Plus className="h-4 w-4" />新对话</Button><Button variant="ghost" size="icon" className="hidden md:flex" onClick={() => setSidebarOpen((open) => !open)} aria-label="收起侧栏">{sidebarOpen ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}</Button></header>
        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          {messages.length === 0 ? <EmptyState onSuggestion={(text) => sendMessage(text)} /> : <div className="mx-auto max-w-4xl px-3 py-6 md:px-8">{messages.map((message, index) => <ChatMessage key={`${activeConversation.conversationId}-${index}`} message={message} />)}</div>}
        </div>
        <div className="border-t bg-card/80 p-3 md:p-5"><div className="mx-auto max-w-4xl"><div className="flex items-end gap-2 rounded-xl border bg-background p-2 shadow-sm"><Input value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendMessage(input); } }} placeholder="问我行情、历史、新闻，或让 Agent 分析一个股票 / ETF / LOF" disabled={sending} className="border-0 shadow-none focus-visible:ring-0" /><Button onClick={() => void sendMessage(input)} disabled={sending || !input.trim()} size="icon" className="shrink-0"><Send className="h-4 w-4" /></Button></div><p className="mt-2 text-center text-[11px] text-muted-foreground">Agent 会根据问题自动选择数据工具；内容仅供研究和模拟交易参考</p></div></div>
      </main>
    </div>
  );
}

function EmptyState({ onSuggestion }: { onSuggestion: (text: string) => void }) {
  return <div className="flex h-full flex-col items-center justify-center px-4"><div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10"><MessageSquare className="h-7 w-7 text-primary" /></div><h2 className="mt-4 text-lg font-semibold">今天想研究什么？</h2><p className="mt-1 text-center text-sm text-muted-foreground">行情、历史、新闻和综合分析都可以直接问</p><div className="mt-6 grid w-full max-w-xl gap-2 sm:grid-cols-2">{SUGGESTIONS.map((suggestion) => <button key={suggestion.label} onClick={() => onSuggestion(suggestion.text)} className="rounded-xl border bg-card px-4 py-3 text-left text-sm transition-colors hover:border-primary/40 hover:bg-primary/5">{suggestion.label}<span className="mt-1 block text-xs text-muted-foreground">{suggestion.text}</span></button>)}</div></div>;
}

function formatRelativeTime(value: string) {
  const diff = Date.now() - new Date(value).getTime();
  if (diff < 60_000) return "刚刚";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`;
  return new Date(value).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}
