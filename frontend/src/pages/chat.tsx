import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import {
  ChatMessage,
  type ChatMessagePart,
  type ChatMessageData,
  type ChatInteraction,
  type ChatReference,
  type TaskOutcomeData,
} from "@/components/chat/ChatMessage";
import type { A2UIAction, A2UIMessage } from "@/components/chat/A2UIRenderer";
import { getLLMConfig, type LLMConfig } from "@/api";
import {
  a2uiMessageKey,
  a2uiMessages,
  conversationSearchText,
  hasRunningTask,
  hydrateServerConversation,
  isArtifactEmbeddedInA2UI,
  isRunningTaskMessage,
  latestConversationReferences,
  loadStore,
  newConversation,
  saveStore,
  titleFromMessages,
  type ChatStore,
  type Conversation,
} from "@/features/chat/store";
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
  Square,
  Trash2,
  X,
} from "lucide-react";

const SUGGESTIONS = [
  { label: "分析 ETF 510300", text: "分析 ETF 510300，适合我的短中期交易吗？" },
  {
    label: "回测 ETF 510300",
    text: "回测 ETF 510300 最近三年表现，优先控制最大回撤",
  },
  { label: "查询实时行情", text: "查询 600519 的实时行情" },
  { label: "对比两个基金", text: "对比 ETF 510300 和 ETF 159915 的近期走势" },
  { label: "Agent 能做什么？", text: "你能帮我做哪些事情？" },
];

const TERMINAL_TASK_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "interrupted",
  "waiting_user",
  "superseded",
]);
const MAX_MISSING_TASK_RETRIES = 10;

function terminalMessageStatus(
  status: string | null,
): ChatMessageData["status"] | null {
  return status && TERMINAL_TASK_STATUSES.has(status)
    ? (status as ChatMessageData["status"])
    : null;
}

function waitForReconnect(delayMs: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    let timer = 0;
    const onAbort = () => {
      window.clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      reject(new DOMException("请求已取消", "AbortError"));
    };
    if (signal.aborted) {
      onAbort();
      return;
    }
    timer = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, delayMs);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export function ChatPage() {
  const [searchParams] = useSearchParams();
  const requestedConversationId = searchParams.get("conversation");
  const [store, setStore] = useState<ChatStore>(() => loadStore());
  const [input, setInput] = useState("");
  const [search, setSearch] = useState("");
  const [sending, setSending] = useState(false);
  const [branchingMessageId, setBranchingMessageId] = useState<string | null>(
    null,
  );
  const [llmConfig, setLlmConfig] = useState<LLMConfig | null>(null);
  const [editingMessageIndex, setEditingMessageIndex] = useState<number | null>(
    null,
  );
  const [sidebarOpen, setSidebarOpen] = useState(
    () => window.matchMedia("(min-width: 768px)").matches,
  );
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [referencesByConversation, setReferencesByConversation] = useState<
    Record<string, ChatReference[]>
  >({});
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const activeTaskIdRef = useRef<string | null>(null);
  const taskEventCursorRef = useRef<Record<string, number>>({});
  const inputComposingRef = useRef(false);

  const activeConversation = useMemo(
    () =>
      store.conversations.find(
        (item) => item.conversationId === store.activeId,
      ) || store.conversations[0],
    [store],
  );

  useEffect(() => {
    void getLLMConfig()
      .then(setLlmConfig)
      .catch(() => {
        // Chat still uses the backend's environment-owned model if status loading fails.
      });
  }, []);

  useEffect(() => {
    const desktop = window.matchMedia("(min-width: 768px)");
    const syncSidebar = (event: MediaQueryListEvent) => {
      if (event.matches) setSidebarOpen(true);
    };
    desktop.addEventListener("change", syncSidebar);
    return () => desktop.removeEventListener("change", syncSidebar);
  }, []);

  // `sending` protects the page's single live stream from concurrent sends.
  // Navigation remains available because stream events are written back by
  // conversationId/taskId rather than by whichever conversation is visible.
  const activeConversationTaskId = [
    ...(activeConversation?.messages || []),
  ]
    .reverse()
    .find(isRunningTaskMessage)?.taskId;
  const activeConversationSending = Boolean(activeConversationTaskId);

  const selectedReferences = activeConversation
    ? referencesByConversation[activeConversation.conversationId] ||
      latestConversationReferences(activeConversation)
    : [];

  const visibleConversations = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return store.conversations
      .filter((item) => !query || conversationSearchText(item).includes(query))
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }, [search, store.conversations]);

  useEffect(() => saveStore(store), [store]);

  useEffect(() => {
    if (!requestedConversationId) return;
    setStore((current) =>
      current.conversations.some(
        (conversation) =>
          conversation.conversationId === requestedConversationId,
      ) && current.activeId !== requestedConversationId
        ? { ...current, activeId: requestedConversationId }
        : current,
    );
  }, [requestedConversationId]);

  useEffect(() => {
    let mounted = true;
    void fetch("/api/chat/conversations")
      .then((response) => (response.ok ? response.json() : null))
      .then((payload: { conversations?: Record<string, unknown>[] } | null) => {
        if (!mounted || !payload?.conversations?.length) return;
        const conversations = payload.conversations.map(
          hydrateServerConversation,
        );
        setStore((current) => {
          const serverIds = new Set(
            conversations.map((conversation) => conversation.conversationId),
          );
          const localOnly = current.conversations.filter(
            (conversation) => !serverIds.has(conversation.conversationId),
          );
          const merged = conversations.map((serverConversation) => {
            const localConversation = current.conversations.find(
              (conversation) =>
                conversation.conversationId ===
                serverConversation.conversationId,
            );
            // A live task owns its local message until its stream finishes. The
            // initial server-history request may otherwise replace it with a
            // snapshot, after which the same SSE events are appended again.
            return localConversation && hasRunningTask(localConversation)
              ? localConversation
              : serverConversation;
          });
          merged.push(...localOnly);
          const activeId = merged.some(
            (conversation) => conversation.conversationId === current.activeId,
          )
            ? current.activeId
            : conversations[0].conversationId;
          return { activeId, conversations: merged };
        });
      })
      .catch(() => {
        // Keep the local browser history when the backend history is unavailable.
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [activeConversation?.messages]);

  useEffect(() => {
    if (!sourcesOpen || selectedReferences.length > 0) return;
    setSourcesOpen(false);
  }, [
    activeConversation?.conversationId,
    selectedReferences.length,
    sourcesOpen,
  ]);

  const updateConversation = useCallback(
    (conversationId: string, update: (item: Conversation) => Conversation) => {
      setStore((current) => ({
        ...current,
        conversations: current.conversations.map((item) =>
          item.conversationId === conversationId ? update(item) : item,
        ),
      }));
    },
    [],
  );

  const startNewConversation = () => {
    const conversation = newConversation();
    setStore((current) => ({
      activeId: conversation.conversationId,
      conversations: [conversation, ...current.conversations],
    }));
    setInput("");
    setEditingMessageIndex(null);
    setSidebarOpen(true);
  };

  const branchConversation = async (messageId: string) => {
    if (sending || branchingMessageId || !activeConversation) return;
    const sourceConversationId = activeConversation.conversationId;
    setBranchingMessageId(messageId);
    try {
      const response = await fetch(
        `/api/chat/conversations/${encodeURIComponent(sourceConversationId)}/branches`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ through_message_id: messageId }),
        },
      );
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as {
          detail?: string;
        } | null;
        throw new Error(
          payload?.detail || `创建分支失败（${response.status}）`,
        );
      }
      const conversation = hydrateServerConversation(
        (await response.json()) as Record<string, unknown>,
      );
      setStore((current) => ({
        activeId: conversation.conversationId,
        conversations: [
          conversation,
          ...current.conversations.filter(
            (item) => item.conversationId !== conversation.conversationId,
          ),
        ],
      }));
      setInput("");
      setEditingMessageIndex(null);
      setSidebarOpen(false);
    } catch (error) {
      window.alert(
        error instanceof Error ? error.message : "创建分支失败，请稍后重试。",
      );
    } finally {
      setBranchingMessageId(null);
    }
  };

  const openConversation = (conversationId: string) => {
    setStore((current) => ({ ...current, activeId: conversationId }));
    setInput("");
    setEditingMessageIndex(null);
    setSidebarOpen(false);
  };

  const removeConversation = async (conversationId: string) => {
    if (sending) return;
    if (!window.confirm("确定删除这条历史会话吗？删除后无法恢复。")) return;
    const response = await fetch(
      `/api/chat/conversations/${encodeURIComponent(conversationId)}`,
      { method: "DELETE" },
    ).catch(() => null);
    if (!response || (!response.ok && response.status !== 404)) return;
    setStore((current) => {
      const remaining = current.conversations.filter(
        (item) => item.conversationId !== conversationId,
      );
      const next = remaining.length ? remaining : [newConversation()];
      return {
        activeId:
          current.activeId === conversationId
            ? next[0].conversationId
            : current.activeId,
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
    const conversationId = renamingId;
    const title = renameValue.trim() || "新对话";
    updateConversation(renamingId, (item) => ({
      ...item,
      title,
      updatedAt: new Date().toISOString(),
    }));
    void fetch(
      `/api/chat/conversations/${encodeURIComponent(conversationId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      },
    );
    setRenamingId(null);
  };

  const appendToAssistant = useCallback(
    (
      conversationId: string,
      messageIndex: number,
      part: {
        type: ChatMessagePart["type"];
        content: ChatMessagePart["content"];
        widgetType?: string;
      },
    ) => {
      updateConversation(conversationId, (item) => {
        const messages = [...item.messages];
        const assistant = messages[messageIndex];
        if (!assistant || assistant.role !== "assistant") return item;
        // Keep the assistant loading state until the stream sends its final status.
        // A2UI surfaces can arrive progressively while the Agent is still working.
        assistant.loading = true;
        assistant.status = "running";
        if (
          part.type === "text" &&
          assistant.parts[assistant.parts.length - 1]?.type === "text"
        ) {
          const previous = assistant.parts[assistant.parts.length - 1];
          if (
            typeof previous.content === "string" &&
            typeof part.content === "string"
          ) {
            previous.content += part.content;
          }
        } else if (
          part.type === "a2ui" &&
          assistant.parts[assistant.parts.length - 1]?.type === "a2ui"
        ) {
          const previous = assistant.parts[assistant.parts.length - 1];
          const previousMessages = a2uiMessages(
            previous.content as A2UIMessage | A2UIMessage[],
          );
          const nextMessages = a2uiMessages(
            part.content as A2UIMessage | A2UIMessage[],
          );
          const existingKeys = new Set(
            assistant.parts.flatMap((existingPart) =>
              existingPart.type === "a2ui"
                ? a2uiMessages(
                    existingPart.content as A2UIMessage | A2UIMessage[],
                  ).map(a2uiMessageKey)
                : [],
            ),
          );
          previous.content = [
            ...previousMessages,
            ...nextMessages.filter((message) => {
              const key = a2uiMessageKey(message);
              if (existingKeys.has(key)) return false;
              existingKeys.add(key);
              return true;
            }),
          ];
        } else {
          if (
            part.type === "interaction" &&
            assistant.parts.some(
              (existingPart) =>
                existingPart.type === "interaction" &&
                (existingPart.content as ChatInteraction).interaction_id ===
                  (part.content as ChatInteraction).interaction_id,
            )
          ) {
            return item;
          }
          if (part.type === "a2ui") {
            const existingKeys = new Set(
              assistant.parts.flatMap((existingPart) =>
                existingPart.type === "a2ui"
                  ? a2uiMessages(
                      existingPart.content as A2UIMessage | A2UIMessage[],
                    ).map(a2uiMessageKey)
                  : [],
              ),
            );
            const uniqueMessages = a2uiMessages(
              part.content as A2UIMessage | A2UIMessage[],
            ).filter((message) => !existingKeys.has(a2uiMessageKey(message)));
            if (!uniqueMessages.length) return item;
            assistant.parts.push({ ...part, content: uniqueMessages });
            return { ...item, messages, updatedAt: new Date().toISOString() };
          }
          if (
            part.type === "artifact" &&
            (isArtifactEmbeddedInA2UI(assistant.parts, part.content) ||
              (typeof (part.content as Record<string, unknown>)?.artifact_id ===
                "string" &&
                assistant.parts.some(
                  (existingPart) =>
                    existingPart.type === "artifact" &&
                    (existingPart.content as Record<string, unknown>)
                      ?.artifact_id ===
                      (part.content as Record<string, unknown>).artifact_id,
                )))
          ) {
            return item;
          }
          assistant.parts.push(part);
        }
        return { ...item, messages, updatedAt: new Date().toISOString() };
      });
    },
    [updateConversation],
  );

  const setAssistantOutcome = useCallback(
    (conversationId: string, messageIndex: number, outcome: TaskOutcomeData) => {
      updateConversation(conversationId, (item) => {
        const messages = [...item.messages];
        const message = messages[messageIndex];
        if (!message || message.role !== "assistant") return item;
        messages[messageIndex] = { ...message, outcome };
        return { ...item, messages, updatedAt: new Date().toISOString() };
      });
    },
    [updateConversation],
  );

  const appendToTask = useCallback(
    (
      conversationId: string,
      taskId: string,
      part: {
        type: ChatMessagePart["type"];
        content: ChatMessagePart["content"];
        widgetType?: string;
      },
    ) => {
      updateConversation(conversationId, (item) => {
        const messageIndex = item.messages.findIndex(
          (message) =>
            message.role === "assistant" && message.taskId === taskId,
        );
        if (messageIndex < 0) return item;
        const messages = [...item.messages];
        const assistant = messages[messageIndex];
        assistant.loading = true;
        assistant.status = "running";
        const previous = assistant.parts[assistant.parts.length - 1];
        if (part.type === "text" && previous?.type === "text") {
          if (
            typeof previous.content === "string" &&
            typeof part.content === "string"
          ) {
            previous.content += part.content;
          }
        } else if (part.type === "a2ui" && previous?.type === "a2ui") {
          const previousMessages = a2uiMessages(
            previous.content as A2UIMessage | A2UIMessage[],
          );
          const nextMessages = a2uiMessages(
            part.content as A2UIMessage | A2UIMessage[],
          );
          const existingKeys = new Set(
            assistant.parts.flatMap((existingPart) =>
              existingPart.type === "a2ui"
                ? a2uiMessages(
                    existingPart.content as A2UIMessage | A2UIMessage[],
                  ).map(a2uiMessageKey)
                : [],
            ),
          );
          previous.content = [
            ...(previousMessages as A2UIMessage[]),
            ...nextMessages.filter((message) => {
              const key = a2uiMessageKey(message);
              if (existingKeys.has(key)) return false;
              existingKeys.add(key);
              return true;
            }),
          ];
        } else {
          if (
            part.type === "interaction" &&
            assistant.parts.some(
              (existingPart) =>
                existingPart.type === "interaction" &&
                (existingPart.content as ChatInteraction).interaction_id ===
                  (part.content as ChatInteraction).interaction_id,
            )
          ) {
            return item;
          }
          if (part.type === "a2ui") {
            const existingKeys = new Set(
              assistant.parts.flatMap((existingPart) =>
                existingPart.type === "a2ui"
                  ? a2uiMessages(
                      existingPart.content as A2UIMessage | A2UIMessage[],
                    ).map(a2uiMessageKey)
                  : [],
              ),
            );
            const uniqueMessages = a2uiMessages(
              part.content as A2UIMessage | A2UIMessage[],
            ).filter((message) => !existingKeys.has(a2uiMessageKey(message)));
            if (!uniqueMessages.length) return item;
            assistant.parts.push({ ...part, content: uniqueMessages });
            return { ...item, messages, updatedAt: new Date().toISOString() };
          }
          if (
            part.type === "artifact" &&
            (isArtifactEmbeddedInA2UI(assistant.parts, part.content) ||
              (typeof (part.content as Record<string, unknown>)?.artifact_id ===
                "string" &&
                assistant.parts.some(
                  (existingPart) =>
                    existingPart.type === "artifact" &&
                    (existingPart.content as Record<string, unknown>)
                      ?.artifact_id ===
                      (part.content as Record<string, unknown>).artifact_id,
                )))
          ) {
            return item;
          }
          assistant.parts.push(part);
        }
        return { ...item, messages, updatedAt: new Date().toISOString() };
      });
    },
    [updateConversation],
  );

  const setTaskOutcome = useCallback(
    (conversationId: string, taskId: string, outcome: TaskOutcomeData) => {
      updateConversation(conversationId, (item) => ({
        ...item,
        messages: item.messages.map((message) =>
          message.role === "assistant" && message.taskId === taskId
            ? { ...message, outcome }
            : message,
        ),
        updatedAt: new Date().toISOString(),
      }));
    },
    [updateConversation],
  );

  const setAssistantReferences = useCallback(
    (
      conversationId: string,
      messageIndex: number,
      references: ChatReference[],
    ) => {
      updateConversation(conversationId, (item) => {
        const message = item.messages[messageIndex];
        if (!message || message.role !== "assistant") return item;
        const messages = [...item.messages];
        messages[messageIndex] = { ...message, references };
        return { ...item, messages, updatedAt: new Date().toISOString() };
      });
    },
    [updateConversation],
  );

  const setTaskReferences = useCallback(
    (conversationId: string, taskId: string, references: ChatReference[]) => {
      updateConversation(conversationId, (item) => {
        const messageIndex = item.messages.findIndex(
          (message) =>
            message.role === "assistant" && message.taskId === taskId,
        );
        if (messageIndex < 0) return item;
        const messages = [...item.messages];
        messages[messageIndex] = { ...messages[messageIndex], references };
        return { ...item, messages, updatedAt: new Date().toISOString() };
      });
    },
    [updateConversation],
  );

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || sending || !activeConversation) return;
      const conversationId = activeConversation.conversationId;
      const editedIndex = editingMessageIndex;
      const baseMessages =
        editedIndex === null
          ? activeConversation.messages
          : activeConversation.messages.slice(0, editedIndex);
      const supersededBaseMessages = baseMessages.map((message) => ({
        ...message,
        parts: message.parts.map((part) => {
          if (part.type !== "interaction") return part;
          const interaction = part.content as ChatInteraction;
          return interaction.status === "pending"
            ? {
                ...part,
                content: { ...interaction, status: "cancelled" as const },
              }
            : part;
        }),
      }));
      const userMessage: ChatMessageData = {
        id: crypto.randomUUID(),
        role: "user",
        createdAt: new Date().toISOString(),
        parts: [{ type: "text", content: text.trim() }],
      };
      const taskId = crypto.randomUUID();
      const assistantMessage: ChatMessageData = {
        id: crypto.randomUUID(),
        role: "assistant",
        createdAt: new Date().toISOString(),
        references: [],
        parts: [],
        loading: true,
        status: "pending",
        taskId,
      };
      const assistantIndex = baseMessages.length + 1;
      const requestId = requestIdRef.current + 1;
      const abortController = new AbortController();
      requestIdRef.current = requestId;
      abortControllerRef.current = abortController;
      activeTaskIdRef.current = taskId;
      setSending(true);
      setInput("");
      setEditingMessageIndex(null);
      updateConversation(conversationId, (item) => ({
        ...item,
        title:
          item.title === "新对话"
            ? titleFromMessages([userMessage])
            : item.title,
        messages: [...supersededBaseMessages, userMessage, assistantMessage],
        updatedAt: new Date().toISOString(),
      }));

      let lastEventId = taskEventCursorRef.current[taskId] || 0;
      let taskStatus: string | null = null;

      try {
        const appendStreamData = (data: Record<string, unknown>) => {
          if (requestIdRef.current !== requestId) return;
          if (data.text !== undefined)
            appendToAssistant(conversationId, assistantIndex, {
              type: "text",
              content: String(data.text),
            });
          if (data.html !== undefined)
            appendToAssistant(conversationId, assistantIndex, {
              type: "widget",
              content: String(data.html),
              widgetType: typeof data.type === "string" ? data.type : undefined,
            });
          if (data.a2ui !== undefined)
            appendToAssistant(conversationId, assistantIndex, {
              type: "a2ui",
              content: data.a2ui as A2UIMessage | A2UIMessage[],
            });
          if (data.artifact !== undefined)
            appendToAssistant(conversationId, assistantIndex, {
              type: "artifact",
              content: data.artifact as ChatMessagePart["content"],
            });
          if (data.interaction !== undefined)
            appendToAssistant(conversationId, assistantIndex, {
              type: "interaction",
              content: data.interaction as ChatInteraction,
            });
          if (Array.isArray(data.references))
            setAssistantReferences(
              conversationId,
              assistantIndex,
              data.references as ChatReference[],
            );
          if (data.acceptance && typeof data.acceptance === "object")
            setAssistantOutcome(
              conversationId,
              assistantIndex,
              data.acceptance as TaskOutcomeData,
            );
        };

        const consumeStream = async (response: Response) => {
          const reader = response.body?.getReader();
          if (!reader) throw new Error("没有收到流式响应");
          const decoder = new TextDecoder();
          let buffer = "";
          let pendingEventId: number | null = null;
          while (!abortController.signal.aborted) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            for (const rawLine of lines) {
              const line = rawLine.trimEnd();
              if (line.startsWith("id:")) {
                const parsedEventId = Number(line.slice(3).trim());
                pendingEventId = Number.isFinite(parsedEventId)
                  ? parsedEventId
                  : null;
                continue;
              }
              if (!line.startsWith("data:")) continue;
              if (pendingEventId !== null && pendingEventId <= lastEventId)
                continue;
              try {
                const data = JSON.parse(line.slice(5).trim()) as Record<
                  string,
                  unknown
                >;
                appendStreamData(data);
                if (pendingEventId !== null) {
                  lastEventId = pendingEventId;
                  taskEventCursorRef.current[taskId] = pendingEventId;
                }
              } catch {
                // Ignore incomplete SSE frames.
              }
            }
          }
        };

        const getTaskStatus = async () => {
          const response = await fetch(
            `/api/chat/tasks/${encodeURIComponent(taskId)}`,
            {
              signal: abortController.signal,
            },
          );
          if (response.status === 404) return null;
          if (!response.ok)
            throw new Error(`任务状态查询失败（${response.status}）`);
          return (await response.json()) as { status?: string };
        };

        let initialTaskAccepted = false;
        try {
          const initialResponse = await fetch("/api/chat/send", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message: text.trim(),
              conversation_id: conversationId,
              task_id: taskId,
              user_message_id: userMessage.id,
              assistant_message_id: assistantMessage.id,
              edit_message_id:
                editedIndex === null
                  ? undefined
                  : activeConversation.messages[editedIndex]?.id,
            }),
            signal: abortController.signal,
          });
          if (!initialResponse.ok)
            throw new Error(`请求失败（${initialResponse.status}）`);
          initialTaskAccepted = true;
          await consumeStream(initialResponse);
        } catch (error) {
          if (abortController.signal.aborted) throw error;
          // The task is durable on the server. If the first stream drops, resume it below.
          const task = await getTaskStatus();
          if (!task && !initialTaskAccepted) throw error;
          taskStatus = task?.status || null;
        }

        let reconnectAttempt = 0;
        let missingTaskRetries = 0;
        while (!abortController.signal.aborted) {
          try {
            const task = await getTaskStatus();
            taskStatus = task?.status || null;
            if (!task) {
              missingTaskRetries += 1;
              if (missingTaskRetries >= MAX_MISSING_TASK_RETRIES) {
                taskStatus = "interrupted";
                break;
              }
              await waitForReconnect(
                Math.min(250 * 2 ** reconnectAttempt, 3000),
                abortController.signal,
              );
              reconnectAttempt = Math.min(reconnectAttempt + 1, 4);
              continue;
            }
            missingTaskRetries = 0;
            if (terminalMessageStatus(taskStatus)) break;

            await waitForReconnect(
              Math.min(250 * 2 ** reconnectAttempt, 3000),
              abortController.signal,
            );
            const response = await fetch(
              `/api/chat/tasks/${encodeURIComponent(taskId)}/stream`,
              {
                signal: abortController.signal,
                headers: lastEventId
                  ? { "Last-Event-ID": String(lastEventId) }
                  : undefined,
              },
            );
            if (!response.ok)
              throw new Error(`流式重连失败（${response.status}）`);
            await consumeStream(response);
            reconnectAttempt = 0;
          } catch (error) {
            if (abortController.signal.aborted) throw error;
            reconnectAttempt = Math.min(reconnectAttempt + 1, 4);
            await waitForReconnect(
              Math.min(250 * 2 ** reconnectAttempt, 3000),
              abortController.signal,
            );
          }
        }
      } catch (error) {
        if (abortController.signal.aborted) {
          appendToAssistant(conversationId, assistantIndex, {
            type: "text",
            content: "\n\n已停止生成。",
          });
        } else {
          appendToAssistant(conversationId, assistantIndex, {
            type: "text",
            content: `请求失败：${error instanceof Error ? error.message : "未知错误"}`,
          });
        }
      } finally {
        const terminalStatus = abortController.signal.aborted
          ? "cancelled"
          : terminalMessageStatus(taskStatus) || "interrupted";
        updateConversation(conversationId, (item) => ({
          ...item,
          messages: item.messages.map((message, index) =>
            index === assistantIndex && message.role === "assistant"
              ? {
                  ...message,
                  loading: false,
                  status: terminalStatus,
                }
              : message,
          ),
        }));
        if (requestIdRef.current === requestId) {
          abortControllerRef.current = null;
          activeTaskIdRef.current = null;
          setSending(false);
        }
      }
    },
    [
      activeConversation,
      appendToAssistant,
      editingMessageIndex,
      sending,
      setAssistantReferences,
      setAssistantOutcome,
      updateConversation,
    ],
  );

  const stopGeneration = useCallback(() => {
    const taskId = activeConversationTaskId;
    if (taskId) {
      void fetch(`/api/chat/tasks/${encodeURIComponent(taskId)}/cancel`, {
        method: "POST",
      }).catch(() => {
        // Keep the local stream available to report a backend cancellation failure.
      });
    }
    if (taskId && activeTaskIdRef.current === taskId) {
      abortControllerRef.current?.abort();
    }
  }, [activeConversationTaskId]);

  const resumedTaskId = useMemo(() => {
    const pendingMessage = [...(activeConversation?.messages || [])]
      .reverse()
      .find(isRunningTaskMessage);
    return pendingMessage?.taskId || null;
  }, [activeConversation?.messages]);

  useEffect(() => {
    if (!resumedTaskId || abortControllerRef.current) return;
    activeTaskIdRef.current = resumedTaskId;
    setSending(true);

    const conversationId = activeConversation?.conversationId;
    if (!conversationId) return;
    const reconnectController = new AbortController();
    abortControllerRef.current = reconnectController;
    let disposed = false;
    let lastEventId = taskEventCursorRef.current[resumedTaskId] || 0;
    let missingTaskRetries = 0;

    const waitBeforeReconnect = () =>
      new Promise<void>((resolve) => window.setTimeout(resolve, 1000));

    const consumeReconnectStream = async () => {
      let terminalStatus: ChatMessageData["status"] | null = null;
      while (!disposed && !reconnectController.signal.aborted) {
        try {
          const response = await fetch(
            `/api/chat/tasks/${encodeURIComponent(resumedTaskId)}/stream`,
            {
              signal: reconnectController.signal,
              headers: lastEventId
                ? { "Last-Event-ID": String(lastEventId) }
                : undefined,
            },
          );
          if (response.status === 404) {
            missingTaskRetries += 1;
            if (missingTaskRetries >= MAX_MISSING_TASK_RETRIES) {
              terminalStatus = "interrupted";
              break;
            }
            await waitBeforeReconnect();
            continue;
          }
          missingTaskRetries = 0;
          if (!response.ok) throw new Error(`重连失败（${response.status}）`);
          const reader = response.body?.getReader();
          if (!reader) throw new Error("重连没有收到流式响应");
          const decoder = new TextDecoder();
          let buffer = "";
          let pendingEventId: number | null = null;
          while (!disposed && !reconnectController.signal.aborted) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            for (const line of lines) {
              if (line.startsWith("id:")) {
                const parsedEventId = Number(line.slice(3).trim());
                pendingEventId = Number.isFinite(parsedEventId)
                  ? parsedEventId
                  : null;
                continue;
              }
              if (!line.startsWith("data:")) continue;
              if (pendingEventId !== null && pendingEventId <= lastEventId)
                continue;
              try {
                const data = JSON.parse(line.slice(5).trim());
                if (data.text !== undefined) {
                  appendToTask(conversationId, resumedTaskId, {
                    type: "text",
                    content: data.text,
                  });
                }
                if (data.html !== undefined) {
                  appendToTask(conversationId, resumedTaskId, {
                    type: "widget",
                    content: data.html,
                    widgetType: data.type,
                  });
                }
                if (data.a2ui !== undefined) {
                  appendToTask(conversationId, resumedTaskId, {
                    type: "a2ui",
                    content: data.a2ui,
                  });
                }
                if (data.artifact !== undefined) {
                  appendToTask(conversationId, resumedTaskId, {
                    type: "artifact",
                    content: data.artifact,
                  });
                }
                if (data.interaction !== undefined) {
                  appendToTask(conversationId, resumedTaskId, {
                    type: "interaction",
                    content: data.interaction as ChatInteraction,
                  });
                }
                if (Array.isArray(data.references)) {
                  setTaskReferences(
                    conversationId,
                    resumedTaskId,
                    data.references as ChatReference[],
                  );
                }
                if (data.acceptance && typeof data.acceptance === "object") {
                  setTaskOutcome(
                    conversationId,
                    resumedTaskId,
                    data.acceptance as TaskOutcomeData,
                  );
                }
                if (pendingEventId !== null) {
                  lastEventId = pendingEventId;
                  taskEventCursorRef.current[resumedTaskId] = pendingEventId;
                }
              } catch {
                // Ignore incomplete SSE frames.
              }
            }
          }
          const taskResponse = await fetch(
            `/api/chat/tasks/${encodeURIComponent(resumedTaskId)}`,
            { signal: reconnectController.signal },
          );
          if (taskResponse.status === 404) {
            missingTaskRetries += 1;
            if (missingTaskRetries >= MAX_MISSING_TASK_RETRIES) {
              terminalStatus = "interrupted";
              break;
            }
            await waitBeforeReconnect();
            continue;
          }
          if (!taskResponse.ok)
            throw new Error(`任务状态查询失败（${taskResponse.status}）`);
          missingTaskRetries = 0;
          const task = (await taskResponse.json()) as { status?: string };
          if (task.status && TERMINAL_TASK_STATUSES.has(task.status)) {
            terminalStatus = task.status as ChatMessageData["status"];
            break;
          }
        } catch {
          if (reconnectController.signal.aborted || disposed) break;
          await waitBeforeReconnect();
        }
      }
      if (
        !disposed &&
        reconnectController.signal.aborted &&
        !terminalStatus
      ) {
        terminalStatus = "cancelled";
      }
      if (!disposed) {
        if (terminalStatus) {
          updateConversation(conversationId, (item) => ({
            ...item,
            messages: item.messages.map((message) =>
              message.role === "assistant" && message.taskId === resumedTaskId
                ? { ...message, loading: false, status: terminalStatus }
                : message,
            ),
          }));
        }
        activeTaskIdRef.current = null;
        if (abortControllerRef.current === reconnectController) {
          abortControllerRef.current = null;
        }
        setSending(false);
      }
    };

    void consumeReconnectStream();
    return () => {
      disposed = true;
      reconnectController.abort();
      if (abortControllerRef.current === reconnectController) {
        abortControllerRef.current = null;
        if (activeTaskIdRef.current === resumedTaskId) {
          activeTaskIdRef.current = null;
        }
        setSending(false);
      }
    };
  }, [
    activeConversation?.conversationId,
    appendToTask,
    resumedTaskId,
    setTaskReferences,
  ]);

  const beginEdit = useCallback(
    (messageIndex: number) => {
      if (sending || !activeConversation) return;
      const message = activeConversation.messages[messageIndex];
      if (!message || message.role !== "user") return;
      const lastUserIndex = activeConversation.messages.reduce(
        (lastIndex, currentMessage, currentIndex) =>
          currentMessage.role === "user" ? currentIndex : lastIndex,
        -1,
      );
      if (messageIndex !== lastUserIndex) return;
      setInput(
        message.parts
          .filter((part) => part.type === "text")
          .map((part) => part.content)
          .join("\n"),
      );
      setEditingMessageIndex(messageIndex);
    },
    [activeConversation, sending],
  );

  const cancelEdit = useCallback(() => {
    setEditingMessageIndex(null);
    setInput("");
  }, []);

  const handleA2UIAction = useCallback(async (action: A2UIAction) => {
    await fetch("/api/chat/a2ui/actions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(action),
    }).catch(() => {
      // Interactive surfaces remain usable locally if the action endpoint is unavailable.
    });
  }, []);

  const handleInteraction = useCallback(
    async (interaction: ChatInteraction) => {
      if (interaction.status !== "pending" || !activeConversation) return;
      const response = await fetch(
        `/api/chat/tasks/${encodeURIComponent(interaction.task_id)}/respond`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            interaction_id: interaction.interaction_id,
            option_id: interaction.selected_option,
          }),
        },
      );
      if (!response.ok) return;
      const result = (await response.json()) as { last_event_id?: number };
      if (
        typeof result.last_event_id === "number" &&
        Number.isFinite(result.last_event_id)
      ) {
        taskEventCursorRef.current[interaction.task_id] = Math.max(
          taskEventCursorRef.current[interaction.task_id] || 0,
          result.last_event_id,
        );
      }
      updateConversation(activeConversation.conversationId, (item) => ({
        ...item,
        messages: item.messages.map((message) =>
          message.taskId !== interaction.task_id
            ? message
            : {
                ...message,
                loading: true,
                status: "running",
                parts: message.parts.map((part) => {
                  if (part.type !== "interaction") return part;
                  const content = part.content as ChatInteraction;
                  return content.interaction_id !== interaction.interaction_id
                    ? part
                    : {
                        ...part,
                        content: {
                          ...content,
                          status: "answered",
                          selected_option: interaction.selected_option,
                        },
                      };
                }),
              },
        ),
        updatedAt: new Date().toISOString(),
      }));
      setSending(true);
    },
    [activeConversation, updateConversation],
  );

  const regenerateMessage = useCallback(
    (messageIndex: number) => {
      if (sending || !activeConversation) return;
      const previousUser = [...activeConversation.messages]
        .slice(0, messageIndex)
        .reverse()
        .find((message) => message.role === "user");
      if (!previousUser) return;
      const text = previousUser.parts
        .filter((part) => part.type === "text")
        .map((part) => String(part.content))
        .join("\n")
        .trim();
      if (text) void sendMessage(text);
    },
    [activeConversation, sendMessage, sending],
  );

  const openReferences = useCallback(
    (references: ChatReference[]) => {
      if (!activeConversation) return;
      setReferencesByConversation((current) => ({
        ...current,
        [activeConversation.conversationId]: references,
      }));
      setSourcesOpen(true);
    },
    [activeConversation],
  );

  const messages = activeConversation?.messages || [];

  return (
    <div className="flex h-full min-h-0 bg-transparent">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-[#020817]/45 backdrop-blur-sm md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        className={`absolute inset-y-0 left-0 z-30 flex w-[292px] shrink-0 flex-col border-r border-white/80 bg-white/82 shadow-xl backdrop-blur-2xl transition-transform md:relative md:translate-x-0 md:shadow-none ${sidebarOpen ? "translate-x-0" : "-translate-x-full md:-translate-x-full"}`}
      >
        <div className="flex h-16 items-center justify-between border-b px-4">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 via-blue-500 to-violet-500 text-white shadow-lg shadow-blue-500/20">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <p className="text-sm font-semibold">Agent 对话</p>
              <p className="text-[11px] text-muted-foreground">
                研究与模拟交易
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="p-3">
          <Button
            className="w-full justify-start gap-2"
            onClick={startNewConversation}
          >
            <Plus className="h-4 w-4" />
            新对话
          </Button>
        </div>
        <div className="px-3 pb-3">
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="搜索标题或对话内容"
              className="pl-9"
            />
          </div>
        </div>
        <div className="flex items-center justify-between px-4 pb-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          <span>最近会话</span>
          <span>{store.conversations.length}</span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
          {visibleConversations.length === 0 ? (
            <p className="px-3 py-6 text-center text-xs text-muted-foreground">
              没有匹配的会话
            </p>
          ) : (
            visibleConversations.map((conversation) => (
              <div
                key={conversation.conversationId}
                className={`group mb-1 flex items-center gap-1 rounded-xl border px-2 py-2 transition-colors ${conversation.conversationId === store.activeId ? "border-blue-100 bg-blue-50/80 text-primary" : "border-transparent hover:border-blue-100/60 hover:bg-white"}`}
              >
                {renamingId === conversation.conversationId ? (
                  <Input
                    autoFocus
                    value={renameValue}
                    onChange={(event) => setRenameValue(event.target.value)}
                    onBlur={commitRename}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") commitRename();
                      if (event.key === "Escape") setRenamingId(null);
                    }}
                    className="h-8 min-w-0 text-sm"
                  />
                ) : (
                  <button
                    className="min-w-0 flex-1 text-left"
                    onClick={() =>
                      openConversation(conversation.conversationId)
                    }
                  >
                    <span className="block truncate text-sm">
                      {conversation.title}
                    </span>
                    <span className="block text-[11px] text-muted-foreground">
                      {hasRunningTask(conversation)
                        ? "Agent 正在后台执行"
                        : formatRelativeTime(conversation.updatedAt)}
                    </span>
                  </button>
                )}
                <div className="hidden shrink-0 gap-0.5 group-hover:flex">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => beginRename(conversation)}
                    aria-label="重命名"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-muted-foreground hover:text-destructive"
                    onClick={() =>
                      removeConversation(conversation.conversationId)
                    }
                    aria-label="删除"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
        <div className="border-t px-4 py-3 text-[11px] leading-relaxed text-muted-foreground">
          <div className="flex items-center gap-2">
            <Archive className="h-3.5 w-3.5" />
            会话与任务状态会自动保存
          </div>
          <p className="mt-1">刷新后可继续恢复；研究结论不构成收益承诺。</p>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center gap-3 border-b border-white/80 bg-white/60 px-4 backdrop-blur-xl md:px-6">
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-5 w-5" />
          </Button>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h1 className="truncate text-base font-semibold">
                {activeConversation?.title || "新对话"}
              </h1>
            </div>
            <p className="text-xs text-muted-foreground">
              {messages.length
                ? `${Math.ceil(messages.length / 2)} 轮对话`
                : "从一个问题开始"}
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={startNewConversation}
            className="hidden gap-1.5 sm:flex"
          >
            <Plus className="h-4 w-4" />
            新对话
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="hidden md:flex"
            onClick={() => setSidebarOpen((open) => !open)}
            aria-label="收起侧栏"
          >
            {sidebarOpen ? (
              <ChevronLeft className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </Button>
        </header>
        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <EmptyState onSuggestion={(text) => sendMessage(text)} />
          ) : (
            <div className="mx-auto max-w-5xl px-3 py-6 md:px-8">
              {messages.map((message, index) => (
                <ChatMessage
                  key={`${activeConversation.conversationId}-${index}`}
                  message={message}
                  editable={
                    !sending &&
                    message.role === "user" &&
                    index ===
                      messages.reduce(
                        (lastIndex, currentMessage, currentIndex) =>
                          currentMessage.role === "user"
                            ? currentIndex
                            : lastIndex,
                        -1,
                      )
                  }
                  onEdit={() => beginEdit(index)}
                  onRegenerate={
                    message.role === "assistant"
                      ? () => regenerateMessage(index)
                      : undefined
                  }
                  onBranch={
                    !sending &&
                    message.role === "assistant" &&
                    !message.loading &&
                    (!message.status || message.status === "completed") &&
                    message.parts.length > 0 &&
                    message.id
                      ? () => void branchConversation(message.id!)
                      : undefined
                  }
                  branching={branchingMessageId === message.id}
                  onOpenReferences={openReferences}
                  onAction={handleA2UIAction}
                  onInteraction={handleInteraction}
                />
              ))}
            </div>
          )}
        </div>
        <div className="border-t border-white/80 bg-white/72 p-3 backdrop-blur-xl md:p-5">
          <div className="mx-auto max-w-5xl">
            {editingMessageIndex !== null && (
              <div className="mb-2 flex items-center justify-between rounded-lg bg-primary/5 px-3 py-2 text-xs text-muted-foreground">
                <span>正在编辑最后一条消息，提交后会重新生成回复</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={cancelEdit}
                  className="h-7 px-2"
                >
                  <X className="h-3.5 w-3.5" />
                  取消
                </Button>
              </div>
            )}
            <div className="flex items-end gap-2 rounded-2xl border border-blue-100/80 bg-white/85 p-2 shadow-[0_16px_40px_rgba(8,26,74,0.1)]">
              <div
                className="hidden max-w-[180px] shrink-0 items-center gap-1.5 border-r pr-2 text-xs sm:flex"
                title="模型由 LLM_MODEL 环境变量配置"
              >
                <span className="text-muted-foreground">模型</span>
                <code className="truncate font-medium">
                  {llmConfig?.model || "环境配置"}
                </code>
              </div>
              <Input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onCompositionStart={() => {
                  inputComposingRef.current = true;
                }}
                onCompositionEnd={() => {
                  window.setTimeout(() => {
                    inputComposingRef.current = false;
                  }, 0);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    if (
                      inputComposingRef.current ||
                      event.nativeEvent.isComposing ||
                      event.keyCode === 229
                    ) {
                      return;
                    }
                    event.preventDefault();
                    void sendMessage(input);
                  }
                }}
                placeholder="分析股票、ETF/LOF，或查询场外基金净值、费率与同类筛选"
                disabled={sending}
                className="border-0 shadow-none focus-visible:ring-0"
              />
              {activeConversationSending ? (
                <Button
                  variant="destructive"
                  onClick={stopGeneration}
                  aria-label="停止生成"
                  title="停止生成"
                  size="icon"
                  className="shrink-0"
                >
                  <Square className="h-4 w-4 fill-current" />
                </Button>
              ) : (
                <Button
                  onClick={() => void sendMessage(input)}
                  disabled={sending || !input.trim()}
                  size="icon"
                  className="shrink-0"
                  aria-label={
                    editingMessageIndex !== null ? "重新生成" : "发送"
                  }
                >
                  <Send className="h-4 w-4" />
                </Button>
              )}
            </div>
            <p className="mt-2 text-center text-[11px] text-muted-foreground">
              Agent 会根据问题自动选择数据工具；内容仅供研究和模拟交易参考
            </p>
          </div>
        </div>
      </main>
      {sourcesOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/20 md:hidden"
          onClick={() => setSourcesOpen(false)}
        />
      )}
      {sourcesOpen && (
        <aside className="fixed inset-y-0 right-0 z-40 flex w-[min(92vw,340px)] shrink-0 flex-col border-l bg-card shadow-2xl md:relative md:z-0 md:w-[320px] md:shadow-none">
          <div className="flex h-14 shrink-0 items-center justify-between border-b px-4">
            <div>
              <p className="text-sm font-semibold">Reference</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {selectedReferences.length} 个参考链接
              </p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSourcesOpen(false)}
              aria-label="关闭数据来源"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            <div className="space-y-1.5">
              {selectedReferences.map((reference, index) => (
                <article
                  key={`${reference.url || reference.title}-${index}`}
                  className="rounded-lg border border-border/70 bg-background/50 px-3 py-2 transition-colors hover:border-primary/40 hover:bg-primary/[0.03]"
                >
                  <div className="flex items-start gap-2">
                    <div className="min-w-0 flex-1">
                      <a
                        href={reference.url}
                        target="_blank"
                        rel="noreferrer"
                        className="line-clamp-2 text-sm font-medium leading-5 text-foreground hover:text-primary"
                      >
                        {reference.title}
                      </a>
                      {reference.snippet && (
                        <p className="mt-1 line-clamp-1 text-xs leading-5 text-muted-foreground">
                          {reference.snippet}
                        </p>
                      )}
                    </div>
                    <ChevronRight className="mt-1 h-4 w-4 shrink-0 text-muted-foreground" />
                  </div>
                </article>
              ))}
            </div>
          </div>
        </aside>
      )}
    </div>
  );
}

function EmptyState({
  onSuggestion,
}: {
  onSuggestion: (text: string) => void;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center overflow-y-auto px-4 py-10">
      <div className="relative">
        <div className="absolute inset-0 rounded-3xl bg-blue-500/25 blur-2xl" />
        <div className="relative flex h-16 w-16 items-center justify-center rounded-3xl border border-white bg-gradient-to-br from-cyan-400 via-blue-500 to-violet-500 text-white shadow-xl shadow-blue-500/20">
          <MessageSquare className="h-7 w-7" />
        </div>
      </div>
      <p className="eyebrow mt-6">AI Research Workspace</p>
      <h2 className="mt-2 text-center text-2xl font-semibold tracking-[-0.035em] text-[#0b1730]">
        今天想研究什么？
      </h2>
      <p className="mt-2 max-w-lg text-center text-sm leading-6 text-muted-foreground">
        查询结构化行情、核对资讯、比较基金或设计回测。Agent
        会展示研究计划、数据来源和风险边界。
      </p>
      <div className="mt-7 grid w-full max-w-2xl gap-2.5 sm:grid-cols-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion.label}
            onClick={() => onSuggestion(suggestion.text)}
            className="group rounded-2xl border border-white/90 bg-white/75 px-4 py-3.5 text-left text-sm shadow-[0_10px_30px_rgba(8,26,74,0.05)] backdrop-blur transition-all hover:-translate-y-0.5 hover:border-blue-200 hover:bg-white hover:shadow-[var(--shadow-soft)]"
          >
            <span className="font-medium text-[#172642] group-hover:text-primary">
              {suggestion.label}
            </span>
            <span className="mt-1 block text-xs text-muted-foreground">
              {suggestion.text}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function formatRelativeTime(value: string) {
  const diff = Date.now() - new Date(value).getTime();
  if (diff < 60_000) return "刚刚";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`;
  return new Date(value).toLocaleDateString("zh-CN", {
    month: "numeric",
    day: "numeric",
  });
}
