import type { A2UIMessage } from "@/components/chat/A2UIRenderer";
import type {
  ChatMessageData,
  ChatMessagePart,
  ChatReference,
} from "@/components/chat/ChatMessage";

const STORAGE_KEY = "a-share-agent:chat:conversations";
const LEGACY_CURRENT_KEY = "a-share-agent:chat:current";
const LEGACY_HISTORY_KEY = "a-share-agent:chat:history";

export interface Conversation {
  conversationId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessageData[];
}

export interface ChatStore {
  activeId: string;
  conversations: Conversation[];
}

export function a2uiMessages(content: A2UIMessage | A2UIMessage[]): A2UIMessage[] {
  return Array.isArray(content) ? content : [content];
}

export function a2uiMessageKey(message: A2UIMessage): string {
  return JSON.stringify(message);
}

export function dedupeParts(parts: ChatMessagePart[]): ChatMessagePart[] {
  const artifactIds = new Set<string>();
  const a2uiKeys = new Set<string>();
  const result: ChatMessagePart[] = [];

  for (const part of parts) {
    if (part.type === "artifact") {
      const content = part.content;
      const artifactId =
        content && typeof content === "object" && !Array.isArray(content)
          ? (content as Record<string, unknown>).artifact_id
          : undefined;
      if (typeof artifactId === "string") {
        if (artifactIds.has(artifactId)) continue;
        artifactIds.add(artifactId);
      }
      result.push(part);
      continue;
    }

    if (part.type !== "a2ui") {
      result.push(part);
      continue;
    }

    const content = a2uiMessages(part.content as A2UIMessage | A2UIMessage[]).filter(
      (message) => {
        const key = a2uiMessageKey(message);
        if (a2uiKeys.has(key)) return false;
        a2uiKeys.add(key);
        return true;
      },
    );
    if (content.length > 0) result.push({ ...part, content });
  }

  return result;
}

export function normalizeConversation(conversation: Conversation): Conversation {
  return {
    ...conversation,
    messages: conversation.messages.map((message) => ({
      ...message,
      createdAt: message.createdAt || conversation.updatedAt || new Date().toISOString(),
      references: message.references || [],
      parts: dedupeParts(message.parts),
    })),
  };
}

export function hasRunningTask(conversation: Conversation): boolean {
  return conversation.messages.some(
    (message) => message.role === "assistant" && message.loading && message.taskId,
  );
}

export function newConversation(): Conversation {
  const now = new Date().toISOString();
  return {
    conversationId: crypto.randomUUID(),
    title: "新对话",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function titleFromMessages(messages: ChatMessageData[]): string {
  const firstUser = messages.find((message) => message.role === "user");
  const text = firstUser?.parts
    .filter((part) => part.type === "text")
    .map((part) => part.content)
    .join(" ")
    .trim();
  if (!text) return "新对话";
  return text.length > 30 ? `${text.slice(0, 30)}…` : text;
}

export function conversationSearchText(conversation: Conversation): string {
  return [
    conversation.title,
    ...conversation.messages.flatMap((message) =>
      message.parts
        .filter((part) => part.type === "text")
        .map((part) => (typeof part.content === "string" ? part.content : "")),
    ),
  ]
    .join("\n")
    .toLocaleLowerCase();
}

export function latestConversationReferences(conversation?: Conversation): ChatReference[] {
  if (!conversation) return [];
  for (let index = conversation.messages.length - 1; index >= 0; index -= 1) {
    const references = (conversation.messages[index].references || []).filter(
      (reference) => reference.url,
    );
    if (references.length > 0) return references;
  }
  return [];
}

export function loadStore(): ChatStore {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as ChatStore;
      if (parsed.conversations?.length) {
        return {
          ...parsed,
          conversations: parsed.conversations.map(normalizeConversation),
        };
      }
    }

    const current = JSON.parse(localStorage.getItem(LEGACY_CURRENT_KEY) || "null");
    const history = JSON.parse(localStorage.getItem(LEGACY_HISTORY_KEY) || "[]");
    const migrated = [...(Array.isArray(history) ? history : [])];
    if (
      current?.messages?.length &&
      !migrated.some((item) => item.conversationId === current.conversationId)
    ) {
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
      return {
        activeId: conversations[0].conversationId,
        conversations: conversations.map(normalizeConversation),
      };
    }
  } catch {
    // Fall through to a clean local store.
  }
  const first = newConversation();
  return { activeId: first.conversationId, conversations: [first] };
}

export function saveStore(store: ChatStore): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch {
    // Chat remains usable when browser storage is unavailable.
  }
}

export function hydrateServerConversation(raw: Record<string, unknown>): Conversation {
  const rawMessages = Array.isArray(raw.messages) ? raw.messages : [];
  return {
    conversationId: String(raw.conversation_id || crypto.randomUUID()),
    title: String(raw.title || "新对话"),
    createdAt: String(raw.created_at || new Date().toISOString()),
    updatedAt: String(raw.updated_at || new Date().toISOString()),
    messages: rawMessages.map((rawMessage) => {
      const message = rawMessage as Record<string, unknown>;
      const status =
        typeof message.status === "string"
          ? (message.status as ChatMessageData["status"])
          : undefined;
      return {
        id: typeof message.id === "string" ? message.id : undefined,
        role: message.role === "assistant" ? "assistant" : "user",
        createdAt:
          typeof message.created_at === "string"
            ? message.created_at
            : new Date().toISOString(),
        references: Array.isArray(message.references)
          ? (message.references as ChatReference[])
          : [],
        parts: Array.isArray(message.parts)
          ? dedupeParts(message.parts as ChatMessageData["parts"])
          : [],
        loading: Boolean(message.loading),
        status,
        taskId: typeof message.task_id === "string" ? message.task_id : undefined,
      };
    }),
  };
}
