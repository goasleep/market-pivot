"""Tortoise models for durable chat state.

The first four tables mirror the existing SQLite schema so existing local
conversation data remains readable. The search table is a derived index and
can be rebuilt from ``chat_messages`` at any time.
"""

from tortoise import fields
from tortoise.models import Model


class ChatConversation(Model):
    conversation_id = fields.CharField(max_length=255, primary_key=True)
    title = fields.TextField()
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64, db_index=True)

    class Meta:
        table = "chat_conversations"


class ChatMessage(Model):
    message_id = fields.CharField(max_length=255, primary_key=True)
    conversation_id = fields.CharField(max_length=255, db_index=True)
    role = fields.CharField(max_length=32)
    parts_json = fields.TextField()
    status = fields.CharField(max_length=32, default="completed")
    task_id = fields.CharField(max_length=255, null=True)
    position = fields.IntField()
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "chat_messages"
        indexes = (("conversation_id", "position"),)


class ChatTask(Model):
    task_id = fields.CharField(max_length=255, primary_key=True)
    conversation_id = fields.CharField(max_length=255, db_index=True)
    message_id = fields.CharField(max_length=255)
    status = fields.CharField(max_length=32)
    error = fields.TextField(null=True)
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "chat_tasks"
        indexes = (("conversation_id", "created_at"),)


class ChatMessageSearch(Model):
    """Derived plain-text search index for one chat message."""

    message_id = fields.CharField(max_length=255, primary_key=True)
    conversation_id = fields.CharField(max_length=255, db_index=True)
    content = fields.TextField()

    class Meta:
        table = "chat_message_search"


class ChatTaskEvent(Model):
    """Durable SSE events.

    The legacy schema uses ``(task_id, sequence)`` as its primary key. Tortoise
    supports one model primary key, so ``id`` is a stable surrogate while the
    legacy pair remains unique at the database level.
    """

    id = fields.CharField(max_length=511, primary_key=True)
    task_id = fields.CharField(max_length=255)
    sequence = fields.IntField()
    event = fields.CharField(max_length=64)
    data = fields.TextField()
    created_at = fields.CharField(max_length=64)

    class Meta:
        table = "chat_task_events"
        unique_together = (("task_id", "sequence"),)


class ChatTaskState(Model):
    """Durable request/checkpoint state used when a chat task is waiting."""

    task_id = fields.CharField(max_length=255, primary_key=True)
    state_json = fields.TextField()
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "chat_task_states"


class ChatTaskInteraction(Model):
    """A persisted user decision required before a task can continue."""

    interaction_id = fields.CharField(max_length=255, primary_key=True)
    task_id = fields.CharField(max_length=255, db_index=True)
    kind = fields.CharField(max_length=64)
    question = fields.TextField()
    options_json = fields.TextField()
    payload_json = fields.TextField()
    status = fields.CharField(max_length=32, default="pending")
    selected_option = fields.CharField(max_length=128, null=True)
    created_at = fields.CharField(max_length=64)
    responded_at = fields.CharField(max_length=64, null=True)

    class Meta:
        table = "chat_task_interactions"
        indexes = (("task_id", "status"),)


class ChatMessageReference(Model):
    """Message references using the legacy composite key."""

    id = fields.CharField(max_length=511, primary_key=True)
    message_id = fields.CharField(max_length=255)
    position = fields.IntField()
    reference_json = fields.TextField()
    created_at = fields.CharField(max_length=64)

    class Meta:
        table = "chat_message_references"
        unique_together = (("message_id", "position"),)


__models__ = [
    ChatConversation,
    ChatMessage,
    ChatTask,
    ChatTaskState,
    ChatTaskInteraction,
    ChatMessageSearch,
    ChatTaskEvent,
    ChatMessageReference,
]
