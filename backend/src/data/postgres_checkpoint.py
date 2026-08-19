"""PostgreSQL checkpoint compatibility name backed by the shared ORM saver."""

from data.sqlite_checkpoint import OrmCheckpointSaver


class PostgresCheckpointSaver(OrmCheckpointSaver):
    """Use the same SQLAlchemy checkpoint models with PostgreSQL."""
