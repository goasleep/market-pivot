"""Tortoise ORM models for application persistence.

The project deliberately uses one ORM for chat, simulation, automation,
artifacts, settings, and experiment metadata.  Payload-heavy records keep
their nested values as JSON text because they are application snapshots rather
than queryable relational aggregates.
"""

from tortoise import fields
from tortoise.models import Model


class CacheEntry(Model):
    key = fields.CharField(max_length=512, primary_key=True)
    value = fields.TextField()
    timestamp = fields.FloatField()

    class Meta:
        table = "cache"


class SimulationAccountRecord(Model):
    account_id = fields.CharField(max_length=64, primary_key=True)
    status = fields.CharField(max_length=32)
    current_date = fields.CharField(max_length=32, default="")
    config_json = fields.TextField()
    portfolio_json = fields.TextField()
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "simulation_accounts"


class SimulationOrderRecord(Model):
    order_id = fields.CharField(max_length=128, primary_key=True)
    account_id = fields.CharField(max_length=64, db_index=True)
    status = fields.CharField(max_length=32)
    order_json = fields.TextField()
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "simulation_orders"


class SimulationSnapshotRecord(Model):
    id = fields.CharField(max_length=160, primary_key=True)
    account_id = fields.CharField(max_length=64, db_index=True)
    snapshot_date = fields.CharField(max_length=32)
    snapshot_json = fields.TextField()
    created_at = fields.CharField(max_length=64)

    class Meta:
        table = "simulation_snapshots"
        unique_together = (("account_id", "snapshot_date"),)


class AutomationTaskRecord(Model):
    account_id = fields.CharField(max_length=64, primary_key=True)
    config_json = fields.TextField()
    status = fields.CharField(max_length=32, default="idle")
    last_run_id = fields.CharField(max_length=128, null=True)
    last_run_date = fields.CharField(max_length=32, null=True)
    last_error = fields.TextField(null=True)
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "automation_tasks"


class AgentRunRecord(Model):
    run_id = fields.CharField(max_length=128, primary_key=True)
    account_id = fields.CharField(max_length=64, db_index=True)
    run_date = fields.CharField(max_length=32)
    trigger = fields.CharField(max_length=32)
    status = fields.CharField(max_length=32)
    summary_json = fields.TextField()
    idempotency_key = fields.CharField(max_length=512, unique=True)
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "agent_runs"


class AgentDecisionRecord(Model):
    decision_id = fields.CharField(max_length=128, primary_key=True)
    run_id = fields.CharField(max_length=128, db_index=True)
    account_id = fields.CharField(max_length=64, db_index=True)
    ticker = fields.CharField(max_length=32)
    decision_json = fields.TextField()
    created_at = fields.CharField(max_length=64)

    class Meta:
        table = "agent_decisions"


class AutomationEventRecord(Model):
    event_id = fields.CharField(max_length=128, primary_key=True)
    account_id = fields.CharField(max_length=64, db_index=True)
    run_id = fields.CharField(max_length=128, null=True)
    event_type = fields.CharField(max_length=64)
    payload_json = fields.TextField()
    created_at = fields.CharField(max_length=64)

    class Meta:
        table = "automation_events"


class BacktestExperimentRecord(Model):
    experiment_id = fields.CharField(max_length=128, primary_key=True)
    status = fields.CharField(max_length=32)
    payload_json = fields.TextField()
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "backtest_experiments"


class StrategyDeploymentRecord(Model):
    deployment_id = fields.CharField(max_length=128, primary_key=True)
    deployment_key = fields.CharField(max_length=512, unique=True)
    experiment_id = fields.CharField(max_length=128, db_index=True)
    account_id = fields.CharField(max_length=64, db_index=True)
    status = fields.CharField(max_length=32)
    strategy_name = fields.CharField(max_length=160)
    strategy_version = fields.CharField(max_length=64)
    strategy_sha256 = fields.CharField(max_length=64)
    strategy_spec_json = fields.TextField()
    portfolio_spec_json = fields.TextField(null=True)
    universe_json = fields.TextField()
    asset_type = fields.CharField(max_length=32)
    execution_json = fields.TextField()
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64)
    activated_at = fields.CharField(max_length=64, null=True)
    archived_at = fields.CharField(max_length=64, null=True)

    class Meta:
        table = "strategy_deployments"


class StrategyRuntimeStateRecord(Model):
    id = fields.CharField(max_length=256, primary_key=True)
    deployment_id = fields.CharField(max_length=128, db_index=True)
    ticker = fields.CharField(max_length=32)
    state_json = fields.TextField()
    last_evaluated_date = fields.CharField(max_length=32, null=True)
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "strategy_runtime_states"
        unique_together = (("deployment_id", "ticker"),)


class ResearchStrategyCandidateRecord(Model):
    candidate_id = fields.CharField(max_length=128, primary_key=True)
    status = fields.CharField(max_length=32, db_index=True)
    payload_json = fields.TextField()
    created_at = fields.CharField(max_length=64)
    updated_at = fields.CharField(max_length=64)

    class Meta:
        table = "research_strategy_candidates"


class ArtifactRecord(Model):
    artifact_id = fields.CharField(max_length=128, primary_key=True)
    name = fields.CharField(max_length=512)
    artifact_type = fields.CharField(max_length=64)
    mime_type = fields.CharField(max_length=255)
    relative_path = fields.CharField(max_length=1024, unique=True)
    size_bytes = fields.IntField()
    sha256 = fields.CharField(max_length=64)
    ticker = fields.CharField(max_length=32, null=True)
    asset_type = fields.CharField(max_length=32, null=True)
    source = fields.CharField(max_length=64)
    conversation_id = fields.CharField(max_length=255, null=True)
    task_id = fields.CharField(max_length=255, null=True)
    metadata_json = fields.TextField()
    created_at = fields.CharField(max_length=64, db_index=True)

    class Meta:
        table = "artifacts"


__models__ = [
    CacheEntry,
    SimulationAccountRecord,
    SimulationOrderRecord,
    SimulationSnapshotRecord,
    AutomationTaskRecord,
    AgentRunRecord,
    AgentDecisionRecord,
    AutomationEventRecord,
    BacktestExperimentRecord,
    StrategyDeploymentRecord,
    StrategyRuntimeStateRecord,
    ResearchStrategyCandidateRecord,
    ArtifactRecord,
]
