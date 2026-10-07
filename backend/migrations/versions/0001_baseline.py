"""Baseline: the schema as QuantumSentinel 1.2 created it.

Before versioned migrations, the application built its tables with
``create_all`` and patched a few columns in place on SQLite only. This
revision is therefore written to *ensure* the baseline rather than assume an
empty database:

* a new database gets every table, index and constraint;
* a database created by an earlier release keeps its data and gains whatever
  it is missing: tables, columns (on PostgreSQL the old in-place patches never
  ran) and indexes.

The table definitions are frozen here on purpose. Later schema changes are
separate revisions and never edit this file.

Revision ID: 0001
Revises:
Create Date: 2026-10-06
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

baseline = sa.MetaData()

sa.Table('research_workers', baseline,
    sa.Column('id', sa.String(length=128), nullable=False),
    sa.Column('hostname', sa.String(length=255), nullable=False),
    sa.Column('pid', sa.Integer(), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('current_job_id', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
)
sa.Table('server_signing_keys', baseline,
    sa.Column('key_id', sa.String(), nullable=False),
    sa.Column('algorithm', sa.String(length=32), nullable=False),
    sa.Column('public_key', sa.Text(), nullable=False),
    sa.Column('fingerprint', sa.String(length=64), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('retired_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('key_id'),
    sa.UniqueConstraint('fingerprint'),
)
sa.Table('signals', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('asset', sa.String(), nullable=False),
    sa.Column('signal_type', sa.String(), nullable=False),
    sa.Column('confidence', sa.Numeric(), nullable=False),
    sa.Column('features', sa.JSON(), nullable=True),
    sa.Column('sba_iterations', sa.Integer(), nullable=True),
    sa.Column('engine_version', sa.String(), nullable=True),
    sa.Column('generated_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id'),
)
sa.Table('users', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('password_hash', sa.String(), nullable=False),
    sa.Column('tier', sa.String(), nullable=True),
    sa.Column('beginner_mode', sa.Boolean(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('watchlist', sa.JSON(), nullable=True),
    sa.Column('preferred_exchanges', sa.JSON(), nullable=True),
    sa.Column('user_timezone', sa.String(length=64), nullable=True),
    sa.Column('role', sa.String(length=32), server_default='user', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.Index('ix_users_email', 'email', unique=True),
)
sa.Table('api_keys', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('name', sa.String(length=80), nullable=False),
    sa.Column('key_prefix', sa.String(length=16), nullable=False),
    sa.Column('key_hash', sa.String(length=128), nullable=False),
    sa.Column('hmac_secret_encrypted', sa.Text(), nullable=True),
    sa.Column('scopes', sa.JSON(), nullable=True),
    sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('is_revoked', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.Index('ix_api_keys_key_hash', 'key_hash', unique=True),
    sa.Index('ix_api_keys_user_id', 'user_id'),
)
sa.Table('audit_logs', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=True),
    sa.Column('action', sa.String(), nullable=False),
    sa.Column('resource_type', sa.String(), nullable=True),
    sa.Column('resource_id', sa.String(), nullable=True),
    sa.Column('metadata_json', sa.JSON(), nullable=True),
    sa.Column('pqc_signature', sa.Text(), nullable=True),
    sa.Column('signing_key_id', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
)
sa.Table('idempotency_records', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('idempotency_key', sa.String(length=128), nullable=False),
    sa.Column('request_hash', sa.String(length=64), nullable=False),
    sa.Column('response_json', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'idempotency_key', name='uq_idempotency_user_key'),
    sa.Index('ix_idempotency_records_user_id', 'user_id'),
)
sa.Table('key_pairs', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('algorithm', sa.String(length=50), nullable=False),
    sa.Column('public_key', sa.Text(), nullable=False),
    sa.Column('private_key', sa.Text(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('rotation_count', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
)
sa.Table('paper_accounts', baseline,
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('cash_micros', sa.BigInteger(), nullable=False),
    sa.Column('reserved_micros', sa.BigInteger(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('user_id'),
)
sa.Table('positions', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('asset', sa.String(), nullable=False),
    sa.Column('quantity', sa.Numeric(), nullable=True),
    sa.Column('avg_entry_price', sa.Numeric(), nullable=True),
    sa.Column('realized_pnl', sa.Numeric(), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
)
sa.Table('refresh_tokens', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('family_id', sa.String(), nullable=False),
    sa.Column('is_used', sa.Boolean(), nullable=True),
    sa.Column('is_revoked', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('token_hash', name='uq_refresh_token_hash'),
    sa.Index('ix_refresh_tokens_family_id', 'family_id'),
    sa.Index('ix_refresh_tokens_token_hash', 'token_hash'),
    sa.Index('ix_refresh_tokens_user_id', 'user_id'),
)
sa.Table('research_experiments', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('strategy_id', sa.String(), nullable=False),
    sa.Column('strategy_version', sa.String(), nullable=False),
    sa.Column('dataset_id', sa.String(), nullable=False),
    sa.Column('dataset_json', sa.JSON(), nullable=True),
    sa.Column('dataset_hash', sa.String(length=64), nullable=False),
    sa.Column('parameters_json', sa.JSON(), nullable=False),
    sa.Column('parameter_hash', sa.String(length=64), nullable=False),
    sa.Column('code_commit', sa.String(), nullable=False),
    sa.Column('random_seed', sa.Integer(), nullable=False),
    sa.Column('execution_model', sa.String(), nullable=False),
    sa.Column('latency_model', sa.String(), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('result_hash', sa.String(length=64), nullable=False),
    sa.Column('results_json', sa.JSON(), nullable=False),
    sa.Column('validation_gates_json', sa.JSON(), nullable=False),
    sa.Column('manifest_signature', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('strategy_hash', sa.String(length=64), nullable=True),
    sa.Column('dependency_lock_hash', sa.String(length=64), nullable=True),
    sa.Column('engine_version', sa.String(length=32), nullable=True),
    sa.Column('signing_key_id', sa.String(), nullable=True),
    sa.Column('manifest_json', sa.JSON(), nullable=True),
    sa.Column('approved_by', sa.String(), nullable=True),
    sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['approved_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.Index('ix_research_experiments_user_id', 'user_id'),
)
sa.Table('research_jobs', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('kind', sa.String(length=64), nullable=False),
    sa.Column('params_json', sa.JSON(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('cancel_requested', sa.Boolean(), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('max_attempts', sa.Integer(), nullable=False),
    sa.Column('worker_id', sa.String(length=128), nullable=True),
    sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('result_json', sa.JSON(), nullable=True),
    sa.Column('error_status', sa.Integer(), nullable=True),
    sa.Column('error_detail', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.Index('ix_research_jobs_status_created', 'status', 'created_at'),
    sa.Index('ix_research_jobs_user_status', 'user_id', 'status'),
)
sa.Table('research_trials', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('family_hash', sa.String(length=64), nullable=False),
    sa.Column('config_hash', sa.String(length=64), nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'family_hash', 'config_hash', name='uq_research_trial_config'),
    sa.Index('ix_research_trials_family_hash', 'family_hash'),
    sa.Index('ix_research_trials_user_id', 'user_id'),
)
sa.Table('strategies', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('assets', sa.JSON(), nullable=True),
    sa.Column('config', sa.JSON(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
)
sa.Table('trades', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('asset', sa.String(), nullable=False),
    sa.Column('side', sa.String(), nullable=False),
    sa.Column('quantity', sa.Numeric(), nullable=False),
    sa.Column('order_type', sa.String(), nullable=True),
    sa.Column('limit_price', sa.Numeric(), nullable=True),
    sa.Column('stop_price', sa.Numeric(), nullable=True),
    sa.Column('time_in_force', sa.String(), nullable=True),
    sa.Column('status', sa.String(), nullable=True),
    sa.Column('alpaca_order_id', sa.String(), nullable=True),
    sa.Column('filled_price', sa.Numeric(), nullable=True),
    sa.Column('pqc_signature', sa.Text(), nullable=True),
    sa.Column('submitted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('filled_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
)
sa.Table('webhooks', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('url', sa.String(length=2048), nullable=False),
    sa.Column('secret_hash', sa.String(length=128), nullable=False),
    sa.Column('event_types', sa.JSON(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('last_delivery_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.Index('ix_webhooks_user_id', 'user_id'),
)
sa.Table('audit_chain_links', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('sequence', sa.Integer(), nullable=False),
    sa.Column('audit_log_id', sa.String(), nullable=False),
    sa.Column('previous_hash', sa.String(length=64), nullable=False),
    sa.Column('entry_hash', sa.String(length=64), nullable=False),
    sa.Column('checkpoint_signature', sa.Text(), nullable=False),
    sa.Column('signing_key_id', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['audit_log_id'], ['audit_logs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('entry_hash'),
    sa.UniqueConstraint('sequence'),
    sa.Index('ix_audit_chain_links_audit_log_id', 'audit_log_id', unique=True),
)
sa.Table('backtests', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('strategy_id', sa.String(), nullable=True),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('start_date', sa.DateTime(timezone=True), nullable=True),
    sa.Column('end_date', sa.DateTime(timezone=True), nullable=True),
    sa.Column('initial_capital', sa.Numeric(), nullable=True),
    sa.Column('final_capital', sa.Numeric(), nullable=False),
    sa.Column('sharpe_ratio', sa.Numeric(), nullable=True),
    sa.Column('max_drawdown', sa.Numeric(), nullable=True),
    sa.Column('win_rate', sa.Numeric(), nullable=True),
    sa.Column('total_trades', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(), nullable=True),
    sa.Column('result_json', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['strategy_id'], ['strategies.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.Index('ix_backtests_user_id', 'user_id'),
)
sa.Table('order_security_records', baseline,
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('trade_id', sa.String(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=False),
    sa.Column('canonical_order', sa.Text(), nullable=False),
    sa.Column('request_hash', sa.String(length=64), nullable=False),
    sa.Column('nonce', sa.String(length=128), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('signer_key_id', sa.String(), nullable=True),
    sa.Column('signature', sa.Text(), nullable=False),
    sa.Column('signature_mode', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['trade_id'], ['trades.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'nonce', name='uq_order_security_user_nonce'),
    sa.Index('ix_order_security_records_trade_id', 'trade_id', unique=True),
    sa.Index('ix_order_security_records_user_id', 'user_id'),
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    baseline.create_all(bind, tables=[t for t in baseline.sorted_tables if t.name not in existing])
    for table in baseline.sorted_tables:
        if table.name not in existing:
            continue
        columns = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name not in columns:
                op.add_column(table.name, sa.Column(column.name, column.type, nullable=column.nullable,
                                                    server_default=column.server_default))
        indexes = {i["name"] for i in inspector.get_indexes(table.name)}
        for index in table.indexes:
            if index.name not in indexes:
                index.create(bind)


def downgrade() -> None:
    raise NotImplementedError("the baseline cannot be downgraded: it would drop every table")
