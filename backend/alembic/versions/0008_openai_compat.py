"""OpenAI-compatible AI provider columns on user_ai_config

Revision ID: 0008_openai_compat
Revises: 0007_progressive_unlock
Create Date: 2026-11-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0008_openai_compat"
down_revision: Union[str, None] = "0007_progressive_unlock"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("user_ai_config", sa.Column("openai_base_url", sa.String(length=512), nullable=True))
    op.add_column("user_ai_config", sa.Column("openai_model", sa.String(length=256), nullable=True))
    op.add_column("user_ai_config", sa.Column("openai_api_key_enc", sa.Text(), nullable=True))


def downgrade() -> None:
    # Users on the removed provider fall back to the local default.
    op.execute("UPDATE user_ai_config SET provider = 'ollama' WHERE provider = 'openai_compat'")
    op.drop_column("user_ai_config", "openai_api_key_enc")
    op.drop_column("user_ai_config", "openai_model")
    op.drop_column("user_ai_config", "openai_base_url")
