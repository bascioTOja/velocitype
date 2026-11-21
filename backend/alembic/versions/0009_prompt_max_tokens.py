"""Per-user max response tokens for the AI-coach prompts

Revision ID: 0009_prompt_max_tokens
Revises: 0008_openai_compat
Create Date: 2026-11-21
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009_prompt_max_tokens"
down_revision: Union[str, None] = "0008_openai_compat"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NULL = use the built-in default (like the prompt text columns).
    op.add_column("user_prompts", sa.Column("analysis_max_tokens", sa.Integer(), nullable=True))
    op.add_column("user_prompts", sa.Column("drill_max_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("user_prompts", "drill_max_tokens")
    op.drop_column("user_prompts", "analysis_max_tokens")
