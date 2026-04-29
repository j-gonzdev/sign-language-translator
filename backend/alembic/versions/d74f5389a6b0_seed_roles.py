"""seed_roles

Revision ID: d74f5389a6b0
Revises: d450ccd6a497
Create Date: 2026-04-29 09:34:41.999139

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd74f5389a6b0'
down_revision: Union[str, None] = 'd450ccd6a497'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("INSERT INTO rol (nombre) VALUES ('user'), ('admin')")


def downgrade() -> None:
    op.execute("DELETE FROM rol WHERE nombre IN ('user', 'admin')")