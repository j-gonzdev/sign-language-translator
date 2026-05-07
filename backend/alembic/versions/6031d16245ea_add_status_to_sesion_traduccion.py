"""add_status_to_sesion_traduccion

Revision ID: 6031d16245ea
Revises: d74f5389a6b0
Create Date: 2026-05-07 16:28:47.080632

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '6031d16245ea'
down_revision: Union[str, None] = 'd74f5389a6b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE TYPE sesionestatus AS ENUM ('COMPLETADA', 'INTERRUMPIDA')")
    op.add_column(
        'sesion_traduccion',
        sa.Column(
            'status',
            sa.Enum('COMPLETADA', 'INTERRUMPIDA', name='sesionestatus', create_type=False),
            nullable=False,
            server_default='COMPLETADA'
        )
    )


def downgrade() -> None:
    op.drop_column('sesion_traduccion', 'status')
    op.execute("DROP TYPE sesionestatus")