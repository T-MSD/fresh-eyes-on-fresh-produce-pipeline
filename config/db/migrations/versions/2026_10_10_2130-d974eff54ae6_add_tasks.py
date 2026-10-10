"""add tasks

Revision ID: d974eff54ae6
Revises: df4f6c6fce8f
Create Date: 2026-10-10 21:30:46.896194

A task is one problem on one area (restock, move_product or top_up), opened on the frame where
it appears and closed on the first later frame where it is gone. done_at is the time staff marked
it done. Composite foreign keys keep the area and both frames on the task's camera.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d974eff54ae6"
down_revision: str | Sequence[str] | None = "df4f6c6fce8f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tasks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("area_id", sa.Integer(), nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column(
            "kind", sa.Enum("restock", "move_product", "top_up", name="task_kind"), nullable=False
        ),
        sa.Column("opened_frame_id", sa.Integer(), nullable=False),
        sa.Column("closed_frame_id", sa.Integer(), nullable=True),
        sa.Column("done_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "closed_frame_id <> opened_frame_id", name=op.f("ck_tasks_closed_on_another_frame")
        ),
        sa.ForeignKeyConstraint(
            ["area_id", "camera_id"],
            ["areas.id", "areas.camera_id"],
            name=op.f("fk_tasks_area_id_areas"),
        ),
        sa.ForeignKeyConstraint(
            ["closed_frame_id", "camera_id"],
            ["frames.id", "frames.camera_id"],
            name=op.f("fk_tasks_closed_frame_id_frames"),
        ),
        sa.ForeignKeyConstraint(
            ["opened_frame_id", "camera_id"],
            ["frames.id", "frames.camera_id"],
            name=op.f("fk_tasks_opened_frame_id_frames"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tasks")),
        sa.UniqueConstraint(
            "area_id", "kind", "opened_frame_id", name=op.f("uq_tasks_area_id_kind_opened_frame_id")
        ),
    )


def downgrade() -> None:
    op.drop_table("tasks")
    sa.Enum(name="task_kind").drop(op.get_bind())
