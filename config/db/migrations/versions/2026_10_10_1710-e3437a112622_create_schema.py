"""create schema

Revision ID: e3437a112622
Revises:
Create Date: 2026-10-10 17:10:38.694108

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e3437a112622"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cameras",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cameras")),
    )
    op.create_table(
        "products",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_products")),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("git_sha", sa.String(), nullable=True),
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("running", "done", "failed", name="run_status"),
            server_default="running",
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("compliance_points", sa.Integer(), nullable=True),
        sa.Column("stock_points", sa.Integer(), nullable=True),
        sa.Column("labels_sha256", sa.String(), nullable=True),
        sa.Column("labelled_areas", sa.Integer(), nullable=True),
        sa.CheckConstraint("labelled_areas >= 0", name=op.f("ck_runs_labelled_areas_non_negative")),
        sa.CheckConstraint("score <= 2 * labelled_areas", name=op.f("ck_runs_score_within_max")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
    )
    op.create_table(
        "areas",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("product_id", sa.String(), nullable=False),
        sa.Column("top_left", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("top_right", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("bottom_right", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("bottom_left", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.CheckConstraint(
            "array_ndims(bottom_left) = 1 AND cardinality(bottom_left) = 2 AND array_position(bottom_left, NULL) IS NULL",
            name=op.f("ck_areas_bottom_left_is_xy"),
        ),
        sa.CheckConstraint(
            "array_ndims(bottom_right) = 1 AND cardinality(bottom_right) = 2 AND array_position(bottom_right, NULL) IS NULL",
            name=op.f("ck_areas_bottom_right_is_xy"),
        ),
        sa.CheckConstraint(
            "array_ndims(top_left) = 1 AND cardinality(top_left) = 2 AND array_position(top_left, NULL) IS NULL",
            name=op.f("ck_areas_top_left_is_xy"),
        ),
        sa.CheckConstraint(
            "array_ndims(top_right) = 1 AND cardinality(top_right) = 2 AND array_position(top_right, NULL) IS NULL",
            name=op.f("ck_areas_top_right_is_xy"),
        ),
        sa.CheckConstraint("number > 0", name=op.f("ck_areas_number_positive")),
        sa.ForeignKeyConstraint(
            ["camera_id"], ["cameras.id"], name=op.f("fk_areas_camera_id_cameras")
        ),
        sa.ForeignKeyConstraint(
            ["product_id"], ["products.id"], name=op.f("fk_areas_product_id_products")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_areas")),
        sa.UniqueConstraint("camera_id", "number", name=op.f("uq_areas_camera_id_number")),
    )
    op.create_table(
        "frames",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "(width IS NULL) = (height IS NULL)", name=op.f("ck_frames_size_both_or_neither")
        ),
        sa.ForeignKeyConstraint(
            ["camera_id"], ["cameras.id"], name=op.f("fk_frames_camera_id_cameras")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_frames")),
        sa.UniqueConstraint("camera_id", "filename", name=op.f("uq_frames_camera_id_filename")),
    )
    op.create_index("ix_frames_camera_id_ts", "frames", ["camera_id", "ts"], unique=False)
    op.create_table(
        "predictions",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("frame_id", sa.Integer(), nullable=False),
        sa.Column("area_id", sa.String(), nullable=False),
        sa.Column("planogram_compliant", sa.Boolean(), nullable=False),
        sa.Column(
            "stock_level", sa.Enum("empty", "half", "full", name="stock_level"), nullable=False
        ),
        sa.Column("detected_product_id", sa.String(), nullable=True),
        sa.Column("compliance_confidence", sa.Double(), nullable=False),
        sa.Column("stock_confidence", sa.Double(), nullable=False),
        sa.Column("is_fallback", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.CheckConstraint(
            "compliance_confidence BETWEEN 0 AND 1",
            name=op.f("ck_predictions_compliance_confidence_unit"),
        ),
        sa.CheckConstraint(
            "stock_confidence BETWEEN 0 AND 1", name=op.f("ck_predictions_stock_confidence_unit")
        ),
        sa.ForeignKeyConstraint(
            ["area_id"], ["areas.id"], name=op.f("fk_predictions_area_id_areas")
        ),
        sa.ForeignKeyConstraint(
            ["detected_product_id"],
            ["products.id"],
            name=op.f("fk_predictions_detected_product_id_products"),
        ),
        sa.ForeignKeyConstraint(
            ["frame_id"], ["frames.id"], name=op.f("fk_predictions_frame_id_frames")
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], name=op.f("fk_predictions_run_id_runs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "frame_id", "area_id", name=op.f("pk_predictions")),
    )
    op.create_index(
        "ix_predictions_run_id_area_id", "predictions", ["run_id", "area_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_run_id_area_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_frames_camera_id_ts", table_name="frames")
    op.drop_table("frames")
    op.drop_table("areas")
    op.drop_table("runs")
    op.drop_table("products")
    op.drop_table("cameras")
    sa.Enum(name="stock_level").drop(op.get_bind())
    sa.Enum(name="run_status").drop(op.get_bind())
