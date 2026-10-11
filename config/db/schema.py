from datetime import UTC, datetime
from typing import Any, Literal, get_args

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Dialect,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    TypeDecorator,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

StockLevel = Literal["empty", "half", "full"]
RunStatus = Literal["running", "done", "failed"]
TaskKind = Literal["restock", "move_product", "top_up"]

STOCK_VALUES: dict[StockLevel, float] = {"empty": 0.0, "half": 0.5, "full": 1.0}
CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


class UTCDateTime(TypeDecorator[datetime]):
    """A `timestamptz` that reads back in UTC, whatever the session time zone."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    planogram_path: Mapped[str]
    images_dir: Mapped[str]
    loaded_at: Mapped[datetime] = mapped_column(UTCDateTime, server_default=func.now())


class Camera(Base):
    __tablename__ = "cameras"
    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id"),
        UniqueConstraint("id", "dataset_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    external_id: Mapped[str]
    name: Mapped[str | None]

    areas: Mapped[list["Area"]] = relationship(back_populates="camera", order_by="Area.number")
    frames: Mapped[list["Frame"]] = relationship(back_populates="camera", order_by="Frame.ts")


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id"),
        UniqueConstraint("id", "dataset_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    external_id: Mapped[str]
    name: Mapped[str]


class Area(Base):
    __tablename__ = "areas"
    __table_args__ = (
        ForeignKeyConstraint(["camera_id", "dataset_id"], ["cameras.id", "cameras.dataset_id"]),
        ForeignKeyConstraint(["product_id", "dataset_id"], ["products.id", "products.dataset_id"]),
        UniqueConstraint("dataset_id", "external_id"),
        UniqueConstraint("camera_id", "number"),
        UniqueConstraint("id", "camera_id"),
        CheckConstraint("number > 0", name="number_positive"),
        *(
            CheckConstraint(
                f"array_ndims({c}) = 1 AND cardinality({c}) = 2"
                f" AND array_position({c}, NULL) IS NULL",
                name=f"{c}_is_xy",
            )
            for c in CORNERS
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int]
    external_id: Mapped[str]
    camera_id: Mapped[int]
    number: Mapped[int]
    product_id: Mapped[int]
    top_left: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    top_right: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    bottom_right: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    bottom_left: Mapped[list[int]] = mapped_column(ARRAY(Integer))

    camera: Mapped[Camera] = relationship(back_populates="areas")
    product: Mapped[Product] = relationship(overlaps="areas,camera")


class Frame(Base):
    __tablename__ = "frames"
    __table_args__ = (
        UniqueConstraint("camera_id", "filename"),
        UniqueConstraint("id", "camera_id"),
        Index("ix_frames_camera_id_ts", "camera_id", "ts"),
        CheckConstraint("(width IS NULL) = (height IS NULL)", name="size_both_or_neither"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    filename: Mapped[str]
    ts: Mapped[datetime] = mapped_column(UTCDateTime)
    width: Mapped[int | None]
    height: Mapped[int | None]

    camera: Mapped[Camera] = relationship(back_populates="frames")


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint("id", "dataset_id"),
        CheckConstraint("score <= 2 * labelled_areas", name="score_within_max"),
        CheckConstraint("labelled_areas >= 0", name="labelled_areas_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    method: Mapped[str]
    git_sha: Mapped[str | None]
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    status: Mapped[RunStatus] = mapped_column(
        Enum(*get_args(RunStatus), name="run_status"), server_default="running"
    )
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    score: Mapped[int | None]
    compliance_points: Mapped[int | None]
    stock_points: Mapped[int | None]
    labels_sha256: Mapped[str | None]
    labelled_areas: Mapped[int | None]

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "dataset_id"], ["runs.id", "runs.dataset_id"], ondelete="CASCADE"
        ),
        ForeignKeyConstraint(["camera_id", "dataset_id"], ["cameras.id", "cameras.dataset_id"]),
        ForeignKeyConstraint(["frame_id", "camera_id"], ["frames.id", "frames.camera_id"]),
        ForeignKeyConstraint(["area_id", "camera_id"], ["areas.id", "areas.camera_id"]),
        ForeignKeyConstraint(
            ["detected_product_id", "dataset_id"], ["products.id", "products.dataset_id"]
        ),
        UniqueConstraint("run_id", "frame_id", "area_id"),
        Index("ix_predictions_run_id_area_id", "run_id", "area_id"),
        CheckConstraint("compliance_confidence BETWEEN 0 AND 1", name="compliance_confidence_unit"),
        CheckConstraint("stock_confidence BETWEEN 0 AND 1", name="stock_confidence_unit"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int]
    frame_id: Mapped[int]
    area_id: Mapped[int]
    dataset_id: Mapped[int]
    camera_id: Mapped[int]
    planogram_compliant: Mapped[bool]
    stock_level: Mapped[StockLevel] = mapped_column(Enum(*get_args(StockLevel), name="stock_level"))
    detected_product_id: Mapped[int | None]
    compliance_confidence: Mapped[float | None]
    stock_confidence: Mapped[float | None]
    is_fallback: Mapped[bool] = mapped_column(server_default=text("false"))

    run: Mapped[Run] = relationship(back_populates="predictions")
    frame: Mapped[Frame] = relationship()
    area: Mapped[Area] = relationship(overlaps="frame")
    detected_product: Mapped[Product | None] = relationship(overlaps="predictions,run")

    @property
    def stock_value(self) -> float:
        return STOCK_VALUES[self.stock_level]


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        *(
            ForeignKeyConstraint(
                [frame, "run_id", "area_id"],
                ["predictions.frame_id", "predictions.run_id", "predictions.area_id"],
                ondelete="CASCADE",
            )
            for frame in ("opened_frame_id", "closed_frame_id")
        ),
        UniqueConstraint("run_id", "area_id", "kind", "opened_frame_id"),
        Index(
            "uq_tasks_run_id_area_id_open",
            "run_id",
            "area_id",
            unique=True,
            postgresql_where=text("closed_frame_id IS NULL"),
        ),
        CheckConstraint("closed_frame_id <> opened_frame_id", name="closed_on_another_frame"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int]
    area_id: Mapped[int]
    kind: Mapped[TaskKind] = mapped_column(Enum(*get_args(TaskKind), name="task_kind"))
    opened_frame_id: Mapped[int]
    closed_frame_id: Mapped[int | None]
    done_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
