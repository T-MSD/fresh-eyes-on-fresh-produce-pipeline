from datetime import datetime
from typing import Any, Literal, get_args

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

StockLevel = Literal["empty", "half", "full"]
RunStatus = Literal["running", "done", "failed"]

STOCK_VALUES: dict[StockLevel, float] = {"empty": 0.0, "half": 0.5, "full": 1.0}
CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


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


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String, unique=True)
    name: Mapped[str | None]

    areas: Mapped[list["Area"]] = relationship(back_populates="camera", order_by="Area.number")
    frames: Mapped[list["Frame"]] = relationship(back_populates="camera", order_by="Frame.ts")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String, unique=True)
    name: Mapped[str]


class Area(Base):
    __tablename__ = "areas"
    __table_args__ = (
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
    external_id: Mapped[str] = mapped_column(String, unique=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    number: Mapped[int]
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    top_left: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    top_right: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    bottom_right: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    bottom_left: Mapped[list[int]] = mapped_column(ARRAY(Integer))

    camera: Mapped[Camera] = relationship(back_populates="areas")
    product: Mapped[Product] = relationship()


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
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    width: Mapped[int | None]
    height: Mapped[int | None]

    camera: Mapped[Camera] = relationship(back_populates="frames")


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint("score <= 2 * labelled_areas", name="score_within_max"),
        CheckConstraint("labelled_areas >= 0", name="labelled_areas_non_negative"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    method: Mapped[str]
    git_sha: Mapped[str | None]
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    status: Mapped[RunStatus] = mapped_column(
        Enum(*get_args(RunStatus), name="run_status"), server_default="running"
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    score: Mapped[int | None]
    compliance_points: Mapped[int | None]
    stock_points: Mapped[int | None]
    labels_sha256: Mapped[str | None]
    labelled_areas: Mapped[int | None]
    dataset_loaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        ForeignKeyConstraint(["frame_id", "camera_id"], ["frames.id", "frames.camera_id"]),
        ForeignKeyConstraint(["area_id", "camera_id"], ["areas.id", "areas.camera_id"]),
        Index("ix_predictions_run_id_area_id", "run_id", "area_id"),
        CheckConstraint("compliance_confidence BETWEEN 0 AND 1", name="compliance_confidence_unit"),
        CheckConstraint("stock_confidence BETWEEN 0 AND 1", name="stock_confidence_unit"),
    )

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    frame_id: Mapped[int] = mapped_column(primary_key=True)
    area_id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int]
    planogram_compliant: Mapped[bool]
    stock_level: Mapped[StockLevel] = mapped_column(Enum(*get_args(StockLevel), name="stock_level"))
    detected_product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"))
    compliance_confidence: Mapped[float | None]
    stock_confidence: Mapped[float | None]
    is_fallback: Mapped[bool] = mapped_column(server_default=text("false"))

    run: Mapped[Run] = relationship(back_populates="predictions")
    frame: Mapped[Frame] = relationship()
    area: Mapped[Area] = relationship(overlaps="frame")
    detected_product: Mapped[Product | None] = relationship()

    @property
    def stock_value(self) -> float:
        return STOCK_VALUES[self.stock_level]


class ActiveDataset(Base):
    __tablename__ = "active_dataset"
    __table_args__ = (CheckConstraint("id = 1", name="single_row"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False, server_default="1")
    planogram_path: Mapped[str]
    images_dir: Mapped[str]
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
