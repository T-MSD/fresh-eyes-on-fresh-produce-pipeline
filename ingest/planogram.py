import re
from collections import Counter
from dataclasses import dataclass

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ingest.errors import InvalidInput

AREA_NUMBER = re.compile(r"[1-9][0-9]*")

Point = tuple[int, int]


class Corners(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point


class RawArea(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    area_id: str
    product_id: str = Field(validation_alias=AliasChoices("product_id", "productd"))
    product_name: str
    image_coordinates: Corners


class RawCamera(BaseModel):
    products: list[RawArea]


class RawStore(BaseModel):
    cameras: dict[str, RawCamera]


RAW_PLANOGRAM = TypeAdapter(dict[str, RawStore])


@dataclass(frozen=True)
class PlanogramProduct:
    external_id: str
    name: str


@dataclass(frozen=True)
class PlanogramArea:
    external_id: str
    camera: str
    number: int
    product: str
    corners: Corners


@dataclass(frozen=True)
class Planogram:
    store: str
    cameras: tuple[str, ...]
    products: tuple[PlanogramProduct, ...]
    areas: tuple[PlanogramArea, ...]


def parse_planogram(raw: bytes) -> Planogram:
    """Parse and check a planogram file; raise InvalidInput listing every problem found."""
    try:
        stores = RAW_PLANOGRAM.validate_json(raw)
    except ValidationError as error:
        raise InvalidInput(
            [
                f"planogram {'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}"
                for e in error.errors()
            ]
        ) from error

    if len(stores) != 1:
        raise InvalidInput(
            [f"planogram must hold exactly one store, found {len(stores)}: {sorted(stores)}"]
        )
    ((store, content),) = stores.items()

    problems: list[str] = []
    areas: list[PlanogramArea] = []
    names: dict[str, set[str]] = {}
    for camera, entry in content.cameras.items():
        for raw_area in entry.products:
            names.setdefault(raw_area.product_id, set()).add(raw_area.product_name)
            number = area_number(raw_area.area_id, camera)
            if number is None:
                problems.append(f"area {raw_area.area_id!r} is not {camera}_<n> with n >= 1")
                continue
            areas.append(
                PlanogramArea(
                    external_id=raw_area.area_id,
                    camera=camera,
                    number=number,
                    product=raw_area.product_id,
                    corners=raw_area.image_coordinates,
                )
            )

    for area_id, times in Counter(a.external_id for a in areas).items():
        if times > 1:
            problems.append(f"area {area_id!r} appears {times} times")
    for product, product_names in names.items():
        if len(product_names) > 1:
            problems.append(f"product {product!r} has several names: {sorted(product_names)}")
    if not names:
        problems.append("planogram has no areas")
    if problems:
        raise InvalidInput(problems)

    return Planogram(
        store=store,
        cameras=tuple(content.cameras),
        products=tuple(PlanogramProduct(id_, name) for id_, (name,) in names.items()),
        areas=tuple(areas),
    )


def area_number(area_id: str, camera: str) -> int | None:
    prefix, separator, suffix = area_id.rpartition("_")
    if not separator or prefix != camera or not AREA_NUMBER.fullmatch(suffix):
        return None
    return int(suffix)
