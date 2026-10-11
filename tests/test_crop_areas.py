import json
from pathlib import Path

import pytest
import typer

from src import crop_areas
from tests.builders import area, encode, planogram_json


def build_dataset(root: Path) -> None:
    (root / "planogram.json").write_bytes(
        planogram_json(
            {
                "6215": [
                    area("6215_1"),
                    area("6215_2", "11909", product_key="productd"),
                ]
            }
        )
    )
    frames = root / "6215"
    frames.mkdir()
    (frames / "100.jpg").write_bytes(encode(640, 480))
    (frames / "200.jpg").write_bytes(b"not a jpeg")


def test_crops_every_area_of_a_readable_frame_including_the_productd_spelling(tmp_path):
    build_dataset(tmp_path)

    crop_areas.main(data_dir=tmp_path, out_dir=tmp_path / "crops")

    index = json.loads((tmp_path / "crops" / "index.json").read_text())
    assert {i["key"]: i["product_id"] for i in index} == {
        "6215/100.jpg/6215_1": "11908",
        "6215/100.jpg/6215_2": "11909",
    }
    assert all((tmp_path / "crops" / i["crop"]).exists() for i in index)


def test_lists_every_area_of_an_unreadable_frame_so_none_is_dropped(tmp_path):
    build_dataset(tmp_path)

    crop_areas.main(data_dir=tmp_path, out_dir=tmp_path / "crops")

    unreadable = json.loads((tmp_path / "crops" / "unreadable.json").read_text())
    assert {i["key"]: i["product_id"] for i in unreadable} == {
        "6215/200.jpg/6215_1": "11908",
        "6215/200.jpg/6215_2": "11909",
    }


def test_reads_the_single_store_whatever_its_name(tmp_path):
    build_dataset(tmp_path)
    planogram = json.loads((tmp_path / "planogram.json").read_text())
    (tmp_path / "planogram.json").write_text(json.dumps({"Padaria": planogram["Frutas e Legumes"]}))

    crop_areas.main(data_dir=tmp_path, out_dir=tmp_path / "crops")

    assert len(json.loads((tmp_path / "crops" / "index.json").read_text())) == 2


def test_refuses_a_planogram_with_more_than_one_store(tmp_path):
    build_dataset(tmp_path)
    planogram = json.loads((tmp_path / "planogram.json").read_text())
    planogram["Padaria"] = planogram["Frutas e Legumes"]
    (tmp_path / "planogram.json").write_text(json.dumps(planogram))

    with pytest.raises(typer.BadParameter, match="exactly one store"):
        crop_areas.main(data_dir=tmp_path, out_dir=tmp_path / "crops")
