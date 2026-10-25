import json

import pytest

from src.dataset.errors import InvalidInput
from src.dataset.planogram import Corners, PlanogramArea, PlanogramProduct, parse_planogram
from tests.builders import area, planogram_json


def problems_of(raw: bytes) -> str:
    with pytest.raises(InvalidInput) as error:
        parse_planogram(raw)
    return "\n".join(error.value.problems)


def test_parses_cameras_products_and_areas():
    planogram = parse_planogram(
        planogram_json(
            {
                "6215": [area("6215_1", "20923", "BANANA DA MADEIRA CNT KG"), area("6215_2")],
                "6383": [area("6383_1")],
            }
        )
    )

    assert planogram.store == "Frutas e Legumes"
    assert planogram.cameras == ("6215", "6383")
    assert planogram.products == (
        PlanogramProduct("20923", "BANANA DA MADEIRA CNT KG"),
        PlanogramProduct("11908", "BANANA CAT I CNT KG"),
    )
    assert planogram.areas[1] == PlanogramArea(
        external_id="6215_2",
        camera="6215",
        number=2,
        product="11908",
        corners=Corners(
            top_left=(299, 350),
            top_right=(464, 347),
            bottom_right=(479, 438),
            bottom_left=(331, 438),
        ),
    )


def test_same_product_in_several_areas_is_one_product():
    planogram = parse_planogram(planogram_json({"6215": [area("6215_1"), area("6215_2")]}))
    assert len(planogram.products) == 1
    assert [a.product for a in planogram.areas] == ["11908", "11908"]


def test_accepts_productd_typo_for_product_id():
    planogram = parse_planogram(planogram_json({"6383": [area("6383_2", product_key="productd")]}))
    assert planogram.areas[0].product == "11908"


def test_accepts_numeric_product_id():
    raw = json.loads(planogram_json({"6215": [area("6215_1")]}))
    raw["Frutas e Legumes"]["cameras"]["6215"]["products"][0]["product_id"] = 11908
    assert parse_planogram(json.dumps(raw).encode()).areas[0].product == "11908"


def test_keeps_negative_corners_of_areas_past_the_image_edge():
    planogram = parse_planogram(planogram_json({"6215": [area("6215_1", top_left=[-12, 350])]}))
    assert planogram.areas[0].corners.top_left == (-12, 350)


def test_rejects_invalid_json():
    assert "planogram" in problems_of(b"{not json")


def test_rejects_area_without_product_id():
    raw = json.loads(planogram_json({"6215": [area("6215_1")]}))
    del raw["Frutas e Legumes"]["cameras"]["6215"]["products"][0]["product_id"]
    assert "product_id" in problems_of(json.dumps(raw).encode())


@pytest.mark.parametrize("corner", [[1, 2, 3], [1], [1.5, 2]])
def test_rejects_corner_that_is_not_two_integers(corner):
    assert "top_left" in problems_of(planogram_json({"6215": [area("6215_1", top_left=corner)]}))


def test_rejects_more_than_one_store():
    raw = json.loads(planogram_json({"6215": [area("6215_1")]}))
    raw["Talho"] = raw["Frutas e Legumes"]
    assert "exactly one store" in problems_of(json.dumps(raw).encode())


def test_rejects_planogram_without_areas():
    assert "no areas" in problems_of(planogram_json({"6215": []}))


@pytest.mark.parametrize("area_id", ["6383_1", "6215-1", "6215_x", "6215_0", "6215_", "6215_١"])
def test_rejects_area_id_that_does_not_match_its_camera(area_id):
    assert area_id in problems_of(planogram_json({"6215": [area(area_id)]}))


def test_rejects_duplicate_area_id():
    assert "6215_1" in problems_of(planogram_json({"6215": [area("6215_1"), area("6215_1")]}))


def test_rejects_product_id_with_two_names():
    raw = planogram_json(
        {"6215": [area("6215_1", "11908", "BANANA"), area("6215_2", "11908", "KIWI")]}
    )
    problems = problems_of(raw)
    assert "11908" in problems
    assert "BANANA" in problems
    assert "KIWI" in problems


def test_reports_every_problem_at_once():
    problems = problems_of(planogram_json({"6215": [area("6215_0"), area("6383_1")]}))
    assert "6215_0" in problems
    assert "6383_1" in problems
