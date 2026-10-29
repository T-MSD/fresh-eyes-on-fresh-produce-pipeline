import json

import pytest
import typer

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("sklearn")

BANANA = {"product_name": "BANANA CAT I CNT KG"}
PEAR = {"product_name": "PERA ROCHA CNT KG"}
LEMON = {"product_name": "LIMAO CNT KG"}


def test_accepts_the_classes_model_pt_was_trained_with(tmp_path):
    from src.finetune import check_head_classes

    (tmp_path / "classes.json").write_text(json.dumps(["bananas", "pears"]))

    check_head_classes(tmp_path, [BANANA, PEAR, BANANA])


def test_stops_when_the_classes_changed_even_if_their_number_did_not(tmp_path):
    from src.finetune import check_head_classes

    (tmp_path / "classes.json").write_text(json.dumps(["bananas", "pears"]))

    with pytest.raises(typer.BadParameter, match="wrong products"):
        check_head_classes(tmp_path, [BANANA, LEMON])


def test_stops_when_model_pt_has_no_saved_class_order(tmp_path):
    from src.finetune import check_head_classes

    with pytest.raises(typer.BadParameter, match="class order of model.pt is unknown"):
        check_head_classes(tmp_path, [BANANA])


def test_method_metrics_scores_like_the_brief():
    import numpy as np

    from src.finetune import method_metrics

    metrics = method_metrics(
        level=np.array([0, 2, 1]),
        ok=np.array([1, 1, 0]),
        truth=np.array([0, 1, 2]),
        true_ok=np.array([1, 0, 0]),
    )

    assert metrics["stock_points"] == pytest.approx((1 - 3 - 2) / 3)
    assert metrics["identification_points"] == pytest.approx((1 - 2 + 1) / 3)
    assert metrics["points_per_area"] == pytest.approx(-4 / 3)
    assert metrics["exact_stock"] == pytest.approx(1 / 3)
    assert metrics["half_found"] == [0, 1]
    assert metrics["confusion"] == {
        "labels": ["empty", "half", "full"],
        "rows_truth_columns_predicted": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
    }
