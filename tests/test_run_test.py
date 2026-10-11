import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("open_clip")
pytest.importorskip("sklearn")

import joblib  # noqa: E402
import torch  # noqa: E402

from src import run_test  # noqa: E402
from src.clip_predict import LEVELS  # noqa: E402
from src.finetune import Net, build_bundle  # noqa: E402
from tests.builders import area, encode, planogram_json  # noqa: E402

TRAIN_NAMES = ["BANANA CAT I CNT KG", "PERA ROCHA CNT KG", "LIMAO CNT KG"]


def train_items() -> list[dict]:
    return [
        {"key": f"k{n}", "product_name": TRAIN_NAMES[n % 3], "camera": "1", "image": "1.jpg"}
        for n in range(12)
    ]


def write_bundle(out: Path) -> dict:
    rng = np.random.default_rng(0)
    items = train_items()
    manual = {i["key"]: {"stock_level": LEVELS[n % 3]} for n, i in enumerate(items)}
    fill = {i["key"]: float(n * 8) for n, i in enumerate(items)}
    bundle = build_bundle(items, manual, rng.normal(size=(len(items), 512)), fill)
    (out / "finetune").mkdir(parents=True)
    joblib.dump(bundle, out / "finetune" / "bundle.joblib")
    net = Net(run_test.BACKBONE, len(bundle["train_classes"]), pretrained=False)
    torch.save(net.state_dict(), out / "finetune" / "model.pt")
    (out / "product_groups.json").write_text(json.dumps({}))
    return bundle


def write_dataset(root: Path, names: list[str]) -> None:
    root.mkdir(exist_ok=True)
    areas = [area(f"6215_{n}", str(n), name) for n, name in enumerate(names, start=1)]
    (root / "planogram.json").write_bytes(planogram_json({"6215": areas}))
    (root / "6215").mkdir()
    (root / "6215" / "100.jpg").write_bytes(encode(640, 480))
    (root / "6215" / "200.jpg").write_bytes(b"not a jpeg")


@pytest.fixture
def isolated_out(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(run_test, "OUT", out)
    monkeypatch.setattr(run_test, "TEST", out / "test")
    monkeypatch.setattr(run_test, "BUNDLE", out / "finetune" / "bundle.joblib")
    return out


def run(data_dir: Path, out: Path) -> dict:
    out_file = out / "test" / "predictions.json"
    run_test.main(
        data_dir=data_dir,
        selftest=False,
        out_file=out_file,
        crops_dir=out / "test" / "crops",
        topk=5,
    )
    return json.loads(out_file.read_text())


def test_predicts_every_area_from_only_the_bundle_weights_and_product_groups(
    tmp_path, isolated_out
):
    write_bundle(isolated_out)
    write_dataset(tmp_path / "data", ["BANANA CAT I CNT KG", "PERA ROCHA CNT KG"])

    result = run(tmp_path / "data", isolated_out)

    assert sorted(result["6215"]) == ["100.jpg", "200.jpg"]
    assert all(
        sorted(result["6215"][frame]) == ["6215_1", "6215_2"] for frame in ("100.jpg", "200.jpg")
    )


def test_unreadable_frame_areas_get_the_empty_compliant_fallback(tmp_path, isolated_out):
    write_bundle(isolated_out)
    write_dataset(tmp_path / "data", ["BANANA CAT I CNT KG"])

    result = run(tmp_path / "data", isolated_out)

    assert result["6215"]["200.jpg"]["6215_1"]["stock_level"] == "empty"
    assert result["6215"]["200.jpg"]["6215_1"]["planogram_compliant"] is True


def test_every_class_unseen_keeps_the_classifier_prediction_instead_of_crashing(
    tmp_path, isolated_out
):
    write_bundle(isolated_out)
    write_dataset(tmp_path / "data", ["MORANGO CNT KG", "PEPINO CNT KG"])

    result = run(tmp_path / "data", isolated_out)

    assert set(result["6215"]["100.jpg"]) == {"6215_1", "6215_2"}


def test_a_product_no_class_rule_knows_is_scored_compliant_instead_of_stopping_the_run(
    tmp_path, isolated_out
):
    write_bundle(isolated_out)
    write_dataset(tmp_path / "data", ["BANANA CAT I CNT KG", "CEBOLA AMARELA CNT KG"])

    result = run(tmp_path / "data", isolated_out)

    assert result["6215"]["100.jpg"]["6215_1"]["planogram_compliant"] is True
    assert set(result["6215"]["100.jpg"]) == {"6215_1", "6215_2"}
