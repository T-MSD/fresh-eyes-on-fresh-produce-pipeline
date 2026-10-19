import json

import numpy as np
import pytest

from src.crop_areas import main as crop
from tests.builders import area, encode, write_dataset, write_frame_bytes


def test_crop_accepts_productd_and_preserves_negative_corners(tmp_path):
    _, images = write_dataset(
        tmp_path / "data",
        {"6383": [area("6383_2", product_key="productd", top_left=[-12, 350])]},
        {"6383": [100]},
    )
    output = tmp_path / "crops"
    crop(data_dir=images, out_dir=output)
    items = json.loads((output / "index.json").read_text())
    assert items[0]["product_id"] == "11908"
    assert items[0]["polygon"][0] == [-12, 350]
    assert (output / items[0]["crop"]).exists()


@pytest.mark.parametrize("readable", [True, False])
def test_unreadable_frame_gets_every_area_as_fallback(tmp_path, readable):
    _, images = write_dataset(
        tmp_path / "data",
        {"6215": [area("6215_1"), area("6215_2")]},
        {"6215": [100] if readable else []},
    )
    write_frame_bytes(images, "6215", 200, b"broken jpeg")
    output = tmp_path / "crops"
    crop(data_dir=images, out_dir=output)
    fallbacks = json.loads((output / "fallbacks.json").read_text())
    assert set(fallbacks["6215"]["200.jpg"]) == {"6215_1", "6215_2"}
    assert all(
        label == {"planogram_compliant": True, "stock_level": "empty", "stock_value": 0.0}
        for label in fallbacks["6215"]["200.jpg"].values()
    )
    items = json.loads((output / "index.json").read_text())
    assert len(items) == (2 if readable else 0)


def test_all_unseen_products_keep_classifier_without_loading_model(tmp_path):
    from src.prediction_io import NetworkCrop
    from src.run_test import blend_network

    probabilities = np.array([[0.2, 0.3, 0.5]])
    result = blend_network(
        [NetworkCrop(product_name="MANGA CNT KG", crop="100.jpg")],
        tmp_path,
        ["bananas"],
        probabilities,
    )
    np.testing.assert_array_equal(result, probabilities)


def test_new_unreadable_frames_produce_complete_submission_without_training_artifacts(tmp_path):
    from src.run_test import main

    _, images = write_dataset(
        tmp_path / "data", {"6215": [area("6215_1"), area("6215_2")]}, {"6215": []}
    )
    write_frame_bytes(images, "6215", 200, b"broken jpeg")
    output = tmp_path / "predictions.json"
    main(data_dir=images, selftest=False, out_file=output, crops_dir=tmp_path / "crops")
    result = json.loads(output.read_text())
    assert set(result["6215"]["200.jpg"]) == {"6215_1", "6215_2"}
    assert all(label["stock_level"] == "empty" for label in result["6215"]["200.jpg"].values())


def test_final_labels_regenerate_for_all_cameras_with_manual_precedence():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from src.finetune import regenerate_labels

    items = [
        {"key": f"{camera}/100.jpg/{camera}_1", "camera": camera}
        for camera in ["6215", "6383", "6443", "6591"]
    ]
    manual = {
        items[0]["key"]: {"planogram_compliant": True, "stock_level": "empty"},
        items[1]["key"]: {"planogram_compliant": False, "stock_level": "full"},
        items[2]["key"]: {
            "planogram_compliant": True,
            "stock_level": "half",
            "unsure": True,
        },
    }
    zero = {item["key"]: {"planogram_compliant": True} for item in items}
    embedding = np.array([[-1.0, 0], [1.0, 0], [0, 1.0], [1.0, 0]])
    labels = regenerate_labels(items, manual, embedding, np.ones((4, 3)), zero)
    assert set(labels) == {item["key"] for item in items}
    assert labels[items[0]["key"]] == (True, "empty", "manual", True)
    assert labels[items[1]["key"]] == (False, "full", "manual", True)
    assert labels[items[2]["key"]] == (True, "half", "manual", False)
    assert labels[items[3]["key"]][2] == "clip"


def test_held_out_frames_get_no_training_label():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from src.finetune import regenerate_labels

    items = [{"key": f"6215/{ts}.jpg/6215_1", "camera": "6215"} for ts in (100, 200, 300)]
    manual = {
        items[0]["key"]: {"planogram_compliant": True, "stock_level": "empty"},
        items[1]["key"]: {"planogram_compliant": True, "stock_level": "full"},
        items[2]["key"]: {"planogram_compliant": True, "stock_level": "full"},
    }
    zero = {item["key"]: {"planogram_compliant": True} for item in items}
    embedding = np.array([[-1.0, 0], [1.0, 0], [1.0, 0]])
    labels = regenerate_labels(
        items, manual, embedding, np.ones((3, 3)), zero, held={"6215/300.jpg"}
    )
    assert set(labels) == {items[0]["key"], items[1]["key"]}


def write_training_inputs(tmp_path) -> list[dict]:
    crops = tmp_path / "crops"
    crops.mkdir()
    items = []
    manual = {}
    for n, level in enumerate(["empty", "half", "full"]):
        filename = f"{100 + n}.jpg"
        key = f"6215/{filename}/6215_1"
        (crops / filename).write_bytes(encode(64, 64))
        items.append(
            {
                "key": key,
                "camera": "6215",
                "image": filename,
                "area_id": "6215_1",
                "product_name": "BANANA CNT KG",
                "crop": filename,
            }
        )
        manual[key] = {"planogram_compliant": True, "stock_level": level}
    (crops / "index.json").write_text(json.dumps(items))
    (tmp_path / "labels.json").write_text(json.dumps(manual))
    (tmp_path / "clip_zeroshot.json").write_text(json.dumps(manual))
    fill = {key: {"fill": 20.0 + 30 * n} for n, key in enumerate(manual)}
    (tmp_path / "fill_labels.json").write_text(json.dumps(fill))
    np.savez(tmp_path / "clip_embeddings.npz", embeddings=np.eye(3))
    np.save(tmp_path / "clip_level_probs.npy", np.eye(3))
    return items


TINY = {"backbone": "resnet18", "pretrained": False, "size": 64, "epochs": 1}


def test_final_training_rebuilds_model_without_labels_all(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from src.finetune import main

    items = write_training_inputs(tmp_path)
    main(final=True, out_dir=tmp_path, **TINY)
    assert not (tmp_path / "labels_all.json").exists()
    assert (tmp_path / "finetune" / "model.pt").stat().st_size > 0
    assert (tmp_path / "finetune" / "bundle.joblib").stat().st_size > 0
    assert json.loads((tmp_path / "finetune" / "classes.json").read_text()) == ["bananas"]
    predictions = json.loads((tmp_path / "finetune" / "predictions.json").read_text())
    assert set(predictions["6215"]) == {item["image"] for item in items}


def test_final_training_uses_every_label_even_with_a_split(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from src.finetune import main

    write_training_inputs(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test_frames": ["6215/102.jpg"]}))
    main(final=True, split=split, out_dir=tmp_path, **TINY)
    metadata = json.loads((tmp_path / "finetune" / "metadata.json").read_text())
    assert metadata["trained_on"]["manual"] == 3
    assert metadata["inputs"]["split"] is None


def test_evaluation_scores_only_the_split_test_frames(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from src.finetune import main

    write_training_inputs(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test_frames": ["6215/102.jpg"]}))
    main(split=split, out_dir=tmp_path, **TINY)
    metrics = json.loads((tmp_path / "finetune" / "metrics.json").read_text())
    assert metrics["test_set"]["frames"] == 1
    assert metrics["test_set"]["areas"] == 1
    assert metrics["test_set"]["stock_empty_half_full"] == [0, 0, 1]


def test_evaluation_without_a_split_says_how_to_make_one(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import typer

    from src.finetune import main

    write_training_inputs(tmp_path)
    with pytest.raises(typer.BadParameter, match="src.split_labels"):
        main(split=tmp_path / "missing.json", out_dir=tmp_path, **TINY)


def test_score_accounts_for_wrong_compliance_overstock_understock_and_missing():
    from src.prediction_io import Label, score_submission

    labels = {
        "6215/100.jpg/6215_1": Label(planogram_compliant=True, stock_level="empty"),
        "6215/100.jpg/6215_2": Label(planogram_compliant=False, stock_level="full"),
        "6215/100.jpg/6215_3": Label(planogram_compliant=True, stock_level="half"),
    }
    predictions = {
        "6215": {
            "100.jpg": {
                "6215_1": Label(planogram_compliant=False, stock_level="full"),
                "6215_2": Label(planogram_compliant=False, stock_level="half"),
            }
        }
    }
    assert score_submission(predictions, labels) == -11 / 3


def test_merge_fallbacks_preserves_valid_predictions(tmp_path):
    from src.prediction_io import merge_fallbacks

    label = {"planogram_compliant": True, "stock_level": "empty", "stock_value": 0}
    (tmp_path / "fallbacks.json").write_text(
        json.dumps({"6215": {"100.jpg": {"6215_1": label, "6215_2": label}}})
    )
    result = {"6215": {"100.jpg": {"6215_1": {**label, "stock_level": "full", "stock_value": 1}}}}
    merge_fallbacks(result, tmp_path)
    assert result["6215"]["100.jpg"]["6215_1"]["stock_level"] == "full"
    assert result["6215"]["100.jpg"]["6215_2"] == label
