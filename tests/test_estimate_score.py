import json

import pytest

from src.estimate_score import estimate, main, tally
from src.prediction_io import SUBMISSION


def submission(areas: dict[str, tuple[bool, str]], frame: str = "100.jpg") -> dict:
    return SUBMISSION.validate_python(
        {
            "6215": {
                frame: {
                    area: {"planogram_compliant": compliant, "stock_level": level}
                    for area, (compliant, level) in areas.items()
                }
            }
        }
    )


def test_tally_counts_every_outcome_like_the_brief():
    labels = submission(
        {"6215_1": (True, "half"), "6215_2": (True, "half"), "6215_3": (False, "full")}
    )
    predictions = submission({"6215_1": (True, "empty"), "6215_2": (False, "full")})
    outcome = tally(predictions, labels)
    assert outcome.areas == 3
    assert outcome.missing == 1
    assert (outcome.identification_correct, outcome.identification_wrong) == (1, 2)
    assert (outcome.stock_correct, outcome.stock_below, outcome.stock_above) == (0, 1, 2)
    assert outcome.points == 1 - 4 + 0 - 2 - 6


def test_brief_example_scores_320():
    labels = submission({f"6215_{n}": (True, "half") for n in range(289)})
    predictions = submission(
        {
            f"6215_{n}": (n >= 19, "half" if n < 230 else "empty" if n < 265 else "full")
            for n in range(289)
        }
    )
    assert tally(predictions, labels).points == 320


def test_estimate_scales_to_the_validation_set_with_an_interval():
    labels = submission({"6215_1": (True, "full"), "6215_2": (True, "full")}, "100.jpg")
    labels["6215"].update(submission({"6215_1": (True, "full")}, "200.jpg")["6215"])
    predictions = submission({"6215_1": (True, "full"), "6215_2": (True, "full")}, "100.jpg")
    predictions["6215"].update(submission({"6215_1": (False, "full")}, "200.jpg")["6215"])
    result = estimate(predictions, labels, validation_areas=289)
    assert result.per_area == pytest.approx(1.0)
    assert result.expected_total == pytest.approx(289.0)
    assert result.low <= result.expected_total <= result.high
    assert result.low < result.high


def test_estimate_refuses_empty_labels():
    with pytest.raises(ValueError):
        estimate({}, {}, validation_areas=289)


def test_cli_reads_both_files_and_prints_the_estimate(tmp_path, capsys):
    labels = {
        "6215": {
            "100.jpg": {
                "6215_1": {"planogram_compliant": True, "stock_level": "full", "stock_value": 1}
            }
        }
    }
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "out.json").write_text(json.dumps(labels))
    main(predictions=tmp_path / "out.json", labels=tmp_path / "labels.json")
    printed = capsys.readouterr().out
    assert "+2.000 per area" in printed
    assert "+578" in printed


def test_split_scores_only_the_test_frames(tmp_path, capsys):
    label = {"planogram_compliant": True, "stock_level": "full", "stock_value": 1}
    labels = {"6215": {"100.jpg": {"6215_1": label}, "200.jpg": {"6215_1": label}}}
    predictions = {"6215": {"200.jpg": {"6215_1": label}}}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "out.json").write_text(json.dumps(predictions))
    (tmp_path / "split.json").write_text(json.dumps({"test_frames": ["6215/200.jpg"]}))
    main(
        predictions=tmp_path / "out.json",
        labels=tmp_path / "labels.json",
        split=tmp_path / "split.json",
    )
    printed = capsys.readouterr().out
    assert "1 labelled areas, 0 missing" in printed
    assert "+2.000 per area" in printed
