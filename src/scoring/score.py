from pydantic import TypeAdapter

from src.prediction.submission import Label, Submission

MANUAL_LABELS = TypeAdapter(dict[str, Label])


def score_submission(predictions: Submission, labels: dict[str, Label]) -> float:
    if not labels:
        raise ValueError("Cannot score an empty label set")
    total = 0
    for key, truth in labels.items():
        camera, filename, area = key.split("/")
        prediction = predictions.get(camera, {}).get(filename, {}).get(area)
        if prediction is None:
            total -= 5
            continue
        total += 1 if prediction.planogram_compliant == truth.planogram_compliant else -2
        if prediction.stock_level == truth.stock_level:
            total += 1
        else:
            total += -2 if prediction.stock_value < truth.stock_value else -3
    return total / len(labels)
