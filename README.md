# Fresh Eyes on Fresh Produce — pipeline + API

Sensei challenge, JunctionX Lisbon 2026. For every camera image and every planogram area, predict
`planogram_compliant`, `stock_level` and `stock_value`.

## Setup

With uv:

```bash
brew install uv
uv sync                 # core: pipeline + API
uv sync --extra ml      # adds torch, transformers, open_clip
```

With pip:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
src/                                    # all source code, flat
dataset/                                # gitignored
  planogram.json
  training/<camera_id>/<unix_ts>.jpg
out/                                    # pipeline outputs (gitignored)
```

## Status (10 Oct 2026)

Labelling, models and predictions are done for now. Everything runs from `src/`, in this order:

| Step | Command | Output |
| --- | --- | --- |
| crop every planogram area | `python src/crop_areas.py` | `out/crops/` |
| Google Vision labels (hint only) | `python src/label_images.py` | `out/vision_labels.json` |
| CLIP zero-shot predictions | `python src/clip_predict.py --topk 5` | `out/clip_zeroshot.json`, `clip_embeddings.npz` |
| hand-label in the browser | `python src/review_tool.py serve` | `out/labels.json` |
| classifier on CLIP embeddings | `python src/train_classifier.py` | `out/predictions.json` |
| fine-tuned network (all labels) | `python src/finetune.py --final` | `out/finetune/` |
| **final submission file** | 75% network + 25% classifier for stock, zero-shot CLIP for compliance | `out/final_predictions.json` |

`finetune.py` without `--final` holds out 3 cameras and scores manual labels only (stock points per
area, max 1.0; identification about +0.98). Latest held-out estimate for the final blend: about +0.07
stock points, 73% exact level, 36 of 84 true halves found. All labels are 404 by hand
(62 empty, 105 half, 237 full); the other 1,558 areas are CLIP's guess and are never used for scoring.

### Parked: new stock cutoffs

Labels follow the original rubric: empty <25%, half 25-75%, full >75%. To be tried later:
empty 0-34%, half 35-79%, full 80-100%. Only two bands move: 25-34% (half to empty) and
76-79% (full to half). No fill percentages were recorded, so crops have to be looked at again:
`python src/pick_relabel.py` builds `out/relabel_queue.json` (111 crops) and
`python src/review_tool.py serve --relabel --port 8001` saves to `out/labels_v2.json` (not started yet).
Then rerun `finetune.py` on both label sets to compare. That test can say which rule the model
separates better, not which one the hidden annotators used.

Review tool keys: 1/2/3 empty/half/full (saves, next), 4/5/6 the same with wrong product,
X wrong product and U unsure (press before 1/2/3), Enter accept suggestion, T todo/all.
