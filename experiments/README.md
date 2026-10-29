# Experiments

Scripts from finished or rejected experiments, kept for the record. Nothing in `src/` imports them.
Run them from the repository root as `uv run --extra ml --env-file .env python -m experiments.<script>`.
The production pipeline and its score are in the main README, section Model.

| Script | What it was for |
| --- | --- |
| `train_classifier` | CLIP classifier on its own, evaluated on camera folds; superseded by `src.finetune` (its helpers live in `src/classifier.py`) |
| `label_images`, `prelabel` | Google Cloud Vision labels as hints for the review tool (requires Google credentials) |
| `owl_count` | OWL-ViT v2 item counts for loose products |
| `fill_probe` | compare fill cutoff rules on held-out cameras |
| `purchase_rule_predictions` | strategy B predictions |
| `compare_strategies` | page to judge where strategy A and B disagree |
| `hybrid_predictions` | strategy A plus the strategy B changes judged right: the submitted file |
| `pick_half`, `pick_relabel` | one-off labelling queues for half shelves and for a rubric change |
| `score_predictions` | score two submission files against `out/labels.json` |

## Stock strategies (10 Oct 2026)

Two strategies ship side by side. Compliance is the same in both: the zero-shot CLIP flag.

| | Strategy A: blend | Strategy B: purchase rule |
| --- | --- | --- |
| Stock level | 75% fine-tuned network + 25% CLIP classifier, level with the best expected points | "empty" means one typical purchase leaves the tray bare; loose items counted, packages judged by fill |
| Needs | `out/finetune/model.pt`, `out/labels.json` | product groups, OWL-ViT counts, fill percentages |
| Output | `out/final_predictions.before_hybrid.json` | `out/purchase_rule_predictions.json` (regenerated, not tracked) |

**`out/final_predictions.json` is the hybrid:** strategy A plus the strategy B transition rules
selected from hand judgements. It is the submitted file. On `out/labels.json`, it scores below
strategy A; see Results and limits.

### Strategy A history

Strategy A is the production pipeline in the main README. Before the 80/20 split, evaluation held
out whole cameras and reported +0.49 stock points per area, 85% exact level and
identification +0.96; that test set is retired, and its numbers are not comparable with the
split's. The CLIP classifier also ran on its own (`experiments.train_classifier`, writing
`out/predictions.json`), and Google Vision labels (`experiments.label_images`) were tried as
review-tool hints before CLIP replaced them.

### Strategy B, step by step

| Step | Command | Output |
| --- | --- | --- |
| group every product by purchase size and method (count or fill) | `uv run --extra ml --env-file .env python -m src.product_groups` | `out/product_groups.json` |
| count loose items with OWL-ViT v2 | `uv run --extra ml --env-file .env python -m experiments.owl_count` | `out/owl_counts.json` |
| label fill percentages in the browser | `uv run --extra ml --env-file .env python -m src.fill_label_page` | `out/check/fill_labelling.html` -> `out/fill_labels.json` |
| compare cutoff rules on held-out cameras | `uv run --extra ml --env-file .env python -m experiments.fill_probe` | printed table |
| build strategy B predictions | `uv run --extra ml --env-file .env python -m experiments.purchase_rule_predictions` | `out/purchase_rule_predictions.json` |
| judge where A and B disagree | `uv run --extra ml --env-file .env python -m experiments.compare_strategies`, then `--score` | `out/check/compare_strategies.html` |
| keep the selected transition rules | `uv run --extra ml --env-file .env python -m experiments.hybrid_predictions` | `out/final_predictions.hybrid.json` |

What strategy B does:
- `product_groups.py` puts each product in a group with a typical purchase and an "empty means N units
  or fewer" number. Products whose names end in KG are not always loose: three 1 KG apples, `CEBOLA NOVA CNT KG`
  and all grapes are packaged on the shelf, so a small override list exists.
- **Count products:** the OWL count turns a tray empty only if strategy A does not already say full.
  This prevents low counts from overriding a full prediction.
- **Fill products** (packages, bunches, tomatoes and product overrides): ridge regression on the CLIP
  embedding, trained on hand-labelled fill percentages in 10% steps. Cutoffs: empty 0-15%,
  half 16-64%, full 65-100%.

### Results and limits

Score the submitted candidates against `out/labels.json`:

```bash
uv run --env-file .env python -m experiments.score_predictions
```

| Candidate | Total points | Points per labelled area |
| --- | --- | --- |
| A, before hybrid | +731 | +1.809406 |
| Hybrid, after | +682 | +1.688119 |
| Change | −49 | −0.121287 |

These labels also train strategy A. This is a training-label score, not an estimate of performance
on unseen frames. It penalizes missing areas and uses the repository's compliance and stock
penalties. The hybrid regresses against these labels, so the hand judgements below do not
demonstrate an improvement under this evaluation.

Earlier hand judgements of disagreements under the purchase rule:

| File | Exact stock agreement | Points per crop (project scoring) |
| --- | --- | --- |
| A, the blend | 61% | -0.29 |
| B, purchase rule | 39% | -1.08 |
| hybrid (submitted) | 79% | +0.22 |

- B alone lost to A. The hybrid takes fill half-to-full, fill empty-to-full and count
  empty-to-half transitions. Those categories were picked after seeing the judgements, so the
  +0.22 is not a held-out estimate.
- Cutoffs 0-34/35-79/80-100 and per-group "N units" exceptions did not beat 0-15/16-64/65-100 on
  held-out cameras; the differences were inside the noise (about ±0.4 points).
- Counting failed on packages (YOLO-World found nothing; OWL-ViT was judged unusable by hand on the
  crops checked) and on dense loose trays such as tomatoes, so those products use fill percentages.
  OWL-ViT agreed with a hand count exactly on 54% of checked crops, within one item on 75%.
- The earlier ML experiment ran under Python 3.14. Use the repository's pinned Python 3.12
  for installation, regression tests and subsequent runs.
