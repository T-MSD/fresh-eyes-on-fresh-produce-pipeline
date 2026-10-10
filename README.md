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
