# Rules (mandatory)

1. `dataset/` is read-only. Never edit, move or delete it. Fix data quirks in code.
2. Missing data is a stop condition. Ask; never generate synthetic data.
3. Never overwrite `labels/dev_labels.json`. Change single labels only.
4. The submission contains every area of every frame. Never drop an area.
5. No prediction change lands without a before/after score on the dev labels.
6. Never commit or push without explicit permission.
7. Run Python through `uv run --env-file .env`. Keep all source flat in `src/`.

@./AGENTS.md
