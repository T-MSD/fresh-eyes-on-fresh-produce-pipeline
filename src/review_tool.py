"""Keyboard-driven labelling tool for the planogram-area crops made by crop_areas.py.

    uv run --env-file .env python -m src.review_tool serve        # http://127.0.0.1:8000
    uv run --env-file .env python -m src.review_tool export       # out/annotations.json in the official format

Labels are saved to out/labels.json after every keypress. Suggestions from a model can be put
in out/prelabels.json (same shape as labels.json; experiments/prelabel.py makes it from Vision); they are shown pre-filled and accepted
with Enter.
"""

import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
CROPS = OUT / "crops"
LABELS = OUT / "labels.json"
PRELABELS = OUT / "prelabels.json"
VISION = OUT / "vision_labels.json"  # Google Vision labels per crop, shown as a hint
STOCK_VALUES = {"empty": 0, "half": 0.5, "full": 1}

app = typer.Typer(add_completion=False)
lock = threading.Lock()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_labels(labels: dict) -> None:
    tmp = LABELS.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, indent=1))
    os.replace(tmp, LABELS)


def build_annotations(labels: dict) -> dict:
    """Group labels into {camera: {image: {area_id: {...}}}} as in annotations.json."""
    result: dict = {}
    for key, lab in labels.items():
        camera, image, area_id = key.split("/")
        result.setdefault(camera, {}).setdefault(image, {})[area_id] = {
            "planogram_compliant": lab["planogram_compliant"],
            "stock_level": lab["stock_level"],
            "stock_value": STOCK_VALUES[lab["stock_level"]],
        }
    return result


def order_items(items: list[dict]) -> None:
    """Set item["order"]: suspected wrong products and model-unsure crops first, mixed together.

    Those are the cases the model is weakest on, so a label there is worth the most. Wrong
    products and unsure crops alternate, then the rest follow, most "half"-looking first.
    Cameras alternate throughout, so the labels stay spread out.
    """
    probs_file = OUT / "clip_level_probs.npy"
    if not probs_file.exists():
        return
    import numpy as np

    half = np.load(probs_file)[:, 1]
    pre = load_json(PRELABELS)
    wrong = [not pre.get(i["key"], {}).get("planogram_compliant", True) for i in items]
    unsure = [bool(pre.get(i["key"], {}).get("unsure", False)) for i in items]

    def alternate(a: list[int], b: list[int]) -> list[int]:
        out = []
        while a or b:
            for q in (a, b):
                if q:
                    out.append(q.pop(0))
        return out

    def round_robin(queues: list[list[int]]) -> list[int]:
        out = []
        while any(queues):
            out += [q.pop(0) for q in queues if q]
        return out

    cameras = sorted({i["camera"] for i in items})
    by_half = [int(n) for n in np.argsort(-half)]
    first, rest = [], []
    for cam in cameras:
        mine = [n for n in by_half if items[n]["camera"] == cam]
        w = [n for n in mine if wrong[n]]
        u = [n for n in mine if unsure[n] and not wrong[n]]
        first.append(alternate(w, u))
        rest.append([n for n in mine if not wrong[n] and not unsure[n]])
    for rank, n in enumerate(round_robin(first) + round_robin(rest)):
        items[n]["order"] = rank


def apply_queue(items: list[dict]) -> None:
    """If out/label_queue.json exists (src/pick_labels.py), mark those crops and put them first.

    The random ones are marked "blind": the page hides the model's suggestion for them, so the
    labels measure the model instead of echoing it.
    """
    queue = load_json(OUT / "label_queue.json")
    if not queue:
        return
    informative, blind = queue["informative"], queue.get("random", []) + queue.get("blind", [])
    sequence = []
    step = max(len(informative) // (len(blind) + 1), 1)
    pool = list(informative)
    for _k, key in enumerate(blind):
        sequence += pool[:step] + [key]
        pool = pool[step:]
    sequence += pool
    by_key = {i["key"]: i for i in items}
    for position, key in enumerate(sequence):
        item = by_key[key]
        item["queued"] = True
        item["blind"] = key in blind
        item["order"] = -1000 + position


def make_handler(items: list[dict]):
    # Only files named in the index can be served.
    crops = {item["crop"] for item in items}
    fulls = {(i["camera"], i["image"]): i["image_path"] for i in items}
    html = PAGE.encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body: bytes, ctype: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj):
            self.send(json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            parts = path.strip("/").split("/")
            if path == "/":
                self.send(html, "text/html; charset=utf-8")
            elif path == "/api/items":
                self.send_json(items)
            elif path == "/api/labels":
                self.send_json(
                    {
                        "labels": load_json(LABELS),
                        "prelabels": load_json(PRELABELS),
                        "vision": load_json(VISION),
                    }
                )
            elif parts[0] == "crop" and "/".join(parts[1:]) in crops:
                file = CROPS / "/".join(parts[1:])
                self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "image/jpeg")
            elif parts[0] == "full" and len(parts) == 3 and (parts[1], parts[2]) in fulls:
                self.send(Path(fulls[(parts[1], parts[2])]).read_bytes(), "image/jpeg")
            else:
                self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/label":
                return self.send(b"not found", "text/plain", 404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = body["key"]
            if key not in {i["key"] for i in items}:
                return self.send(b"unknown key", "text/plain", 400)
            with lock:
                labels = load_json(LABELS)
                if body.get("clear"):
                    labels.pop(key, None)
                else:
                    if body["stock_level"] not in STOCK_VALUES:
                        return self.send(b"bad stock_level", "text/plain", 400)
                    labels[key] = {
                        "planogram_compliant": bool(body["planogram_compliant"]),
                        "stock_level": body["stock_level"],
                        "unsure": bool(body.get("unsure", False)),
                    }
                save_labels(labels)
            self.send_json({"ok": True, "done": len(labels)})

    return Handler


RELABEL_RUBRIC = "Rubric (NEW cutoffs): empty 0-34% filled · half 35-79% · full 80-100%. Your old label is pre-filled: Enter keeps it, 1/2/3 changes it. Saved separately in out/labels_v2.json."


@app.command()
def serve(port: int = 8000, host: str = "127.0.0.1", relabel: bool = False) -> None:
    """With --relabel: re-check old labels (out/relabel_queue.json) under the new cutoffs."""
    global LABELS, PRELABELS, PAGE
    index = CROPS / "index.json"
    if not index.exists():
        raise typer.BadParameter(
            "out/crops/index.json missing - run python -m src.crop_areas first"
        )
    items = json.loads(index.read_text())
    if relabel:
        queue = json.loads((OUT / "relabel_queue.json").read_text())
        by_key = {i["key"]: i for i in items}
        items = [by_key[k] for k in queue]
        for n, item in enumerate(items):
            item["queued"], item["order"] = True, n
        LABELS, PRELABELS = OUT / "labels_v2.json", OUT / "labels.json"
        start = PAGE.index('<span class="rubric">')
        PAGE = (
            PAGE[:start]
            + f'<span class="rubric">{RELABEL_RUBRIC}</span>\n</footer>'
            + PAGE[PAGE.index("</footer>") + len("</footer>") :]
        )
        PAGE = PAGE.replace(
            "<title>Shelf labelling</title>", "<title>Relabel (new cutoffs)</title>"
        )
    else:
        order_items(items)
        apply_queue(items)
    typer.echo(f"{len(items)} items, open http://{host}:{port}")
    ThreadingHTTPServer((host, port), make_handler(items)).serve_forever()


@app.command()
def export(out: Path = OUT / "annotations.json") -> None:
    annotations = build_annotations(load_json(LABELS))
    out.write_text(json.dumps(annotations, indent=1))
    n = sum(len(a) for c in annotations.values() for a in c.values())
    typer.echo(f"{n} areas -> {out}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Shelf labelling</title>
<style>
  :root { --bg:#111418; --panel:#1b2027; --fg:#e8ecf1; --dim:#8b95a3; --ok:#3ecf8e; --bad:#ff6b6b; --warn:#f5b942; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.4 system-ui, sans-serif;
         height:100vh; display:flex; flex-direction:column; }
  header { padding:10px 16px; display:flex; gap:16px; align-items:baseline; background:var(--panel); }
  header h1 { font-size:18px; margin:0; }
  header .meta { color:var(--dim); }
  header .progress { margin-left:auto; color:var(--dim); }
  main { flex:1; min-height:0; display:grid; grid-template-columns: 1fr 1fr; gap:12px; padding:12px; }
  .pane { background:var(--panel); border-radius:8px; display:flex; align-items:center;
          justify-content:center; min-height:0; overflow:hidden; position:relative; }
  .pane img { max-width:100%; max-height:100%; object-fit:contain; display:block; }
  #fullwrap { position:relative; max-width:100%; max-height:100%; }
  #fullwrap svg { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
  .state { position:absolute; left:10px; bottom:10px; display:flex; gap:8px; font-weight:600; }
  .chip { padding:4px 10px; border-radius:99px; background:#0008; border:2px solid transparent; }
  .chip.ok { border-color:var(--ok); } .chip.bad { border-color:var(--bad); }
  .chip.suggest { border-style:dashed; border-color:var(--warn); }
  .chip.unsure { border-color:var(--warn); }
  footer { padding:10px 16px; background:var(--panel); display:flex; gap:24px; flex-wrap:wrap; color:var(--dim); }
  kbd { background:#2b333d; color:var(--fg); padding:1px 7px; border-radius:4px; font:13px ui-monospace, monospace; }
  .rubric { color:var(--dim); font-size:13px; }
  .vision { position:absolute; left:10px; top:10px; right:10px; display:flex; gap:6px; flex-wrap:wrap; font-size:12px; }
  .vision span { background:#0009; padding:2px 8px; border-radius:99px; color:var(--dim); }
  .empty { padding:40px; color:var(--dim); font-size:18px; }
</style></head><body>
<header>
  <h1 id="product">loading…</h1>
  <span class="meta" id="meta"></span>
  <span class="progress" id="progress"></span>
</header>
<main>
  <div class="pane"><img id="crop" alt="area crop"><div class="vision" id="vision"></div><div class="state" id="state"></div></div>
  <div class="pane"><div id="fullwrap"><img id="full" alt="full image"><svg id="overlay"></svg></div></div>
</main>
<footer>
  <span><kbd>1</kbd> empty &nbsp;<kbd>2</kbd> half &nbsp;<kbd>3</kbd> full &nbsp;(saves + next)</span>
  <span><kbd>4</kbd> <kbd>5</kbd> <kbd>6</kbd> <b>wrong product</b> + empty / half / full, saves in one key &nbsp;<kbd>X</kbd> / <kbd>U</kbd> wrong / unsure (press <b>before</b> 1/2/3) &nbsp;<kbd>Enter</kbd> accept suggestion</span>
  <span><kbd>←</kbd><kbd>→</kbd> prev/next &nbsp;<kbd>T</kbd> todo only/all &nbsp;<kbd>Backspace</kbd> clear</span>
  <span class="rubric">Rubric: empty &lt;25% filled · half 25-75% · full &gt;75%. If torn between two levels, pick the lower (overestimating costs more). Mixed or wrong product = X.</span>
</footer>
<script>
let items = [], labels = {}, pre = {}, vision = {}, hasQueue = false, view = [], pos = 0, todoOnly = true;
let draft = null;

const $ = id => document.getElementById(id);

async function init() {
  items = await (await fetch('/api/items')).json();
  // Suspected wrong products mixed with model-unsure crops first, then the rest (server's `order`), cameras alternating; else a stable shuffle.
  const h = s => { let x = 5381; for (const c of s) x = ((x * 33) ^ c.charCodeAt(0)) >>> 0;
    x = Math.imul(x ^ (x >>> 16), 2246822507); x = Math.imul(x ^ (x >>> 13), 3266489909); return (x ^ (x >>> 16)) >>> 0; };
  items.sort((a, b) => (a.order ?? 1e9) - (b.order ?? 1e9) || h(a.key) - h(b.key));
  const l = await (await fetch('/api/labels')).json();
  labels = l.labels; pre = l.prelabels; vision = l.vision;
  hasQueue = items.some(i => i.queued);
  rebuild(0);
}

function rebuild(keepIndex) {
  view = items.filter(i => !todoOnly || (!labels[i.key] && (!hasQueue || i.queued)));
  pos = Math.min(Math.max(keepIndex, 0), Math.max(view.length - 1, 0));
  render();
}

function current() { return view[pos]; }

function render() {
  const done = items.filter(i => labels[i.key]).length;
  const q = hasQueue ? ` · queue ${items.filter(i => i.queued && labels[i.key]).length}/${items.filter(i => i.queued).length}` : '';
  $('progress').textContent = `${done}/${items.length} labelled${q} · ${todoOnly ? 'todo' : 'all'} view ${view.length ? pos + 1 : 0}/${view.length}`;
  const it = current();
  if (!it) {
    $('product').textContent = 'Nothing left in this view 🎉';
    $('meta').textContent = 'Press T to see all items, or run `review_tool.py export`.';
    $('crop').removeAttribute('src'); $('full').removeAttribute('src');
    $('overlay').innerHTML = ''; $('state').innerHTML = ''; $('vision').innerHTML = '';
    return;
  }
  $('product').textContent = it.product_name;
  $('meta').textContent = `${it.area_id} · camera ${it.camera} · ${new Date(parseInt(it.image) * 1000).toLocaleString()}`;
  $('crop').src = `/crop/${it.crop}`;
  $('full').src = `/full/${it.camera}/${it.image}`;
  const [w, h] = it.image_size;
  const poly = it.polygon.map(p => p.join(',')).join(' ');
  $('overlay').setAttribute('viewBox', `0 0 ${w} ${h}`);
  $('overlay').innerHTML = `<polygon points="${poly}" fill="#ff3b3b22" stroke="#ff3b3b" stroke-width="3"/>`;
  $('vision').innerHTML = it.blind ? '' : (vision[it.crop] || []).slice(0, 8)
    .map(l => `<span>${l.description} ${Math.round(l.score * 100)}%</span>`).join('');
  const saved = labels[it.key], sug = it.blind ? null : pre[it.key];
  const src = saved || sug;
  draft = { compliant: src ? src.planogram_compliant : true, stock: src ? src.stock_level : null,
            unsure: saved ? !!saved.unsure : false, saved: !!saved };  // a suggestion's 'unsure' is the model's, not yours
  renderState();
}

function renderState() {
  const it = current(); if (!it) return;
  const tag = draft.saved ? '' : (pre[it.key] ? ' suggest' : '');
  $('state').innerHTML =
    `<span class="chip ${draft.compliant ? 'ok' : 'bad'}${tag}">${draft.compliant ? 'compliant' : 'WRONG PRODUCT'}</span>` +
    `<span class="chip${tag}">${draft.stock ?? 'stock: ?'}</span>` +
    (draft.unsure ? '<span class="chip unsure">unsure</span>' : '') +
    (draft.saved ? '<span class="chip">saved</span>' : '<span class="chip unsure">NOT SAVED - ' + (draft.stock ? 'press 1/2/3 or Enter' : 'no stock level yet: press 1/2/3 (Enter does nothing)') + '</span>');
}

async function commit(advance) {
  const it = current();
  if (!it || !draft.stock) return;
  const body = { key: it.key, planogram_compliant: draft.compliant, stock_level: draft.stock, unsure: draft.unsure };
  const r = await fetch('/api/label', { method: 'POST', body: JSON.stringify(body) });
  if (!r.ok) { alert('save failed: ' + await r.text()); return; }
  labels[it.key] = { planogram_compliant: draft.compliant, stock_level: draft.stock, unsure: draft.unsure };
  draft.saved = true;
  if (advance) {
    // In todo view the labelled item drops out, so the same index is now the next one.
    if (todoOnly) rebuild(pos); else { pos = Math.min(pos + 1, view.length - 1); render(); }
  } else renderState();
}

async function clearLabel() {
  const it = current(); if (!it || !labels[it.key]) return;
  await fetch('/api/label', { method: 'POST', body: JSON.stringify({ key: it.key, clear: true }) });
  delete labels[it.key];
  render();
}

function move(d) {
  if (draft && !draft.saved && !draft.compliant &&
      !confirm('Wrong product is marked but NOT saved. Press 1/2/3 (or 4/5/6) to save it. Leave without saving?')) return;
  pos = Math.min(Math.max(pos + d, 0), view.length - 1); render();
}

document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === '1' || k === '2' || k === '3') { draft.stock = ['empty', 'half', 'full'][+k - 1]; commit(true); }
  else if (k === '4' || k === '5' || k === '6') { draft.compliant = false; draft.stock = ['empty', 'half', 'full'][+k - 4]; commit(true); }
  else if (k === 'x') { draft.compliant = !draft.compliant; draft.saved ? commit(false) : renderState(); }
  else if (k === 'u') { draft.unsure = !draft.unsure; draft.saved ? commit(false) : renderState(); }
  else if (k === 'enter') { commit(true); }
  else if (k === 'arrowright') move(1);
  else if (k === 'arrowleft') move(-1);
  else if (k === 't') { todoOnly = !todoOnly; const it = current(); rebuild(0);
    if (it && !todoOnly) { pos = Math.max(view.findIndex(v => v.key === it.key), 0); render(); } }
  else if (k === 'backspace') { clearLabel(); }
  else return;
  e.preventDefault();
});

init();
</script></body></html>
"""


if __name__ == "__main__":
    app()
