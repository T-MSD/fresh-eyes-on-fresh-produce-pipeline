"""Page to judge the crops where the current submission and the purchase-rule file disagree.

Both files agree everywhere else, so the strategy with more right answers on these crops is the better
one under YOUR rule. Sides are shuffled (A/B) so the page does not hint which is which; the export
records the mapping. Export -> out/compare_strategies.json, then:  python src/compare_strategies.py --score

    python src/compare_strategies.py [--n 100]
    python src/compare_strategies.py --score
"""

import base64
import collections
import json
import random
from pathlib import Path

import cv2
import typer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
POINTS = {("empty", "empty"): 1, ("half", "half"): 1, ("full", "full"): 1}

app = typer.Typer(add_completion=False)


def level(d: dict, key: str) -> str:
    cam, img, area = key.split("/")
    return d[cam][img][area]["stock_level"]


@app.command()
def main(n: int = 100, score: bool = False) -> None:
    items = {i["key"]: i for i in json.loads((OUT / "crops" / "index.json").read_text())}
    cur = json.loads((OUT / "final_predictions.before_hybrid.json").read_text())
    new = json.loads((OUT / "purchase_rule_predictions.json").read_text())
    groups = json.loads((OUT / "product_groups.json").read_text())
    if score:
        return report(items, cur, new)
    diff = [k for k in items if level(cur, k) != level(new, k)]
    by = collections.defaultdict(list)
    for k in diff:
        by[(groups[items[k]["product_id"]]["method"], level(cur, k), level(new, k))].append(k)
    rng = random.Random(3)
    for ks in by.values():
        rng.shuffle(ks)
    pick = []
    while len(pick) < min(n, len(diff)):  # round-robin over the kinds of disagreement
        for ks in by.values():
            if ks and len(pick) < n:
                pick.append(ks.pop())
    rng.shuffle(pick)
    rows = []
    for k in pick:
        it, g = items[k], groups[items[k]["product_id"]]
        im = cv2.imread(str(OUT / "crops" / it["crop"]))
        im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        flip = rng.random() < 0.5
        a, b = (level(new, k), level(cur, k)) if flip else (level(cur, k), level(new, k))
        rows.append(
            dict(
                key=k,
                product=it["product_name"],
                rule=f"{g['group']}: empty = {g['empty_max_units']} or fewer",
                A=a,
                B=b,
                a_is="new" if flip else "current",
                img=base64.b64encode(
                    cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
                ).decode(),
            )
        )
    page = PAGE.replace("__ROWS__", json.dumps(rows))
    dest = OUT / "check" / "compare_strategies.html"
    dest.write_text(page)
    typer.echo(f"{len(diff)} crops differ; {len(rows)} on the page -> {dest}")
    typer.echo({"/".join(k): len(v) for k, v in by.items()})


def report(items, cur, new) -> None:
    f = json.loads((OUT / "compare_strategies.json").read_text())
    tally = collections.Counter()
    for _k, v in f.items():
        side = v.get("pick")
        if side in ("A", "B"):
            tally["new" if (side == "A") == (v["a_is"] == "new") else "current"] += 1
        elif side == "neither":
            tally["neither"] += 1
    typer.echo(
        f"{len(f)} judged: current right {tally['current']}, new right {tally['new']}, neither {tally['neither']}"
    )


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Compare Strategies</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e2dc;--acc:#2b7de0}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--fg:#eeeee8;--mut:#9a9a92;--card:#1d1d1b;--line:#33332f;--acc:#5aa0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;padding:16px;max-width:900px;margin-inline:auto}
h1{font-size:20px;margin:0 0 4px}.m{color:var(--mut)}.top{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:8px}
.bar{height:6px;background:var(--line);border-radius:3px;margin-bottom:12px}.bar i{display:block;height:100%;background:var(--acc);border-radius:3px}
img{width:100%;border-radius:8px;display:block}.btns{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin:12px 0}
button{font:inherit;padding:12px 6px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;cursor:pointer}button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
kbd{background:var(--line);padding:1px 6px;border-radius:4px;font:13px ui-monospace,monospace}.lv{font-weight:700}
</style></head><body>
<div class="top"><div><h1>Compare Strategies</h1><div class="m" id="p"></div></div><div><button id="prev">&larr; Back</button> <button id="x">Export JSON</button></div></div>
<div class="bar"><i id="b"></i></div><div id="card"></div>
<p class="m">Two systems gave different stock levels for this tray. Which one is right under the purchase rule (empty = one typical purchase leaves it bare or nearly bare)? Keys: <kbd>a</kbd> A is right · <kbd>b</kbd> B is right · <kbd>n</kbd> neither. Saved in this browser; Export JSON when done.</p>
<script>
const R=__ROWS__;let st={},i=0;try{st=JSON.parse(localStorage.getItem('cmp')||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem('cmp',JSON.stringify(st))}catch(e){}};
const first=()=>{const n=R.findIndex(r=>!st[r.key]);return n<0?0:n};i=first();
function show(){const r=R[i],s=st[r.key]||'';const done=R.filter(x=>st[x.key]).length;
 document.getElementById('p').textContent=`${i+1} of ${R.length} · ${done} judged`;document.getElementById('b').style.width=(100*done/R.length)+'%';
 document.getElementById('card').innerHTML=`<div class="m"><b style="color:var(--fg)">${r.product}</b> · ${r.rule}</div><img src="data:image/jpeg;base64,${r.img}">
 <div class="btns"><button class="${s=='A'?'on':''}" onclick="set('A')">A is right<br><span class="lv">${r.A}</span></button><button class="${s=='B'?'on':''}" onclick="set('B')">B is right<br><span class="lv">${r.B}</span></button><button class="${s=='neither'?'on':''}" onclick="set('neither')">Neither</button></div>`}
function set(v){st[R[i].key]=v;save();if(i<R.length-1)i++;show()}
document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);show()};
document.onkeydown=e=>{if(e.key=='a')set('A');else if(e.key=='b')set('B');else if(e.key=='n')set('neither');else if(e.key=='ArrowLeft'){i=Math.max(0,i-1);show()}else if(e.key=='ArrowRight'){i=Math.min(R.length-1,i+1);show()}};
document.getElementById('x').onclick=()=>{const o={};R.forEach(r=>{if(st[r.key])o[r.key]={pick:st[r.key],a_is:r.a_is,A:r.A,B:r.B}});const b=new Blob([JSON.stringify(o,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='compare_strategies.json';a.click()};
show();
</script></body></html>"""

if __name__ == "__main__":
    app()
