"""Build out/check/fill_labelling.html: label package crops with a fill percentage (internal use).

Uses the already-labelled crops of every product the table marks method "fill", spread across products.
The page keeps answers in the browser and exports out/fill_labels.json-style JSON {key: {fill, wrong, unsure}}.

    uv run --env-file .env python -m src.labelling.fill_label_page [--n 150]
    uv run --env-file .env python -m src.labelling.fill_label_page --extra --n 60   # products moved to fill later -> fill_labelling_2.html
"""

import base64
import collections
import json
import random
from pathlib import Path

import cv2
import typer

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "out"

app = typer.Typer(add_completion=False)


@app.command()
def main(n: int = 150, extra: bool = False) -> None:
    items = {i["key"]: i for i in json.loads((OUT / "crops" / "index.json").read_text())}
    labels = json.loads((OUT / "labels.json").read_text())
    groups = json.loads((OUT / "product_groups.json").read_text())
    pool = collections.defaultdict(list)
    if extra:
        # Products moved to "fill" after the first page: every crop of those not covered by out/fill_labels.json.
        done = {items[k]["product_id"] for k in json.loads((OUT / "fill_labels.json").read_text())}
        for k, it in items.items():
            if groups[it["product_id"]]["method"] == "fill" and it["product_id"] not in done:
                pool[it["product_id"]].append(k)
    else:
        for k, _v in labels.items():
            g = groups[items[k]["product_id"]]
            if g["method"] == "fill":
                pool[items[k]["product_id"]].append(k)
    rng = random.Random(4)
    for ks in pool.values():
        rng.shuffle(ks)
    queue = []
    while len(queue) < n and any(pool.values()):  # round-robin over products
        for ks in list(pool.values()):
            if ks and len(queue) < n:
                queue.append(ks.pop())
    rng.shuffle(queue)
    rows = []
    for k in queue:
        it = items[k]
        g = groups[it["product_id"]]
        im = cv2.imread(str(OUT / "crops" / it["crop"]))
        im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        b64 = base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
        rows.append(
            dict(
                key=k,
                product=it["product_name"],
                rule=f"{g['group']}: empty = {g['empty_max_units']} or fewer",
                img=b64,
            )
        )
    page = PAGE.replace("__ROWS__", json.dumps(rows))
    if extra:
        page = (
            page.replace("'fl'", "'fl2'")
            .replace("fill_labels.json", "fill_labels_2.json")
            .replace("<title>Fill Labelling</title>", "<title>Fill Labelling 2</title>")
            .replace("<h1>Fill Labelling</h1>", "<h1>Fill Labelling 2</h1>")
        )
    dest = OUT / "check" / ("fill_labelling_2.html" if extra else "fill_labelling.html")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(page)
    typer.echo(f"{len(rows)} crops from {len({r['product'] for r in rows})} products -> {dest}")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fill Labelling</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e2dc;--acc:#2b7de0}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--fg:#eeeee8;--mut:#9a9a92;--card:#1d1d1b;--line:#33332f;--acc:#5aa0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;padding:16px;max-width:900px;margin-inline:auto}
h1{font-size:20px;margin:0 0 4px}.m{color:var(--mut)}
.top{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:8px}
.bar{height:6px;background:var(--line);border-radius:3px;margin-bottom:12px}.bar i{display:block;height:100%;background:var(--acc);border-radius:3px}
img{width:100%;border-radius:8px;display:block}
.btns{display:grid;grid-template-columns:repeat(auto-fit,minmax(64px,1fr));gap:6px;margin:12px 0}
button{font:inherit;padding:10px 6px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;cursor:pointer}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.row{display:flex;flex-wrap:wrap;gap:16px;align-items:center}label{display:flex;gap:6px;align-items:center}
kbd{background:var(--line);padding:1px 6px;border-radius:4px;font:13px ui-monospace,monospace}
</style></head><body>
<div class="top"><div><h1>Fill Labelling</h1><div class="m" id="p"></div></div><div class="row"><button id="prev">&larr; Back</button><button id="next">Skip &rarr;</button><button id="x">Export JSON</button></div></div>
<div class="bar"><i id="b"></i></div>
<div id="card"></div>
<p class="m">How full is the tray, as a percentage of what it holds when stocked? Keys: <kbd>0</kbd> = 0% · <kbd>1</kbd>–<kbd>9</kbd> = 10–90% · <kbd>f</kbd> = 100% · <kbd>u</kbd> unsure · <kbd>w</kbd> wrong product · <kbd>←</kbd> <kbd>→</kbd> move. Answering moves to the next crop. Saved in this browser; use Export JSON when done.</p>
<script>
const R=__ROWS__;let st={},i=0;try{st=JSON.parse(localStorage.getItem('fl')||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem('fl',JSON.stringify(st))}catch(e){}};
const first=()=>{const n=R.findIndex(r=>st[r.key]===undefined||st[r.key].fill===undefined);return n<0?0:n};i=first();
function show(){const r=R[i],s=st[r.key]||{};const done=R.filter(x=>st[x.key]&&st[x.key].fill!==undefined).length;
 document.getElementById('p').textContent=`${i+1} of ${R.length} · ${done} labelled`;document.getElementById('b').style.width=(100*done/R.length)+'%';
 document.getElementById('card').innerHTML=`<div class="m"><b style="color:var(--fg)">${r.product}</b> · ${r.rule}</div><img src="data:image/jpeg;base64,${r.img}">
 <div class="btns">${[0,10,20,30,40,50,60,70,80,90,100].map(v=>`<button class="${s.fill===v?'on':''}" onclick="set(${v})">${v}%</button>`).join('')}</div>
 <div class="row"><label><input type="checkbox" ${s.unsure?'checked':''} onchange="flag('unsure',this.checked)">unsure</label><label><input type="checkbox" ${s.wrong?'checked':''} onchange="flag('wrong',this.checked)">wrong product</label></div>`}
function flag(k,v){const r=R[i];(st[r.key]=st[r.key]||{})[k]=v;save()}
function set(v){const r=R[i];(st[r.key]=st[r.key]||{}).fill=v;save();if(i<R.length-1)i++;show()}
document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);show()};document.getElementById('next').onclick=()=>{i=Math.min(R.length-1,i+1);show()};
document.onkeydown=e=>{if(e.target.tagName=='INPUT')return;const k=e.key;
 if(k=='0')set(0);else if(/^[1-9]$/.test(k))set(+k*10);else if(k=='f')set(100);else if(k=='u'){const r=R[i];flag('unsure',!(st[r.key]||{}).unsure);show()}
 else if(k=='w'){const r=R[i];flag('wrong',!(st[r.key]||{}).wrong);show()}else if(k=='ArrowLeft'){i=Math.max(0,i-1);show()}else if(k=='ArrowRight'){i=Math.min(R.length-1,i+1);show()}};
document.getElementById('x').onclick=()=>{const o={};R.forEach(r=>{if(st[r.key]&&st[r.key].fill!==undefined)o[r.key]=st[r.key]});const b=new Blob([JSON.stringify(o,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='fill_labels.json';a.click()};
show();
</script></body></html>"""

if __name__ == "__main__":
    app()
