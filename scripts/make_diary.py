"""Speculative-decoding paper diary (lit/diary.html): per paper, what it introduced and why it falls short.

Entries come from the lineage map (lineage_data.py) and the 2025-26 catalogue (lit/papers_2026.json); the
prose for each was drafted into /tmp/diary/out*.json (style: /tmp/diary/STYLE.md) and is kept in
lit/diary.json. Run with --merge after redrafting to refresh lit/diary.json; otherwise it renders from it.

  python3 scripts/make_diary.py [--merge]
"""
import collections
import glob
import html
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/lit/diary.json"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
FAMILIES = ["origins", "what drafts", "draft shape & verification", "systems & serving",
            "hybrid models & exact numerics"]


def merge():
    base = json.load(open("/tmp/diary/base.json"))
    prose = {}
    for path in sorted(glob.glob("/tmp/diary/out*.json")):
        prose.update(json.load(open(path)))
    names = {r["name"]: r["key"] for r in base}
    out = []
    for r in base:
        p = prose.get(r["key"])
        if not p:
            print("missing prose:", r["key"])
            continue
        answered = []
        for prob, kids in (r.get("later_fixes") or {}).items():
            for k in kids:
                if k in names and names[k] not in [a[0] for a in answered]:
                    answered.append((names[k], k))
        out.append({"key": r["key"], "name": r["name"], "venue": r["venue"] or "", "date": r["date"] or "",
                    "family": r["family"], "tag": p.get("tag", ""), "arxiv": r["arxiv"], "on_path": r["on_path"],
                    "proposed": r["proposed"], "intro": p["intro"], "why": p["why_short"], "answered": answered})
    json.dump(out, open(DATA, "w"), indent=1, ensure_ascii=False)
    print(len(out), "entries ->", DATA)


def e(s):
    return html.escape(s or "", quote=True)


def entry(r):
    y, m = (r["date"].split("-") + ["", ""])[:2] if r["date"] else ("", "")
    link = (f'<a class="id" href="https://arxiv.org/abs/{e(r["arxiv"])}" target="_blank" rel="noopener">'
            f'arXiv {e(r["arxiv"])}</a>' if r["arxiv"] else "")
    badge = '<span class="badge">our path</span>' if r["on_path"] and not r["proposed"] else ""
    nxt = ""
    if r["answered"]:
        nxt = ('<p class="next"><span class="lbl">Answered by</span> '
               + ", ".join(f'<a href="#{e(k)}">{e(n)}</a>' for k, n in r["answered"]) + "</p>")
    search = " ".join([r["name"], r["venue"], r["tag"], r["family"], r["intro"], r["why"], r["arxiv"] or ""]).lower()
    cls = "entry" + (" path" if r["on_path"] else "") + (" proposed" if r["proposed"] else "")
    return f'''<article class="{cls}" id="{e(r["key"])}" data-family="{e(r["family"])}" data-path="{int(r["on_path"])}" data-text="{e(search)}">
  <div class="when"><span class="mon">{e(MONTHS[int(m) - 1] if m else "")}</span> <span class="yr">{e(y)}</span></div>
  <div class="body">
    <header><h3>{e(r["name"])}</h3><span class="meta"><span class="venue">{e(r["venue"])}</span>{link}<span class="tag">{e(r["tag"])}</span>{badge}</span></header>
    <p><span class="lbl">Introduces</span> {e(r["intro"])}</p>
    <p class="why"><span class="lbl">Falls short</span> {e(r["why"])}</p>
    {nxt}
  </div>
</article>'''


def render():
    rows = json.load(open(DATA))
    rows.sort(key=lambda r: (r["proposed"], r["date"] or "9999"))
    years = sorted({r["date"][:4] for r in rows if r["date"] and not r["proposed"]})
    body, last_year = [], None
    for r in rows:
        y = "proposal" if r["proposed"] else (r["date"] or "")[:4]
        if y != last_year:
            label = "Our proposal" if y == "proposal" else y
            body.append(f'<h2 class="year" id="y{e(y)}"><span>{e(label)}</span></h2>')
            last_year = y
        body.append(entry(r))
    counts = collections.Counter(r["family"] for r in rows if not r["proposed"])
    chips = "".join(f'<button type="button" class="chip" data-family="{e(f)}" aria-pressed="false">{e(f)} '
                    f'<span class="n">{counts[f]}</span></button>' for f in FAMILIES if counts[f])
    jumps = " ".join(f'<a href="#y{y}">{y}</a>' for y in years) + ' <a href="#yproposal">proposal</a>'
    n_path = sum(r["on_path"] for r in rows)
    page = TEMPLATE.replace("%CHIPS%", chips).replace("%JUMPS%", jumps).replace("%ENTRIES%", "\n".join(body)) \
        .replace("%N%", str(sum(not r["proposed"] for r in rows))).replace("%NPATH%", str(n_path))
    open(f"{ROOT}/lit/diary.html", "w").write(page)
    print("wrote lit/diary.html")


TEMPLATE = r'''<title>Speculative Decoding Diary</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a dated research log; a timeline spine marks months, the path to our proposal runs in blue. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --ink: #161a21; --ink-2: #545c6a; --rule: #dde1e8;
  --path: #2364c9; --path-soft: #e7eefb; --chip: #eceff4;
  --display: "Newsreader", "Iowan Old Style", Georgia, serif;
  --text: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1216; --surface: #161a20; --ink: #e5e8ed; --ink-2: #9ba3b1; --rule: #272d36;
  --path: #74a6ff; --path-soft: #17233a; --chip: #1e242d; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f1216; --surface: #161a20; --ink: #e5e8ed; --ink-2: #9ba3b1; --rule: #272d36;
  --path: #74a6ff; --path-soft: #17233a; --chip: #1e242d; color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--ink); font: 14.5px/1.55 var(--text); padding-inline: 16px; padding-block: 0 48px }
.wrap { max-width: 980px; margin: 0 auto }
a { color: var(--path) }
a:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid var(--path); outline-offset: 2px }
.intro { padding-block: 40px 20px; display: grid; gap: 12px }
h1 { font: 600 clamp(30px, 6vw, 44px)/1.1 var(--display); margin: 0; text-wrap: balance; letter-spacing: -0.01em }
.lede { color: var(--ink-2); max-width: 64ch; margin: 0 }
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 13px; color: var(--ink-2); margin: 0 }
.legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; border: 2px solid var(--ink-2) }
.legend .dot.p { background: var(--path); border-color: var(--path) }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 2; background: color-mix(in srgb, var(--bg) 92%, transparent);
  backdrop-filter: blur(6px); border-bottom: 1px solid var(--rule); padding-block: 10px; display: grid; gap: 8px }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center }
#q { flex: 1 1 220px; min-width: 0; font: inherit; padding: 7px 10px; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); color: var(--ink) }
.chip, .toggle { font: 500 13px var(--text); padding: 5px 10px; border-radius: 999px; border: 1px solid var(--rule); background: var(--chip); color: var(--ink); cursor: pointer }
.chip[aria-pressed="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink) }
.toggle[aria-pressed="true"] { background: var(--path); color: var(--bg); border-color: var(--path) }
.chip .n { font-family: var(--mono); font-size: 11px; opacity: .7 }
.count, .jumps { font-size: 12.5px; color: var(--ink-2) }
.jumps a { font-family: var(--mono); text-decoration: none; margin-right: 6px }
.log { position: relative; margin-top: 8px }
.year { font: 600 22px var(--display); margin: 28px 0 6px; padding-left: 104px; color: var(--ink) }
.entry { display: grid; grid-template-columns: 92px 1fr; gap: 0 12px; position: relative; padding-block: 14px; border-bottom: 1px solid var(--rule) }
.entry::before { content: ""; position: absolute; left: 92px; top: 0; bottom: 0; border-left: 1px solid var(--rule) }
.entry::after { content: ""; position: absolute; left: 88px; top: 22px; width: 9px; height: 9px; border-radius: 50%; background: var(--surface); border: 2px solid var(--ink-2) }
.entry.path::after { background: var(--path); border-color: var(--path) }
.when { font: 12.5px/1.3 var(--mono); color: var(--ink-2); padding-top: 3px; padding-right: 12px; text-align: right }
.when .mon { color: var(--ink); font-weight: 500 }
.body { min-width: 0; padding-left: 14px; display: grid; gap: 6px }
.body header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 10px }
h3 { font: 600 19px/1.25 var(--display); margin: 0 }
.entry.path h3 { color: var(--path) }
.meta { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; font-size: 12.5px; color: var(--ink-2) }
.venue { font-style: italic }
.id { font-family: var(--mono); font-size: 12px; text-decoration: none }
.tag { font-family: var(--mono); font-size: 11.5px; background: var(--chip); padding: 1px 7px; border-radius: 4px; color: var(--ink-2) }
.badge { font: 600 11px var(--text); letter-spacing: .04em; text-transform: uppercase; color: var(--path) }
.body p { margin: 0; max-width: 92ch }
.lbl { font: 600 11px var(--text); letter-spacing: .06em; text-transform: uppercase; color: var(--ink-2); margin-right: 6px }
.why .lbl { color: var(--ink-2) }
.next { font-size: 13.5px; color: var(--ink-2) }
.next a { text-decoration: none; border-bottom: 1px solid color-mix(in srgb, var(--path) 40%, transparent) }
.entry.proposed { background: var(--path-soft); border: 1px solid var(--path); border-radius: 8px; padding-inline: 0 14px; margin-top: 6px }
.entry.proposed h3 { color: var(--path) }
.entry:target { background: var(--path-soft) }
.empty { color: var(--ink-2); padding: 24px 0 }
@media (max-width: 560px) {
  .entry { grid-template-columns: 1fr; padding-left: 18px }
  .entry::before { left: 4px } .entry::after { left: 0 }
  .when { text-align: left; padding: 0 0 2px 14px }
  .year { padding-left: 18px }
}
@media (prefers-reduced-motion: no-preference) { .entry:target { transition: background .6s } }
</style>
<div class="wrap">
  <section class="intro">
    <h1>Speculative Decoding Diary</h1>
    <p class="lede">%N% papers on lossless speculative decoding, from blockwise parallel decoding (2018) to October 2026, in the order they appeared. Each entry says what the paper introduced and why it falls short; “Answered by” jumps to the later papers that took that limitation on. Entries in blue trace the line of work that led to our proposal, group-aware drafting, which closes the diary.</p>
    <p class="legend"><span><span class="dot"></span>paper</span><span><span class="dot p"></span>on the path to group-aware drafting</span><span>Dates are first arXiv versions; venues are first peer-reviewed venues.</span></p>
  </section>
  <div class="bar">
    <div class="row"><input id="q" type="search" placeholder="Search names, ideas, limitations…" aria-label="Search the diary">
      <button type="button" class="toggle" id="pathonly" aria-pressed="false">Path to our proposal</button></div>
    <div class="row">%CHIPS%</div>
    <div class="row"><span class="count" id="count"></span><span class="jumps">Jump to %JUMPS%</span></div>
  </div>
  <main class="log" id="log">
%ENTRIES%
  <p class="empty" id="empty" hidden>No entries match. Clear the search or the filters to see the full diary.</p>
  </main>
</div>
<script>
(() => {
  const q = document.getElementById("q"), pathBtn = document.getElementById("pathonly");
  const chips = [...document.querySelectorAll(".chip")], entries = [...document.querySelectorAll(".entry")];
  const years = [...document.querySelectorAll(".year")], count = document.getElementById("count");
  const store = { get() { try { return JSON.parse(localStorage.getItem("diary-filters") || "{}") } catch { return {} } },
                  set(v) { try { localStorage.setItem("diary-filters", JSON.stringify(v)) } catch {} } };
  const saved = store.get();
  if (saved.q) q.value = saved.q;
  if (saved.path) pathBtn.setAttribute("aria-pressed", "true");
  (saved.fams || []).forEach(f => chips.forEach(c => { if (c.dataset.family === f) c.setAttribute("aria-pressed", "true") }));
  function apply() {
    const terms = q.value.toLowerCase().split(/\s+/).filter(Boolean);
    const fams = chips.filter(c => c.getAttribute("aria-pressed") === "true").map(c => c.dataset.family);
    const pathOnly = pathBtn.getAttribute("aria-pressed") === "true";
    let shown = 0;
    entries.forEach(el => {
      const ok = terms.every(t => el.dataset.text.includes(t)) && (!fams.length || fams.includes(el.dataset.family))
        && (!pathOnly || el.dataset.path === "1");
      el.hidden = !ok; if (ok) shown++;
    });
    years.forEach(h => { let n = h.nextElementSibling, any = false;
      while (n && !n.classList.contains("year")) { if (n.classList.contains("entry") && !n.hidden) any = true; n = n.nextElementSibling }
      h.hidden = !any });
    count.textContent = `Showing ${shown} of ${entries.length}`;
    document.getElementById("empty").hidden = shown > 0;
    store.set({ q: q.value, path: pathOnly, fams });
  }
  q.addEventListener("input", apply);
  pathBtn.addEventListener("click", () => { pathBtn.setAttribute("aria-pressed", String(pathBtn.getAttribute("aria-pressed") !== "true")); apply() });
  chips.forEach(c => c.addEventListener("click", () => { c.setAttribute("aria-pressed", String(c.getAttribute("aria-pressed") !== "true")); apply() }));
  apply();
})();
</script>
'''

if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    render()
