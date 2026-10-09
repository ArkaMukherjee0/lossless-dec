"""Lineage figures for lossless speculative decoding (lit/*.svg, lit/*.pdf), from lineage_data.py.

  l1_tree      tree: parent -- problem (italic) -- + skill -- child
  l2_timeline  problem timeline: one lane per (merged) problem over arXiv date; who raised it, who solved it
  l3_design    design-space matrix: each paper's skills, grouped by axis
  l4_gaps      draft source x other axes: how many papers combine them (blank = untried)

Same academic style as make_schematics.py.

  docker/run.sh python3 scripts/make_lineage.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

from lineage_data import AXES, BANDS, BY_KEY, PAPERS, PATH, PRECURSOR, PROBLEMS, ROOT  # noqa: E402
from make_schematics import GROUP, INK, INK2  # noqa: E402  (also applies the shared rcParams)

OUT = "/workspace/lit"
FAINT = "#b5b5b5"
BLUE, BLUE_LIGHT = "#2a78d6", "#e4eefa"  # the path to our proposal (lineage_data.PATH)
PUBLISHED = [p for p in PAPERS if not p.get("proposed")]  # the timeline, matrix and gaps count published work
plt.rcParams["pdf.fonttype"] = 42  # embed TrueType so the PDF text stays selectable

# Geometry in inches; each figure's axes map 1 data unit to 1 inch, with y growing downward.
FS_NAME, FS_VENUE, FS_PROB, FS_SKILL, FS_BAND = 9.5, 7.5, 7.8, 7.8, 9
BOX_H, PITCH, FAMILY_GAP, BAND_GAP = 0.38, 0.5, 0.16, 0.42
PAD, TRUNK, LABEL_PAD = 0.1, 0.1, 0.07
MARGIN = 0.15


def text_w(s, size, style="normal"):
    prop = FontProperties(family="DejaVu Serif", style=style)
    return 1.06 * max(TextPath((0, 0), line, size=size, prop=prop).get_extents().width
                      for line in s.split("\n")) / 72


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis("off")
    return fig, ax


def line(ax, xs, ys, color=INK, lw=0.8, z=1, **kw):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=z, **kw)


def box(ax, x, y, w, h, ours, edge=INK, face="white"):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=face, edgecolor=edge,
                           linewidth=1.7 if ours else 0.8, zorder=3))


def problem_text(ax, x, y, s, color=INK2, **kw):
    ax.text(x, y, s, fontsize=FS_PROB, color=color, style="italic", linespacing=1.15, zorder=2, **kw)


def skill_text(ax, x, y, s, size=FS_SKILL, color=INK, **kw):
    ax.text(x, y, "+ " + s.replace("\n", "\n   "), fontsize=size, color=color, linespacing=1.15, zorder=2, **kw)


def yy(p):
    return "'" + p["date"][2:4]


# --------------------------------------------------------------------------------------------- tree
def build_tree():
    nodes = {p["key"]: dict(p, branches=[]) for p in PAPERS}
    for p in PAPERS:
        if p["key"] in (ROOT, PRECURSOR):
            continue
        par, prob = p["parents"][0]
        branches = nodes[par]["branches"]
        group = next((kids for q, kids in branches if q == prob), None)
        if group is None:
            branches.append((prob, group := []))
        group.append(nodes[p["key"]])
    root = nodes[ROOT]
    root["branches"].sort(key=lambda b: PROBLEMS[b[0]][0])  # stable: band order, then appearance
    return root, nodes[PRECURSOR]


def walk(n, d=0):
    n["d"] = d
    yield n
    for _, kids in n["branches"]:
        for k in kids:
            yield from walk(k, d + 1)


def tree_layout(root):
    """Columns per depth: [box] trunk  problem  ● bus  + skill  [box] ..."""
    nodes = list(walk(root))
    depth = max(n["d"] for n in nodes) + 1
    colw = [max(max(text_w(n["name"], FS_NAME), text_w(n["venue"], FS_VENUE, "italic")) + 2 * PAD
                for n in nodes if n["d"] == d) for d in range(depth)]
    probw = [max([text_w(PROBLEMS[q][1], FS_PROB, "italic") for n in nodes if n["d"] == d
                  for q, _ in n["branches"]], default=0) for d in range(depth)]
    skillw = [max(text_w("+ " + n["skill"], FS_SKILL) for n in nodes if n["d"] == d) for d in range(depth)]
    xs, bus = [MARGIN], []
    for d in range(depth - 1):
        bus.append(xs[-1] + colw[d] + TRUNK + probw[d] + 3 * LABEL_PAD)
        xs.append(bus[-1] + skillw[d + 1] + 2 * LABEL_PAD)
    for n in nodes:
        n["x"], n["w"] = xs[n["d"]], colw[n["d"]]
        n["bus"] = bus[n["d"]] if n["d"] < depth - 1 else None
    return xs[-1] + colw[-1] + MARGIN


def place(n, cur):
    """Leaves take consecutive rows; a parent sits midway between its first and last junction."""
    if not n["branches"]:
        n["y"] = cur[0]
        cur[0] += PITCH
        return n["y"]
    n["stems"] = []
    for _, kids in n["branches"]:
        ys = [place(k, cur) for k in kids]
        n["stems"].append((ys[0] + ys[-1]) / 2)
    n["y"] = (n["stems"][0] + n["stems"][-1]) / 2
    return n["y"]


def draw_node(ax, n):
    on_path = n["key"] in PATH
    box(ax, n["x"], n["y"], n["w"], BOX_H, n["ours"] or on_path, edge=BLUE if on_path else INK,
        face=BLUE_LIGHT if n.get("proposed") else "white")
    cx = n["x"] + n["w"] / 2
    ax.text(cx, n["y"] - 0.065, n["name"], ha="center", va="center", fontsize=FS_NAME,
            color=BLUE if on_path else INK, zorder=4, weight="bold" if n.get("proposed") else "normal")
    ax.text(cx, n["y"] + 0.095, n["venue"], ha="center", va="center", fontsize=FS_VENUE, color=INK2,
            style="italic", zorder=4)


def draw_edges(ax, n):
    if not n["branches"]:
        return
    xr, xb = n["x"] + n["w"], n["bus"]
    xs = xr
    on = lambda k: n["key"] in PATH and k["key"] in PATH  # noqa: E731  (edge on the blue path)
    hot = dict(color=BLUE, lw=1.8, z=1.5)
    if len(n["stems"]) > 1:
        xs = xr + TRUNK
        line(ax, [xr, xs], [n["y"], n["y"]], **(hot if any(on(k) for _, ks in n["branches"] for k in ks) else {}))
        line(ax, [xs, xs], [n["stems"][0], n["stems"][-1]])
    for (prob, kids), y in zip(n["branches"], n["stems"]):
        path_here = any(on(k) for k in kids)
        line(ax, [xs, xb], [y, y], **(hot if path_here else {}))
        if path_here and len(n["stems"]) > 1:
            line(ax, [xs, xs], [n["y"], y], **hot)
        ax.plot([xb], [y], "o", ms=3.2 if not path_here else 4.2, color=BLUE if path_here else INK, zorder=2)
        problem_text(ax, xs + LABEL_PAD, y - 0.035, PROBLEMS[prob][1], ha="left", va="bottom",
                     color=BLUE if path_here else INK2)
        ky = [k["y"] for k in kids]
        line(ax, [xb, xb], [min(ky + [y]), max(ky + [y])])
        for k in kids:
            if on(k):
                line(ax, [xb, xb], [y, k["y"]], **hot)
            line(ax, [xb, k["x"]], [k["y"], k["y"]], **(hot if on(k) else {}))
            skill_text(ax, xb + LABEL_PAD, k["y"] - 0.035, k["skill"], ha="left", va="bottom",
                       color=BLUE if on(k) else INK)
            draw_edges(ax, k)


def draw_cross_links(ax, nodes):
    """A path node's second parent on the path (e.g. Seer -> group-aware drafting): dashed blue curve."""
    for p in PAPERS:
        if not p.get("proposed"):  # other path nodes' secondary parents are ordinary lineage, not our route
            continue
        for par, prob in p["parents"][1:]:
            if par not in PATH:
                continue
            a, b = nodes[par], nodes[p["key"]]
            start, end = (a["x"] + a["w"], a["y"]), (b["x"] + b["w"], b["y"])  # right edges: clear of neighbours
            ax.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
                                         mutation_scale=12, lw=1.6, color=BLUE, ls=(0, (4, 2)), zorder=1.6))
            problem_text(ax, start[0] + 0.12, start[1] - 0.04, PROBLEMS[prob][1], color=BLUE, ha="left", va="bottom")


def lineage_tree():
    root, pre = build_tree()
    width = tree_layout(root)

    # Bands top to bottom with a little air between families; the root sits level with its first
    # junction so the tree reads from the top-left, and the precursor sits above it.
    cur = [MARGIN + PITCH / 2 + 0.12]
    spans, stems = [], []
    for b, band in enumerate(BANDS):
        y0 = cur[0]
        for prob, kids in root["branches"]:
            if PROBLEMS[prob][0] != b:
                continue
            ys = [place(k, cur) for k in kids]
            stems.append((ys[0] + ys[-1]) / 2)
            cur[0] += FAMILY_GAP
        spans.append((band, y0, cur[0] - PITCH - FAMILY_GAP))
        cur[0] += BAND_GAP
    root["stems"] = stems
    root["y"] = stems[0]
    bottom = cur[0] - BAND_GAP - PITCH + BOX_H / 2
    fig, ax = canvas(width, bottom + 0.55)

    band_x = root["x"] + root["w"] + TRUNK + 0.03
    for band, y0, y1 in spans:
        ax.add_patch(Rectangle((band_x, y0 - PITCH / 2 - 0.12), width - MARGIN - band_x, y1 - y0 + PITCH + 0.2,
                               facecolor=GROUP, edgecolor="none", zorder=0))
        ax.text(width - MARGIN - 0.08, y0 - PITCH / 2 - 0.04, band, ha="right", va="top", fontsize=FS_BAND,
                color=INK2, style="italic")

    draw_edges(ax, root)
    for n in walk(root):
        draw_node(ax, n)
    draw_cross_links(ax, {n["key"]: n for n in walk(root)})

    # Precursor above the root; its dashed edge carries the precursor's problem and the root's skill.
    pre.update(x=root["x"], w=root["w"], y=root["y"] - 1.45)
    draw_node(ax, pre)
    cx = root["x"] + root["w"] / 2
    line(ax, [cx, cx], [pre["y"] + BOX_H / 2, root["y"] - BOX_H / 2], ls=(0, (3, 2)))
    problem_text(ax, cx + LABEL_PAD, pre["y"] + BOX_H / 2 + 0.06, PROBLEMS[root["parents"][0][1]][1],
                 ha="left", va="top")
    skill_text(ax, cx + LABEL_PAD, pre["y"] + BOX_H / 2 + 0.4, root["skill"], ha="left", va="top")

    ax.text(MARGIN, bottom + 0.3,
            "parent ─ problem (italic) ● ─ + skill the child brings ─ child.   Heavy border: run in this repo.   "
            "Blue: the path to our proposal, group-aware drafting (dashed = its second parent).   "
            "Venue = first peer-reviewed venue (arXiv / code otherwise).",
            ha="left", va="center", fontsize=8, color=INK2)
    return fig


# ------------------------------------------------------------------------------------ problem timeline
T0 = (2022, 9)       # left edge of the date axis
MONTHS = 50          # through 2026-10
MW = 0.37            # inches per month
LANE_LABEL_W = 2.45
SUB_H, LANE_GAP = 0.44, 0.1
TBOX_H = 0.36


def month(date):
    y, m = map(int, date.split("-"))
    return (y - T0[0]) * 12 + m - T0[1]


def problem_timeline():
    on_axis = [p for p in PUBLISHED if month(p["date"]) >= 0]
    lanes = {}
    for p in on_axis:
        for i, (par, prob) in enumerate(p["parents"]):
            lane = lanes.setdefault(prob, {"solvers": [], "ghosts": [], "raisers": [], "off_axis": []})
            if p not in lane["solvers"] + lane["ghosts"]:
                lane["ghosts" if i else "solvers"].append(p)
            dest = lane["raisers"] if month(BY_KEY[par]["date"]) >= 0 else lane["off_axis"]
            if BY_KEY[par] not in dest:
                dest.append(BY_KEY[par])
    def first_raised(q):
        return min(p["date"] for p in lanes[q]["raisers"] + lanes[q]["solvers"])
    order = sorted(lanes, key=lambda q: (PROBLEMS[q][0], first_raised(q)))  # stable: then tree order

    x0 = MARGIN + LANE_LABEL_W
    X = lambda d: x0 + (month(d) + 0.5) * MW  # noqa: E731
    xmax = x0 + MONTHS * MW

    # Pack each lane's items (solver boxes, ghost names) into sub-rows so none overlap.
    right = xmax
    for q in order:
        lane = lanes[q]
        items = [(p, True) for p in lane["solvers"]] + [(p, False) for p in lane["ghosts"]]
        items.sort(key=lambda it: it[0]["date"])
        ends, placed = [], []
        for p, solver in items:
            w = (max(text_w(p["name"], 8.8), text_w("+ " + p["skill"].replace("\n", " "), 7.2)) + 2 * PAD
                 if solver else text_w(p["name"], 7.8, "italic") + 0.14)
            left = X(p["date"])
            row = next((r for r, e in enumerate(ends) if e + 0.08 < left), None)
            if row is None:
                ends.append(0)
                row = len(ends) - 1
            ends[row] = left + 0.06 + w
            placed.append((p, solver, row, w))
        lane["items"], lane["rows"] = placed, max(1, len(ends))
        right = max([right] + ends)

    # Vertical layout: bands of lanes; the root lane (band -1) sits on top unshaded.
    y = MARGIN + 0.55
    spans, prev_band = [], None
    for q in order:
        band = PROBLEMS[q][0]
        if band != prev_band:
            if prev_band is not None:
                spans[-1][2] = y - LANE_GAP
                y += 0.25
            spans.append([band, y, None])
            prev_band = band
        lanes[q]["y"] = y + SUB_H / 2
        y += lanes[q]["rows"] * SUB_H + LANE_GAP
    spans[-1][2] = y - LANE_GAP
    height = y + 0.6
    fig, ax = canvas(right + MARGIN + 0.3, height)

    for band, ya, yb in spans:
        if band >= 0:
            ax.add_patch(Rectangle((MARGIN, ya - 0.06), right + 0.3 - MARGIN, yb - ya + 0.12, facecolor=GROUP,
                                   edgecolor="none", zorder=0))
            ax.text(right + 0.25, ya + 0.02, BANDS[band], ha="right", va="top", fontsize=FS_BAND, color=INK2,
                    style="italic", zorder=1)
    for yr in range(T0[0] + 1, T0[0] + 5):
        xg = x0 + month(f"{yr}-01") * MW
        line(ax, [xg, xg], [MARGIN + 0.35, height - 0.45], color="#d0d0d0", lw=0.6, z=0.5)
        for yt in (MARGIN + 0.2, height - 0.32):
            ax.text(xg + 0.05, yt, str(yr), ha="left", va="center", fontsize=9, color=INK2)

    # Paper position: its dot at its date, on the lane line or in a sub-row hanging from it.
    dot = {}
    for q in order:
        for p, solver, row, w in lanes[q]["items"]:
            if solver:
                dot[p["key"]] = (X(p["date"]), lanes[q]["y"] + row * SUB_H)

    for q in order:
        lane, ly = lanes[q], lanes[q]["y"]
        label = PROBLEMS[q][1]
        if lane["off_axis"]:
            label += "\n(" + ", ".join(f"{r['name']} {yy(r)}" for r in lane["off_axis"]) + ")"
        problem_text(ax, MARGIN + 0.05, ly, label, ha="left", va="center")
        xs = [X(p["date"]) for p, *_ in lane["items"]] + [X(r["date"]) for r in lane["raisers"]]
        line(ax, [min(xs), max(xs)], [ly, ly], lw=1.1, z=1)
        for r in lane["raisers"]:  # who raised this problem: a faint drop line from the raiser's dot
            rx, ry = dot[r["key"]]
            line(ax, [rx, rx], [ry, ly], color=FAINT, lw=0.6, z=0.8)
            ax.plot([rx], [ly], "o", ms=4.2, mfc="white", mec=INK, mew=0.8, zorder=2)
        for p, solver, row, w in lane["items"]:
            px, by = X(p["date"]), ly + row * SUB_H
            if row:
                line(ax, [px, px], [ly, by], lw=0.8, z=1)
            if solver:
                ax.plot([px], [by], "o", ms=3.6, color=INK, zorder=4)
                box(ax, px + 0.06, by, w, TBOX_H, p["ours"])
                ax.text(px + 0.06 + PAD, by - 0.07, p["name"], ha="left", va="center", fontsize=8.8, color=INK,
                        zorder=4)
                ax.text(px + 0.06 + PAD, by + 0.09, "+ " + p["skill"].replace("\n", " "), ha="left",
                        va="center", fontsize=7.2, color=INK, zorder=4)
            else:
                ax.plot([px], [by], "o", ms=3.2, color=INK2, zorder=2)
                ax.text(px + 0.1, by - (0.12 if not row else 0), p["name"], ha="left", va="center",
                        fontsize=7.8, color=INK2, style="italic", zorder=2,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6) if row else None)

    ax.text(MARGIN, height - 0.12,
            "Lane = one problem, from when it was first raised (○, drop line from the raising paper) to its latest fix. "
            "Box = paper whose main fix targets this problem (+ its skill); grey italic = paper that also addresses it. "
            "Heavy border: run in this repo. Date = first arXiv version.",
            ha="left", va="center", fontsize=8, color=INK2)
    return fig


# ---------------------------------------------------------------------------------- design-space matrix
def tree_band(p):
    """Band of the root-level problem a paper descends from (-1 for the root and its precursor)."""
    while p["key"] not in (ROOT, PRECURSOR):
        par, prob = p["parents"][0]
        if par == ROOT:
            return PROBLEMS[prob][0]
        p = BY_KEY[par]
    return -1


def design_matrix():
    cols = [(axis, t, label) for axis, vals in AXES for t, label in vals]
    CW, RH, LW, HEAD = 0.27, 0.215, 2.0, 2.1
    axis_gap = 0.14
    xcol, x = [], MARGIN + LW
    for i, (axis, t, label) in enumerate(cols):
        if i and cols[i - 1][0] != axis:
            x += axis_gap
        xcol.append(x + CW / 2)
        x += CW
    table_w = x + MARGIN
    width = table_w + 0.95

    groups = [(-1, "origin")] + list(enumerate(BANDS))
    rows = [(g, label, [p for p in PUBLISHED if tree_band(p) == g]) for g, label in groups]
    nrows = sum(len(r) + 1 for *_, r in rows)
    height = MARGIN + HEAD + nrows * RH + 0.75
    fig, ax = canvas(width, height)

    top = MARGIN + HEAD
    for axis, _ in AXES:  # axis name over its columns
        xs = [xc for (a, *_), xc in zip(cols, xcol) if a == axis]
        ax.text((xs[0] + xs[-1]) / 2, MARGIN + 0.12, axis, ha="center", va="top", fontsize=9, color=INK2,
                style="italic")
        line(ax, [xs[0] - CW / 2 + 0.03, xs[-1] + CW / 2 - 0.03], [MARGIN + 0.32] * 2, color=INK2, lw=0.6)
    for (axis, t, label), xc in zip(cols, xcol):
        ax.text(xc + 0.02, top - 0.08, label, ha="left", va="bottom", rotation=55, rotation_mode="anchor",
                fontsize=8, color=INK)
    line(ax, [MARGIN, table_w - MARGIN], [top, top], lw=1.1)

    y = top + RH * 0.7
    for g, label, ps in rows:
        ax.text(MARGIN, y, label, ha="left", va="center", fontsize=8.5, color=INK2, style="italic")
        y += RH
        for p in ps:
            ax.text(MARGIN + 0.12, y, p["name"], ha="left", va="center", fontsize=8.5, color=INK,
                    weight="bold" if p["ours"] else "normal")
            ax.text(MARGIN + LW - 0.12, y, yy(p), ha="right", va="center", fontsize=7.5, color=INK2)
            for (axis, t, _), xc in zip(cols, xcol):
                if t in p["new"]:
                    ax.plot([xc], [y], "o", ms=5.2, color=INK, zorder=3)
                elif t in p["uses"]:
                    ax.plot([xc], [y], "o", ms=5.2, mfc="white", mec=INK, mew=0.8, zorder=3)
            y += RH
    line(ax, [MARGIN, table_w - MARGIN], [y - RH * 0.4] * 2, lw=1.1)
    for xc in xcol:  # faint column guides
        line(ax, [xc, xc], [top + 0.05, y - RH * 0.5], color="#e6e6e6", lw=0.5, z=0)
    ax.text(MARGIN, y + 0.15, "● the paper's own contribution     ○ inherited / used (best-effort reading)     "
            "bold: run in this repo", ha="left", va="center", fontsize=8, color=INK2)
    return fig


def gap_grid():
    src_axis, *rest = AXES
    srcs = src_axis[1]
    cols = [(axis, t, label) for axis, vals in rest for t, label in vals]
    has = {p["key"]: set(p["new"]) | set(p["uses"]) for p in PUBLISHED}
    count = {(s, t): sum(1 for traits in has.values() if s in traits and t in traits)
             for s, _ in srcs for _, t, _ in cols}

    CW, RH, LW, HEAD = 0.36, 0.3, 2.2, 1.95
    axis_gap = 0.16
    xcol, x = [], MARGIN + LW
    for i, (axis, t, label) in enumerate(cols):
        if i and cols[i - 1][0] != axis:
            x += axis_gap
        xcol.append(x + CW / 2)
        x += CW
    table_w = x + MARGIN
    width = table_w + 0.95
    height = MARGIN + HEAD + len(srcs) * RH + 0.7
    fig, ax = canvas(width, height)
    top = MARGIN + HEAD
    for axis in dict.fromkeys(a for a, *_ in cols):
        xs = [xc for (a, *_), xc in zip(cols, xcol) if a == axis]
        ax.text((xs[0] + xs[-1]) / 2, MARGIN + 0.12, axis, ha="center", va="top", fontsize=9, color=INK2,
                style="italic")
        line(ax, [xs[0] - CW / 2 + 0.03, xs[-1] + CW / 2 - 0.03], [MARGIN + 0.32] * 2, color=INK2, lw=0.6)
    for (axis, t, label), xc in zip(cols, xcol):
        ax.text(xc + 0.02, top - 0.08, label, ha="left", va="bottom", rotation=55, rotation_mode="anchor",
                fontsize=8, color=INK)
    line(ax, [MARGIN, table_w - MARGIN], [top, top], lw=1.1)
    ax.text(MARGIN, top - 0.12, "draft source", ha="left", va="bottom", fontsize=9, color=INK2, style="italic")

    vmax = max(count.values())
    y = top + RH * 0.75
    for s, slabel in srcs:
        ax.text(MARGIN + 0.05, y, slabel, ha="left", va="center", fontsize=8.5, color=INK)
        for (axis, t, _), xc in zip(cols, xcol):
            n = count[(s, t)]
            if n:  # sequential grey: more papers, darker cell; blank cells are the untried combinations
                shade = 0.94 - 0.5 * (n / vmax) ** 0.6
                ax.add_patch(Rectangle((xc - CW / 2 + 0.015, y - RH / 2 + 0.015), CW - 0.03, RH - 0.03,
                                       facecolor=(shade,) * 3, edgecolor="none", zorder=1))
                ax.text(xc, y, str(n), ha="center", va="center", fontsize=8, color=INK if shade > 0.6 else "white",
                        zorder=2)
            else:
                ax.add_patch(Rectangle((xc - CW / 2 + 0.03, y - RH / 2 + 0.03), CW - 0.06, RH - 0.06,
                                       facecolor="none", edgecolor=FAINT, lw=0.5, ls=(0, (2, 2)), zorder=1))
        y += RH
    line(ax, [MARGIN, table_w - MARGIN], [y - RH * 0.45] * 2, lw=1.1)
    ax.text(MARGIN, y + 0.15, "Number of papers whose drafter is of this kind and that also have the column's "
            "trait (● or ○ in the matrix). Dashed empty cell: no paper tried it yet (not every one makes sense).",
            ha="left", va="center", fontsize=8, color=INK2)
    return fig


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, make in [("l1_tree", lineage_tree), ("l2_timeline", problem_timeline),
                       ("l3_design", design_matrix), ("l4_gaps", gap_grid)]:
        fig = make()
        for ext in ("svg", "pdf"):
            fig.savefig(f"{OUT}/{name}.{ext}")
        plt.close(fig)
        print(f"wrote {name}.svg/.pdf")
