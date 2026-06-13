"""Render secondary thesis Markdown report to a clean A4 PDF."""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# ── Page geometry ──────────────────────────────────────────────────────────
PAGE_W, PAGE_H = 8.27, 11.69          # A4 portrait (inches)
ML, MR, MT, MB = 0.85, 0.85, 0.80, 0.70

TX    = ML / PAGE_W                    # text left  (axis fraction)
TW    = (PAGE_W - ML - MR) / PAGE_W   # text width
TTY   = 1.0 - MT / PAGE_H             # text top y
TBY   = MB / PAGE_H                   # text bottom y
TW_IN = PAGE_W - ML - MR              # text width in inches ≈ 6.57"


def lh(in_: float) -> float:
    return in_ / PAGE_H


LH_H1   = lh(0.36)
LH_H2   = lh(0.28)
LH_BODY = lh(0.175)
LH_TBL  = lh(0.162)   # height per text line inside a table cell
LH_GAP  = lh(0.09)

FS_H1 = 17
FS_H2 = 13.5
FS_H3 = 11
FS_BODY = 9.5
FS_TBL  = 8.0
FS_CODE = 8.2

ACCENT   = "#1A4C8B"
TBL_HDR  = "#D0E4F4"
TBL_EVEN = "#F2F7FB"
TBL_ODD  = "#FFFFFF"

# Empirical wrap widths for the fonts/sizes used
WRAP_BODY = 88    # chars at FS_BODY DejaVu Sans across TW_IN
TBL_CHARS = 96    # total chars across the full table width at FS_TBL


# ── Entry point ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=Path, default=Path("SECONDARY_THESIS_REPORT.md"))
    ap.add_argument("--output", type=Path, default=Path("SECONDARY_THESIS_REPORT.pdf"))
    a = ap.parse_args()
    render(a.input, a.output)
    print(f"saved PDF to {a.output}")


# ── Main render loop ───────────────────────────────────────────────────────
def render(md: Path, out: Path) -> None:
    lines = md.read_text(encoding="utf-8").splitlines()
    base  = md.parent

    with PdfPages(out) as pdf:
        pg = _Page(pdf)
        i  = 0
        while i < len(lines):
            raw = lines[i]

            # Image — flush current text page, then dedicate a full page to the image
            m = re.match(r"!\[(?P<cap>.*?)\]\((?P<p>.*?)\)", raw.strip())
            if m:
                pg.flush()
                _image_page(pdf, base / m["p"], m["cap"])
                pg.reset()
                i += 1
                continue

            # Table — collect all consecutive pipe-delimited lines
            if raw.strip().startswith("|"):
                block: list[str] = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    block.append(lines[i])
                    i += 1
                _table(pg, block)
                continue

            # Heading
            lvl = _hlevel(raw.strip())
            if lvl:
                text = _strip(raw.strip().lstrip("#").strip())
                if lvl == 1:
                    pg.ensure(LH_H1 * 2 + lh(0.25))
                    pg.gap(LH_GAP * 0.5)
                    _h1(pg, text)
                    pg.gap(LH_GAP * 0.4)
                elif lvl == 2:
                    pg.ensure(LH_H2 + LH_BODY + lh(0.25))
                    pg.gap(LH_GAP * 0.6)
                    _h2(pg, text)
                    pg.gap(LH_GAP * 0.3)
                else:
                    pg.ensure(lh(0.25))
                    pg.gap(LH_GAP * 0.4)
                    _emit(pg, text, FS_H3, "bold", lh(0.22))
                i += 1
                continue

            # Blockquote
            if raw.strip().startswith(">"):
                _blockquote(pg, _strip(raw.strip().lstrip(">").strip()))
                i += 1
                continue

            # List item
            li = re.match(r"^(\s*)(\d+\.|[-*])\s+(.*)", raw)
            if li:
                depth = len(li.group(1)) // 2
                num   = li.group(2)
                text  = _strip(li.group(3))
                pre   = "  " * depth + ("• " if not num[0].isdigit() else f"{num} ")
                cont  = " " * len(pre)
                for j, wl in enumerate(_wrap(text, WRAP_BODY - len(pre))):
                    _emit(pg, (pre if j == 0 else cont) + wl, FS_BODY, "normal", LH_BODY)
                i += 1
                continue

            # Standalone inline code path
            if re.match(r"^`[^`]+`$", raw.strip()):
                _emit(pg, raw.strip().strip("`"), FS_CODE, "normal", LH_BODY, mono=True)
                i += 1
                continue

            # Horizontal rule
            if re.match(r"^-{3,}$", raw.strip()):
                pg.gap(LH_GAP)
                i += 1
                continue

            # Blank line
            if not raw.strip():
                pg.gap(LH_GAP * 0.4)
                i += 1
                continue

            # Paragraph text
            for wl in _wrap(_strip(raw.strip()), WRAP_BODY):
                _emit(pg, wl, FS_BODY, "normal", LH_BODY)
            i += 1

        pg.flush()


# ── Page state ─────────────────────────────────────────────────────────────
class _Page:
    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self._new()

    def _new(self) -> None:
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.ax  = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.y     = TTY
        self.dirty = False

    def reset(self) -> None:
        self._new()

    def flush(self) -> None:
        if self.dirty:
            self.pdf.savefig(self.fig, bbox_inches="tight")
        plt.close(self.fig)
        self._new()

    def gap(self, h: float) -> None:
        self.y -= h
        if self.y < TBY:
            self.flush()

    def ensure(self, h: float) -> None:
        if self.y - h < TBY:
            self.flush()

    def text(self, x: float, y: float, s: str, **kw: object) -> None:
        self.ax.text(x, y, s, transform=self.ax.transAxes,
                     ha=kw.pop("ha", "left"), va="top", clip_on=True, **kw)
        self.dirty = True

    def rect(self, x: float, y: float, w: float, h: float, c: str) -> None:
        p = mpatches.FancyBboxPatch(
            (x, y - h), w, h,
            boxstyle="square,pad=0",
            facecolor=c, edgecolor="none",
            transform=self.ax.transAxes, clip_on=True,
        )
        self.ax.add_patch(p)
        self.dirty = True

    def hline(self, x0: float, x1: float, y: float, c: str, lw: float = 0.7) -> None:
        self.ax.plot([x0, x1], [y, y], color=c, lw=lw,
                     transform=self.ax.transAxes, clip_on=True)
        self.dirty = True

    def advance(self, h: float) -> None:
        self.y -= h


# ── Text helpers ───────────────────────────────────────────────────────────
def _emit(pg: _Page, text: str, fs: float, fw: str, lh_: float,
          mono: bool = False, color: str = "#111111",
          x: float | None = None) -> None:
    pg.ensure(lh_)
    pg.text(
        x if x is not None else TX, pg.y, text,
        fontsize=fs, fontweight=fw,
        family="DejaVu Sans Mono" if mono else "DejaVu Sans",
        color=color,
    )
    pg.advance(lh_)


def _h1(pg: _Page, text: str) -> None:
    # Rule above, wrapped title text, rule below
    pg.hline(TX, TX + TW, pg.y + lh(0.07), ACCENT, lw=1.5)
    for wl in _wrap(text, 52):
        pg.text(TX, pg.y, wl, fontsize=FS_H1, fontweight="bold",
                family="DejaVu Sans", color=ACCENT)
        pg.advance(LH_H1)
    pg.hline(TX, TX + TW, pg.y + lh(0.02), ACCENT, lw=0.5)


def _h2(pg: _Page, text: str) -> None:
    pg.text(TX, pg.y, text, fontsize=FS_H2, fontweight="bold",
            family="DejaVu Sans", color=ACCENT)
    pg.advance(LH_H2)
    pg.hline(TX, TX + TW * 0.45, pg.y + lh(0.03), "#AAAAAA", lw=0.5)


def _blockquote(pg: _Page, text: str) -> None:
    pg.gap(LH_GAP * 0.3)
    wlines = _wrap(f'"{text}"', WRAP_BODY - 4)
    bh = len(wlines) * LH_BODY + lh(0.07)
    pg.ensure(bh)
    pg.rect(TX, pg.y, 0.004, bh, ACCENT)
    for wl in wlines:
        pg.text(TX + 0.014, pg.y, wl, fontsize=FS_BODY, fontweight="normal",
                family="DejaVu Sans", color="#333333", fontstyle="italic")
        pg.advance(LH_BODY)
    pg.gap(LH_GAP * 0.3)


# ── Table ──────────────────────────────────────────────────────────────────
def _table(pg: _Page, raw: list[str]) -> None:
    rows:    list[list[str]] = []
    sep_idx: int | None      = None
    aligns:  list[str]       = []

    for i, ln in enumerate(raw):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.match(r"^:?-+:?$", c) for c in cells if c):
            sep_idx = i
            aligns = [
                "right"  if (c.endswith(":") and not c.startswith(":")) else
                "center" if (c.startswith(":") and c.endswith(":"))     else "left"
                for c in cells
            ]
        else:
            rows.append(cells)

    if not rows:
        return

    hdr_n = sep_idx if sep_idx is not None else 1
    n_col = max(len(r) for r in rows)
    if not aligns:
        aligns = ["left"] * n_col

    # Normalise lengths and strip markdown
    clean: list[list[str]] = []
    for r in rows:
        while len(r) < n_col:
            r.append("")
        clean.append([_strip(c) for c in r])

    # Proportional char-width limits per column
    col_max = [max(max(len(r[c]) for r in clean), 1) for c in range(n_col)]
    tot     = sum(col_max)
    col_lim = [max(10, int(m / tot * TBL_CHARS)) for m in col_max]

    # Pre-wrap every cell so nothing overflows its column
    PAD     = lh(0.038)
    body_ri = 0
    plan: list[tuple[list[list[str]], int, bool, str]] = []
    for ri, row in enumerate(clean):
        wcells  = [
            textwrap.wrap(row[ci], col_lim[ci], break_long_words=True) or [""]
            for ci in range(n_col)
        ]
        n_lines = max(len(wc) for wc in wcells)
        is_hdr  = ri < hdr_n
        if is_hdr:
            bg = TBL_HDR
        else:
            bg = TBL_EVEN if body_ri % 2 == 0 else TBL_ODD
            body_ri += 1
        plan.append((wcells, n_lines, is_hdr, bg))

    col_frac = [lim / sum(col_lim) * TW for lim in col_lim]

    pg.gap(LH_GAP * 0.5)
    for wcells, n_lines, is_hdr, bg in plan:
        row_h = n_lines * LH_TBL + PAD * 2
        pg.ensure(row_h)
        pg.rect(TX, pg.y, TW, row_h, bg)

        x = TX
        for ci, cell_lines in enumerate(wcells):
            cw = col_frac[ci]
            al = aligns[ci] if ci < len(aligns) else "left"
            ly = pg.y - PAD
            fw = "bold" if is_hdr else "normal"
            for line in cell_lines:
                if al == "right":
                    pg.text(x + cw - 0.006, ly, line,
                            fontsize=FS_TBL, fontweight=fw,
                            family="DejaVu Sans Mono", color="#111111", ha="right")
                else:
                    pg.text(x + 0.005, ly, line,
                            fontsize=FS_TBL, fontweight=fw,
                            family="DejaVu Sans", color="#111111")
                ly -= LH_TBL
            x += cw

        lc, llw = ("#777777", 0.5) if is_hdr else ("#DDDDDD", 0.3)
        pg.hline(TX, TX + TW, pg.y - row_h, lc, llw)
        pg.advance(row_h)

    pg.gap(LH_GAP * 0.5)


# ── Image page ─────────────────────────────────────────────────────────────
def _image_page(pdf: PdfPages, img_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax  = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Caption bar
    ax.text(0.5, 0.962, caption, ha="center", va="top", fontsize=13,
            fontweight="bold", color=ACCENT, family="DejaVu Sans",
            transform=ax.transAxes)
    ax.plot([0.06, 0.94], [0.932, 0.932], color=ACCENT, lw=0.8,
            transform=ax.transAxes)

    # Path label at very bottom
    ax.text(0.5, 0.034, str(img_path).replace("\\", "/"),
            ha="center", va="bottom", fontsize=7.2, color="#777777",
            family="DejaVu Sans Mono", transform=ax.transAxes)

    # Image area bounds
    AX_L, AX_R = 0.04, 0.96
    AX_B, AX_T = 0.07, 0.92

    try:
        img = Image.open(img_path)
        # Cap resolution so we don't bloat the PDF
        img.thumbnail((int(PAGE_W * 180), int(PAGE_H * 180)), Image.Resampling.LANCZOS)

        iw, ih  = img.size
        i_asp   = iw / ih                          # image pixel aspect ratio
        a_w_in  = (AX_R - AX_L) * PAGE_W          # available width  in inches
        a_h_in  = (AX_T - AX_B) * PAGE_H          # available height in inches
        a_asp   = a_w_in / a_h_in

        # Scale to fit while preserving aspect ratio
        if i_asp >= a_asp:                         # image wider → fit width
            dw_in = a_w_in
            dh_in = dw_in / i_asp
        else:                                       # image taller → fit height
            dh_in = a_h_in
            dw_in = dh_in * i_asp

        dw_f = dw_in / PAGE_W
        dh_f = dh_in / PAGE_H
        cx   = (AX_L + AX_R) / 2
        cy   = (AX_B + AX_T) / 2

        ax.imshow(
            img,
            extent=(cx - dw_f / 2, cx + dw_f / 2, cy - dh_f / 2, cy + dh_f / 2),
            aspect="auto",
            interpolation="bilinear",
        )
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"[image not found: {img_path.name}]",
                ha="center", va="center", fontsize=10, color="red",
                transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Markdown helpers ───────────────────────────────────────────────────────
def _hlevel(line: str) -> int:
    m = re.match(r"^(#{1,6})\s+", line)
    return len(m.group(1)) if m else 0


def _strip(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*",        r"\1", text)
    text = re.sub(r"\*(.*?)\*",             r"\1", text)
    text = re.sub(r"`([^`]*)`",             r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text, width,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


if __name__ == "__main__":
    main()
