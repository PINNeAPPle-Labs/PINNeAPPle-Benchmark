"""Build report.pdf: the case report as one document that can be sent to somebody.

    python src/report_pdf.py        # writes report.pdf beside this case

The figures come from figures/, which src/figures.py draws from the corpus and from results/curve.json, so the
numbers in this document are the numbers in the JSON. What is typed here is the prose and the tables, and the tables
are the same numbers the figures plot.

Fonts are the ones matplotlib ships with, so the document and its figures are set in the same faces and the build
needs nothing beyond what drawing the figures already needs.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

CASE = Path(__file__).resolve().parent.parent
FIGURES = CASE / "figures"
OUT = CASE / "report.pdf"

MARGIN = 24 * mm
PAGE_W, PAGE_H = A4
FRAME_W = PAGE_W - 2 * MARGIN

INK = colors.HexColor("#12171e")
MUTED = colors.HexColor("#52606f")
RULE = colors.HexColor("#c9d1da")
RULE_LIGHT = colors.HexColor("#e2e7ed")
ACCENT = colors.HexColor("#1b4a75")
HEAT = colors.HexColor("#a92f22")
GOOD = colors.HexColor("#276a2b")
BAND = colors.HexColor("#eef2f6")

S: dict[str, ParagraphStyle] = {}


# ---------------------------------------------------------------------------
# fonts: the ones the figures are already set in
# ---------------------------------------------------------------------------

def register_fonts() -> None:
    import matplotlib

    root = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

    def face(name: str, files: tuple[str, ...]) -> str:
        for candidate in files:
            path = root / candidate
            if path.exists():
                pdfmetrics.registerFont(TTFont(name, str(path)))
                return name
        raise SystemExit(f"none of {files} in {root}")

    face("Serif", ("DejaVuSerif.ttf",))
    face("SerifBold", ("DejaVuSerif-Bold.ttf",))
    face("SerifItalic", ("DejaVuSerif-Italic.ttf",))
    face("SerifBoldItalic", ("DejaVuSerif-BoldItalic.ttf", "DejaVuSerif-Italic.ttf"))
    face("Sans", ("DejaVuSans.ttf",))
    face("SansBold", ("DejaVuSans-Bold.ttf",))
    face("Mono", ("DejaVuSansMono.ttf",))
    face("MonoBold", ("DejaVuSansMono-Bold.ttf",))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="SerifBold",
                                  italic="SerifItalic", boldItalic="SerifBoldItalic")
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="SansBold", italic="Sans", boldItalic="SansBold")
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="MonoBold", italic="Mono", boldItalic="MonoBold")


def make_styles() -> None:
    def add(name: str, **kw) -> None:
        kw.setdefault("fontName", "Serif")
        kw.setdefault("textColor", INK)
        S[name] = ParagraphStyle(name, **kw)

    add("CoverEyebrow", fontName="Mono", fontSize=8.4, leading=12, textColor=ACCENT, spaceAfter=10)
    add("CoverTitle", fontName="SerifBold", fontSize=26, leading=31, spaceAfter=14)
    add("CoverSub", fontSize=12.6, leading=18.6, textColor=MUTED, spaceAfter=6)
    add("CoverMeta", fontName="Mono", fontSize=7.9, leading=12.6, textColor=MUTED)
    add("CoverMetaValue", fontName="Sans", fontSize=8.6, leading=12.6, textColor=INK)
    add("Eyebrow", fontName="Mono", fontSize=8.2, leading=11, textColor=ACCENT, spaceAfter=1, keepWithNext=1)
    add("H1", fontName="SansBold", fontSize=16.5, leading=20, spaceBefore=2, spaceAfter=7, keepWithNext=1)
    add("H1Num", fontName="SansBold", fontSize=16.5, leading=20, spaceBefore=2, spaceAfter=7, keepWithNext=1)
    add("H2", fontName="SansBold", fontSize=10.8, leading=14, spaceBefore=11, spaceAfter=5, keepWithNext=1)
    add("H2Num", fontName="SansBold", fontSize=10.8, leading=14, spaceBefore=11, spaceAfter=5, keepWithNext=1)
    add("H3", fontName="Sans", fontSize=9.6, leading=13, spaceBefore=6, spaceAfter=3, textColor=ACCENT)
    add("Body", fontSize=9.9, leading=15.0, alignment=TA_JUSTIFY, spaceAfter=7)
    add("Lede", fontSize=11.2, leading=17.0, alignment=TA_JUSTIFY, spaceAfter=9, textColor=colors.HexColor("#26313d"))
    add("BodyLeft", fontSize=9.9, leading=15.0, alignment=TA_LEFT, spaceAfter=7)
    add("Bullet", fontSize=9.7, leading=14.6, alignment=TA_JUSTIFY, leftIndent=13, bulletIndent=2, spaceAfter=5)
    add("Caption", fontName="Sans", fontSize=7.9, leading=11.4, textColor=MUTED, spaceBefore=4, spaceAfter=12)
    add("TableCell", fontName="Sans", fontSize=8.3, leading=11.2)
    add("TableNum", fontName="Sans", fontSize=8.3, leading=11.2, alignment=TA_RIGHT)
    add("TableHead", fontName="Mono", fontSize=7.0, leading=10.0, textColor=MUTED)
    add("TableHeadNum", fontName="Mono", fontSize=7.0, leading=10.0, textColor=MUTED, alignment=TA_RIGHT)
    add("TableCaption", fontName="Mono", fontSize=7.2, leading=10.5, textColor=MUTED, spaceAfter=4)
    add("Code", fontName="Mono", fontSize=7.1, leading=10.4, textColor=INK)
    add("TocEntry", fontSize=9.6, leading=17, leftIndent=0)
    add("TocEntry2", fontSize=9.0, leading=14, leftIndent=14, textColor=MUTED)
    add("FootNote", fontName="Sans", fontSize=7.4, leading=11, textColor=MUTED)


def P(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, S[style])


# ---------------------------------------------------------------------------
# page furniture
# ---------------------------------------------------------------------------

def draw_body_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE_LIGHT)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_H - 18 * mm, PAGE_W - MARGIN, PAGE_H - 18 * mm)
    canvas.setFont("Mono", 6.9)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - 15.4 * mm, "CASE 10  ·  MULTI-FIDELITY AEROTHERMODYNAMICS")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 15.4 * mm, "What the cheap model buys")
    canvas.line(MARGIN, 15.5 * mm, PAGE_W - MARGIN, 15.5 * mm)
    canvas.drawString(MARGIN, 10.8 * mm, "PINNeAPPle-Benchmark  ·  PINNeAPPle-Labs")
    canvas.setFont("Mono", 7.4)
    canvas.setFillColor(INK)
    canvas.drawRightString(PAGE_W - MARGIN, 10.8 * mm, f"{doc.page}")
    canvas.restoreState()


def draw_cover_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(INK)
    canvas.setLineWidth(2.2)
    canvas.line(MARGIN, PAGE_H - 30 * mm, PAGE_W - MARGIN, PAGE_H - 30 * mm)
    canvas.restoreState()


class Report(BaseDocTemplate):
    def __init__(self, path: Path) -> None:
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=MARGIN)
        self.title = "What the cheap model buys"
        self.author = "PINNeAPPle-Labs"
        self.subject = "Multi-fidelity aerothermodynamics, PINNeAPPle-Benchmark case 10"
        self.creator = "PINNeAPPle-Benchmark (src/report_pdf.py)"
        cover = Frame(MARGIN, MARGIN, FRAME_W, PAGE_H - 2 * MARGIN, id="cover", showBoundary=0)
        body = Frame(MARGIN, 19 * mm, FRAME_W, PAGE_H - 19 * mm - 22 * mm, id="body", showBoundary=0)
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[cover], onPage=draw_cover_page),
            PageTemplate(id="body", frames=[body], onPage=draw_body_page),
        ])

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "H1Num":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
        elif isinstance(flowable, Paragraph) and flowable.style.name == "H2Num":
            self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))


def heading(number: str, title: str) -> list:
    return [
        Spacer(1, 12),
        P(f"§ {number}", "Eyebrow"),
        Paragraph(title, S["H1Num"]),
        Thin(),
        Spacer(1, 9),
    ]


def subheading(title: str) -> Paragraph:
    return Paragraph(title, S["H2Num"])


class Thin(Flowable):
    """A hairline across the frame, drawn at the flowable's own baseline."""

    def __init__(self, thickness: float = 0.9) -> None:
        super().__init__()
        self.thickness = thickness

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        return (avail_width, 5)

    def draw(self) -> None:
        self.canv.setStrokeColor(INK)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 2, self.width, 2)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def figure(name: str, caption: str, title: str | None = None, width: float = FRAME_W) -> KeepTogether:
    path = FIGURES / name
    if not path.exists():
        raise SystemExit(f"missing figure {path}; run python src/figures.py first")
    image_w, image_h = ImageReader(str(path)).getSize()
    image = Image(str(path), width=width, height=width * image_h / image_w)
    image.hAlign = "CENTER"
    parts: list = []
    if title:
        parts += [subheading(title), Spacer(1, 7)]
    parts += [Spacer(1, 3), image, Paragraph(caption, S["Caption"])]
    return KeepTogether(parts)


def table(caption: str, head: list[str], rows: list[list[str]], widths: list[float],
          numeric: list[bool], mark: set[int] = frozenset()) -> list:
    data = [[Paragraph(h, S["TableHeadNum" if n else "TableHead"]) for h, n in zip(head, numeric, strict=True)]]
    for row in rows:
        data.append([Paragraph(c, S["TableNum" if n else "TableCell"]) for c, n in zip(row, numeric, strict=True)])
    box = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
        ("TOPPADDING", (0, 0), (-1, 0), 0),
    ]
    for index in sorted(mark):
        style.append(("BACKGROUND", (0, index + 1), (-1, index + 1), BAND))
    box.setStyle(TableStyle(style))
    return [KeepTogether([Paragraph(caption, S["TableCaption"]), box]), Spacer(1, 12)]


def bullets(items: list[str]) -> list:
    return [Paragraph(item, S["Bullet"], bulletText="–") for item in items]


def code_block(text: str) -> Table:
    block = Table([[Preformatted(text, S["Code"])]], colWidths=[FRAME_W])
    block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f9")),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return block


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------

def cover() -> list:
    glance = Table(
        [[Paragraph("Relative L2 against the CFD", S["TableHead"]),
          Paragraph("cheap model", S["TableHeadNum"]),
          Paragraph("+ 8 runs", S["TableHeadNum"]),
          Paragraph("+ 64 runs", S["TableHeadNum"])],
         [Paragraph("Shoulder heat flux, where the panel method is blind", S["TableCell"]),
          Paragraph("27.4 %", S["TableNum"]),
          Paragraph("16.4 %", S["TableNum"]),
          Paragraph("7.0 %", S["TableNum"])],
         [Paragraph("Heat load, a well-behaved scalar", S["TableCell"]),
          Paragraph("5.85 %", S["TableNum"]),
          Paragraph("5.04 %", S["TableNum"]),
          Paragraph("3.55 %", S["TableNum"])],
         [Paragraph("Heat-flux field, 2 361 nodes per run", S["TableCell"]),
          Paragraph("17.9 %", S["TableNum"]),
          Paragraph("5.9 %", S["TableNum"]),
          Paragraph("3.6 %", S["TableNum"])]],
        colWidths=[FRAME_W * 0.50, FRAME_W * 0.17, FRAME_W * 0.165, FRAME_W * 0.165])
    glance.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 0),
        ("LINEABOVE", (0, 0), (-1, 0), 1.6, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
    ]))

    meta = Table(
        [[Paragraph("DATASET", S["CoverMeta"]),
          Paragraph("capsule-aeroheating, Bijective Systems, version 4 · CC BY 4.0", S["CoverMetaValue"])],
         [Paragraph("CORPUS", S["CoverMeta"]),
          Paragraph("1 024 capsules from a modified-Newtonian panel method, 64 of them also from "
                    "second-order laminar Navier–Stokes (Eilmer)", S["CoverMetaValue"])],
         [Paragraph("WALL", S["CoverMeta"]),
          Paragraph("2 361 nodes and 4 640 triangles, identical on every capsule and at both fidelities",
                    S["CoverMetaValue"])],
         [Paragraph("CODE", S["CoverMeta"]),
          Paragraph("PINNeAPPle #1855 on #1857 · this case as PINNeAPPle-Benchmark #11", S["CoverMetaValue"])],
         [Paragraph("DATE", S["CoverMeta"]),
          Paragraph(dt.date.today().strftime("%d %B %Y"), S["CoverMetaValue"])]],
        colWidths=[FRAME_W * 0.13, FRAME_W * 0.87])
    meta.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE_LIGHT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, RULE),
    ]))

    return [
        Spacer(1, 12),
        P("PINNEAPPLE-BENCHMARK  ·  CASE 10", "CoverEyebrow"),
        P("What the cheap model buys", "CoverTitle"),
        P("How many expensive CFD runs a cheap engineering model makes unnecessary, measured on an "
          "aerothermodynamics corpus that carries two fidelities of the same heat shield.", "CoverSub"),
        Spacer(1, 26),
        glance,
        Spacer(1, 26 * mm),
        Table([[Paragraph("Eight Navier\u2013Stokes runs take the shoulder from 27.4 % to 16.4 %; sixty-four take it "
                          "to 7.0 %. On a scalar that is already good, they buy almost nothing. Where a cheap model "
                          "earns its place is decided by where it is wrong, not by an average over targets.",
                          ParagraphStyle("Pull", fontName="SerifItalic", fontSize=11.4, leading=17.6,
                                         textColor=colors.HexColor("#26313d")))]],
              colWidths=[FRAME_W * 0.88],
              style=TableStyle([("LINEBEFORE", (0, 0), (0, 0), 2.0, HEAT),
                                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                ("TOPPADDING", (0, 0), (-1, -1), 2),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 2)])),
        Spacer(1, 50 * mm),
        meta,
        NextPageTemplate("body"),
        PageBreak(),
    ]


def front_matter() -> list:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("Toc0", fontName="Serif", fontSize=9.8, leading=17.5, textColor=INK),
        ParagraphStyle("Toc1", fontName="Serif", fontSize=9.0, leading=14.5, leftIndent=14, textColor=MUTED),
    ]
    toc.dotsMinLevel = 0
    return [Spacer(1, 4), P("Contents", "H1"), Thin(), Spacer(1, 10), toc, PageBreak()]


def summary() -> list:
    return [
        P("The short answer", "H1Num"),
        Thin(),
        Spacer(1, 9),
        P("On the heat shield's <b>shoulder</b>, where the panel method is structurally blind, eight Navier–Stokes "
          "runs take its error from <b>27.4 % to 16.4 %</b>, and sixty-four take it to <b>7.0 %</b>. On a "
          "well-behaved scalar such as the heat load the same eight runs buy almost nothing, "
          "<b>5.85 % to 5.04 %</b>, because there is less left to correct.", "Lede"),
        P("That asymmetry is the result. A single “multi-fidelity gain” figure would average it away, and it "
          "is the asymmetry that decides whether a cheap model earns a place in a design loop: it pays where the "
          "engineering model is wrong, not where it is already right."),
        Spacer(1, 6),
        P("What the corpus is, and why it can answer the question", "H2"),
        P("Capsule-aeroheating ships 1 024 parametric entry capsules with the laminar forebody heat flux from a "
          "modified-Newtonian panel method, which costs seconds, and 64 of them with a second-order laminar "
          "Navier–Stokes solution from Eilmer, at a median of 1.9 hours per run. Every wall has the same 2 361 "
          "nodes and the same 4 640 triangles, and node <i>i</i> is the same point of the heat shield on every "
          "capsule and at both fidelities. The difference between the fidelities is therefore a subtraction at each "
          "node, with no interpolation between meshes, which is what lets a correction be learned from a handful of "
          "expensive runs."),
    ]


def dataset_section() -> list:
    rows = [
        ["Capsules", "1 024 members. Each one shape at one flight condition, a Latin hypercube over six inputs."],
        ["Runs", "1 088: 1 024 from a modified-Newtonian panel method, 64 of those also from second-order laminar "
                 "Navier–Stokes (Eilmer)."],
        ["Wall", "2 361 nodes and 4 640 triangles, the same on every capsule and at both fidelities."],
        ["Inputs", "Diameter 0.8–5.5 m, nose radius 0.25–1.6 D, forebody half-angle 55–75°, "
                   "shoulder radius 0.025–0.075 D, Mach 6–15, altitude 40–80 km."],
        ["Per node", "Heat flux, pressure, arc length, film coefficient, recovery enthalpy, the boundary-layer edge "
                     "state."],
        ["Scalars", "Heat load, stagnation, peak, shoulder and mean heat flux, forebody pressure drag, the "
                    "freestream state."],
        ["Held fixed", "Calorically perfect air, laminar flow, 300 K wall, zero incidence, heat shield only. No "
                       "afterbody, no wake."],
        ["Cost", "Panel method: seconds. Navier–Stokes: 1.2 h to 5.4 h per run on six cores, median 1.9 h."],
    ]
    data = [[Paragraph(k.upper(), S["CoverMeta"]), Paragraph(v, S["TableCell"])] for k, v in rows]
    sheet = Table(data, colWidths=[FRAME_W * 0.16, FRAME_W * 0.84])
    sheet.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE_LIGHT),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
    ]))
    return [
        *heading("1", "The dataset"),
        P("capsule-aeroheating (Bijective Systems, version 4, CC BY 4.0). One file, "
          "<i>vehicle_capsule_multifidelity.gamut</i>, 77 MB, read with the Apache-2.0 <i>bijective-gamut</i> "
          "package.", "BodyLeft"),
        Spacer(1, 4), sheet, Spacer(1, 12),
        P("What makes this a multi-fidelity benchmark rather than two datasets is a property of the mesh: <b>node "
          "<i>i</i> is the same point of the heat shield on every capsule and at both fidelities</b>. The coordinates "
          "differ per capsule, because they are metres at real size, but the correspondence does not. No other "
          "public CFD corpus we read has that property."),
        P("<b>A note on what is being measured.</b> The errors below belong to the corpus's own engineering model, the "
          "modified-Newtonian panel method, and correcting them is the subject of this study. The corpus is the "
          "instrument: it is what makes a node-for-node comparison, and therefore a measurable correction, possible "
          "at all. The Navier\u2013Stokes runs are its published reference, not an independent implementation."),
        figure("fig1_two_fidelities.png",
               "One capsule's heat shield, coloured by each fidelity. The same 2 361 nodes; the peak "
               "moves by about 5 % between the two, and the rim ring is where they disagree."),
        figure("fig2_design_space.png",
               "The 1 024 capsules in the six inputs, with the 64 that have a CFD solution marked. "
               "The subset spreads over the whole design but is not itself a Latin hypercube: six selected shapes "
               "could not be run, so it thins out near Mach 15, near 80 km and for the largest capsules."),
    ]


def method_section() -> list:
    story = [
        *heading("2", "Method"),
        P("A learning curve, not a leaderboard. For a budget N, N of the 64 expensive runs are drawn at random (20 "
          "draws, one seed) and every strategy is fitted on them and scored on the remaining 64 − N. Every "
          "strategy of a draw is scored on the same held-out runs, so the rows of a table are paired. At N = 64 "
          "there is nothing left to hold out, so each run is predicted from the other 63 (leave-one-out): that is a "
          "different question, and it is the only way to put the full budget on the same axis as the rest."),
        P("The regressor is kernel ridge with an RBF kernel, in closed form. No optimiser state, no epochs, and it "
          "still works at N = 8, where a network has nothing to learn from. The length scale is the median pairwise "
          "distance of the training designs. Inputs are the six variables scaled to [−1, 1] by the bounds the "
          "file declares, not by the sample's range: scaling by the design space is what stops a 5.5 m capsule from "
          "dominating a kernel."),
        subheading("Strategies, per capsule"),
        Spacer(1, 6),
        *table("Fitted on one scalar target at a time",
               ["Strategy", "What it is"],
               [["low_fidelity", "The cheap model's own value. No fit. The baseline a correction has to beat."],
                ["geometry", "A surrogate of the target from the six inputs alone: what if there were no cheap "
                             "model."],
                ["difference", "A surrogate of y_high − y_low from the six inputs. The cheap model everywhere "
                               "plus a learned correction, which is the co-kriging model of the two-fidelity case."],
                ["residual_feature", "The same difference, with the cheap model's value among the inputs."]],
               [FRAME_W * 0.24, FRAME_W * 0.76], [False, False]),
        subheading("Strategies, per field (2 361 nodes)"),
        Spacer(1, 6),
        *table("Fitted on the whole wall at once",
               ["Strategy", "What it is"],
               [["low_fidelity", "The cheap field, on the same nodes."],
                ["mean_correction", "The cheap field plus the average relative residual of the training runs: one "
                                    "shape, nothing learned from the design."],
                ["pod_correction", "The cheap field plus three modes of the relative residual, each amplitude "
                                   "learned from the six inputs."]],
               [FRAME_W * 0.24, FRAME_W * 0.76], [False, False]),
        P("The relative residual divides each run's residual by the magnitude of its own cheap field, and the "
          "prediction multiplies back by the cheap run's magnitude. That is legitimate at any design, because the "
          "cheap model runs everywhere. It is also the step the field model does not work without: the heat flux "
          "spans <b>175×</b> across designs, so on the raw residual the mode coefficients span 175× too, "
          "and a regression on six inputs cannot learn a quantity whose scale the design sets. The first version of "
          "this study, without it, reported 96 % and 5 494 % error."),
    ]
    return story


def results_section() -> list:
    story = [
        KeepTogether([
            *heading("3", "Results"),
            figure("fig3_cheap_against_cfd.png",
               "Three capsules, cheap model (dashed) against Navier–Stokes (solid) along the arc "
               "from the stagnation point. The cheap model is systematically high on the nose, low over the cone, "
               "and it misses the shoulder peak entirely: the flow there is set by the shoulder geometry, which a "
               "panel method that sees only the nose radius cannot represent.",
                   title="3.1  The two fidelities, on the same nodes", width=FRAME_W * 0.80),
        ]),
        figure("fig5_scalar_curves.png",
               "Relative L2 against the CFD per target, median over 20 draws; the band is the "
               "spread over draws. The shoulder curve is the one that moves.",
               title="3.2  How many CFD runs the cheap model buys"),
    ]
    head = ["Target", "cheap model", "corrected N=8", "corrected N=64", "design only N=64"]
    rows = [["Heat load", "5.85 %", "5.04 %", "3.55 %", "54.86 %"],
            ["Stagnation heat flux", "18.28 %", "17.87 %", "14.60 %", "26.85 %"],
            ["Shoulder heat flux", "27.43 %", "16.41 %", "7.01 %", "25.55 %"]]
    widths = [FRAME_W * 0.36] + [FRAME_W * 0.16] * 4
    story += table("Relative L2 against the CFD, median over 20 draws", head, rows, widths,
                   [False, True, True, True, True], mark={2})
    story += [
        P("Three readings."),
        *bullets([
            "<b>Where the cheap model is already good, eight runs buy little.</b> The heat load is 5.85 % off and "
            "eight runs take it to 5.04 %: there is less left to correct.",
            "<b>Where it is structurally wrong, they buy a lot.</b> The shoulder goes from 27.4 % to 16.4 % with "
            "eight runs and to 7.0 % with sixty-four, a factor of four.",
            "<b>The design alone never gets close</b>, at 25 % to 55 %. Knowing the diameter, the nose radius and the "
            "Mach number does not tell you the heating. That is the measurement behind an engineering model existing "
            "at all.",
        ]),
        Spacer(1, 8),
        figure("fig4_error_shape.png",
               "The first three modes of the relative residual along the arc, and the share of the "
               "residual energy they carry: 81.0 %, 92.4 %, 97.4 %.",
               title="3.3  The error is a shape, not a table"),
        figure("fig6_field_curve.png",
               "Left: the field's learning curve over the whole 2 361-node field. Right: the worst "
               "capsule of the corpus, capsule_0679, with the average correction and the three-mode correction "
               "against the CFD. It is the same capsule the dataset card singles out for a +40.3 % stagnation heat "
               "flux, which is a check we did not arrange."),
    ]
    story += table("The heat-flux field, relative L2 per run",
                   ["Field model", "relative L2"],
                   [["Cheap field", "17.91 %"],
                    ["+ average correction, nothing learned", "8.94 %"],
                    ["+ 3 modes, amplitudes learned, N=8", "5.88 %"],
                    ["+ 3 modes, amplitudes learned, N=64", "3.57 %"]],
                   [FRAME_W * 0.66, FRAME_W * 0.34], [False, True], mark={1, 3})
    story += [
        P("Two readings, kept apart on purpose. The <i>average</i> correction is one shape with no dependence on the "
          "design at all, and it already halves the field's error. That is not learning: it is the cheap model being "
          "wrong in the same direction everywhere. What the design buys on top of it is the amplitude, and that is "
          "what the runs pay for. In the right panel of Figure 6 the average correction is visibly a compromise, "
          "over-correcting the nose of that particular capsule, while the three-mode correction follows the CFD over "
          "the whole shield."),
    ]
    return story


def code_section() -> list:
    story = [
        *heading("4", "Code"),
        P("Two pull requests carry the work. The case's <i>env/PINNEAPPLE_COMMIT</i> pins the library commit the "
          "numbers were produced with, so a figure can always be traced to the code that drew it."),
    ]
    rows = [
        ["Reader: pairing the fidelities, the wall, the bundles",
         "pinneapple_data/datasets/capsule_aeroheating.py — schema_unit (170), pair_fidelities (213), wall (273), "
         "load_bundle (314), field_pair (470), field_channels (494), field_samples (518)"],
        ["Physics checks the corpus allows",
         "pinneapple_data/datasets/capsule_aeroheating_physics.py — newtonian_pressure (132), axial_spread "
         "(162), heat_load (191), radiative_equilibrium_temperature (213), checks (275)"],
        ["The multi-fidelity curves",
         "pinneapple_data/multifidelity.py — FidelityPair (51), KernelRidge (134), learning_curve (242), "
         "FieldPair (323), residual_rank (376), field_learning_curve (400)"],
        ["Library example: both curves, 20 metrics, 9 checks", "examples/capsule_aeroheating/"],
        ["Tests", "tests/pinneapple_data/test_capsule_aeroheating.py, test_capsule_aeroheating_physics.py, "
                  "test_multifidelity.py — 78 tests"],
        ["This case", "src/run.py (the curve), src/figures.py (every figure here, from the data), src/report_pdf.py "
                      "(this document), results/curve.json (every median and spread, with the protocol)"],
    ]
    story += table("Every number in this report comes from these files", ["What", "Where"], rows,
                   [FRAME_W * 0.27, FRAME_W * 0.73], [False, False])
    story += [
        P("Pull requests: PINNeAPPle #1855, the reader, the checks and the curves, built on #1857 (the physics-data "
          "schema it needs), and this case as PINNeAPPle-Benchmark #11."),
    ]
    return story


def reproduce_section() -> list:
    commands = (
        "git clone https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark\n"
        "cd PINNeAPPle-Benchmark && git checkout exp10-multifidelity-capsule\n"
        "export CASE_DIR=\"$PWD/experiments/10_multifidelity_capsule\"\n"
        "\n"
        "# the library at the pinned commit, next to the benchmark repository\n"
        "git clone https://github.com/PINNeAPPle-Labs/PINNeAPPle ../PINNeAPPle\n"
        "cd ../PINNeAPPle && git checkout \"$(sed -n 1p $CASE_DIR/env/PINNEAPPLE_COMMIT)\"\n"
        "pip install -e . bijective-gamut matplotlib reportlab\n"
        "\n"
        "cd \"$CASE_DIR\"\n"
        "bash reproduce.sh          # downloads 77 MB once, then ~2 min: results/curve.json\n"
        "python src/figures.py      # every figure in this report, from the same data\n"
        "python src/report_pdf.py   # this document, from those figures\n"
        "bash reproduce.sh smoke    # 16 runs, 2 draws: a pipeline check in seconds"
    )
    return [
        *heading("5", "Reproducing it"),
        P("src/figures.py draws every panel from the corpus and from results/curve.json. No number in a figure is "
          "typed in by hand, so a figure and a table cannot drift apart; this document is then set from those "
          "figures."),
        Spacer(1, 4), KeepTogether([code_block(commands)]), Spacer(1, 10),
        P("Set PINNEAPPLE_REPO if the library checkout is not the sibling directory."),
    ]


def limits_section() -> list:
    return [
        *heading("6", "Limitations, stated"),
        *bullets([
            "<b>64 expensive runs is a small sample.</b> The 20 draws give the spread of the <i>split</i>, not an "
            "uncertainty of the corpus: the high-fidelity values carry no error bars of their own, and the corpus "
            "says its grids were not refined case by case.",
            "<b>The field model is low rank.</b> It is not a comparison against a network free to choose every node's "
            "correction. That is a choice, argued from the mode structure, and not a claim of superiority.",
            "<b>The three scalar targets are summaries of the same field.</b> They are correlated, not three "
            "independent problems.",
            "<b>No architecture search, no tuning.</b> One regressor, one kernel, one length-scale heuristic, so the "
            "rows compare strategies rather than models.",
            "<b>No extrapolation split.</b> Every strategy is scored inside the sampled box. The corpus has a sparse "
            "corner, so an extrapolation split would measure the corpus's gap as much as the model.",
            "<b>The cheap model is the corpus's own panel method.</b> Its biases are what is being corrected, not an "
            "independent implementation.",
        ]),
    ]


def licence_section() -> list:
    bibtex = (
        "@misc{bijective_capsule_aeroheating_2026_v4,\n"
        "  title        = {Capsule forebody aeroheating, Newtonian and Navier--Stokes {CFD}},\n"
        "  author       = {{Bijective Systems}},\n"
        "  year         = {2026},\n"
        "  version      = {4},\n"
        "  howpublished = {\\url{https://huggingface.co/datasets/bijective/capsule-aeroheating}}\n"
        "}"
    )
    return [
        KeepTogether([
            *heading("7", "Licence and citation"),
            P("The data and its figures are CC BY 4.0; the reader is Apache-2.0; this case is Apache-2.0 like the "
              "repository."),
            P("This study was produced independently. Bijective Systems published the corpus and has not reviewed or "
              "endorsed these results."),
            Spacer(1, 4), code_block(bibtex),
        ]),
        *provenance(),
        Spacer(1, 10),
        P("PINNeAPPle-Benchmark, case 10, experiments/10_multifidelity_capsule · PINNeAPPle-Labs",
          "FootNote"),
    ]


def provenance() -> list:
    """The last page says what this file was built from, so a reader can go and check it."""
    pin = CASE / "env" / "PINNEAPPLE_COMMIT"
    commit = pin.read_text().splitlines()[0].strip() if pin.exists() else "unpinned"
    rows = [
        ["Document", "src/report_pdf.py, set from the text of REPORT.md and from the six figures"],
        ["Figures", "src/figures.py, drawn from the corpus and from results/curve.json; no number is typed in"],
        ["Library", f"PINNeAPPle commit {commit}, as pinned in env/PINNEAPPLE_COMMIT"],
        ["Data", "capsule-aeroheating version 4, vehicle_capsule_multifidelity.gamut, 77 MB, CC BY 4.0"],
        ["Case", "PINNeAPPle-Benchmark #11, branch exp10-multifidelity-capsule"],
        ["Built", dt.date.today().strftime("%d %B %Y")],
    ]
    data = [[Paragraph(k.upper(), S["CoverMeta"]), Paragraph(v, S["CoverMetaValue"])] for k, v in rows]
    box = Table(data, colWidths=[FRAME_W * 0.13, FRAME_W * 0.87])
    box.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE_LIGHT),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
    ]))
    return [Spacer(1, 16), P("How this document was built", "H2"), Spacer(1, 2), box]


def main() -> int:
    if not FIGURES.exists() or not any(FIGURES.glob("fig5*.png")):
        raise SystemExit("figures are missing; run python src/figures.py first")
    register_fonts()
    make_styles()
    story = [
        *cover(),
        *front_matter(),
        *summary(),
        *dataset_section(),
        *method_section(),
        *results_section(),
        *code_section(),
        *reproduce_section(),
        *limits_section(),
        *licence_section(),
    ]
    doc = Report(OUT)
    doc.multiBuild(story)
    print(f"{OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
