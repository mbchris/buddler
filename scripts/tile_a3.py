# /// script
# requires-python = ">=3.12"
# dependencies = ["pypdf>=5"]
# ///
"""Enlarge each A4 portrait page of a PDF to ~A3 and tile it onto two A4 landscape sheets.

    uv run scripts/tile_a3.py IN.pdf OUT.pdf [--margins 11,12]

Sheet order: page 1 top, page 1 bottom, page 2 top, ... Print at 100 % ("actual size").
Assembly: cut the top sheet along its dashed line and lay that edge on the dashed line of
the bottom sheet; the strip above that line repeats the top sheet's last millimetres, so the
content lines up. The page's own white margins are cropped, so the content is ~98 % of A3.
"""

import argparse
from pathlib import Path

from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

MM = 72 / 25.4
SHEET_W, SHEET_H = 297 * MM, 210 * MM  # A4 landscape
PRINT_MARGIN = 6 * MM  # stays clear of the printer's unprintable border
OVERLAP = 5 * MM  # source content repeated on the bottom sheet, for alignment


def _pdf_string(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _marks(line_y: float, label: str, label_y: float) -> PageObject:
    """Overlay: a dashed full-width line at `line_y` and a small label (PDF points)."""
    overlay = PageObject.create_blank_page(width=SHEET_W, height=SHEET_H)
    stream = DecodedStreamObject()
    left, right = PRINT_MARGIN, SHEET_W - PRINT_MARGIN
    ops = (
        f"q 0.4 G 0.5 w [4 3] 0 d {left:.2f} {line_y:.2f} m {right:.2f} {line_y:.2f} l S Q "
        f"BT /F1 7 Tf 0.3 g {left + 2 * MM:.2f} {label_y:.2f} Td ({_pdf_string(label)}) Tj ET"
    )
    stream.set_data(ops.encode("ascii"))
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    overlay[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    overlay[NameObject("/Contents")] = stream
    return overlay


def _tile(
    page: PageObject, *, scale: float, tx: float, ty: float, overlay: PageObject
) -> PageObject:
    tile = PageObject.create_blank_page(width=SHEET_W, height=SHEET_H)
    tile.merge_transformed_page(page, Transformation().scale(scale).translate(tx, ty))
    tile.merge_page(overlay)
    return tile


def tile_pdf(src: Path, dst: Path, margin_v: float, margin_h: float) -> int:
    reader = PdfReader(src)
    writer = PdfWriter()
    for number, page in enumerate(reader.pages, start=1):
        page_w, page_h = float(page.mediabox.width), float(page.mediabox.height)
        content_w, content_h = page_w - 2 * margin_h, page_h - 2 * margin_v
        half = content_h / 2
        # Largest scale at which the bottom half plus overlap fits between the print margins.
        scale = min(
            (SHEET_W - 2 * PRINT_MARGIN) / content_w,
            (SHEET_H - 2 * PRINT_MARGIN) / (half + OVERLAP),
        )
        tx = (SHEET_W - content_w * scale) / 2 - margin_h * scale
        # Source y (from the page top, points) placed at the sheet's top print margin.
        for part, src_top in (("top", margin_v), ("bottom", margin_v + half - OVERLAP)):
            ty = SHEET_H - PRINT_MARGIN - (page_h - src_top) * scale
            if part == "top":
                cut_y = SHEET_H - PRINT_MARGIN - half * scale
                overlay = _marks(
                    cut_y, f"Page {number}, top half - cut along the dashed line", cut_y - 10
                )
            else:
                align_y = SHEET_H - PRINT_MARGIN - OVERLAP * scale
                overlay = _marks(
                    align_y,
                    f"Page {number}, bottom half - lay the top half's cut edge on the dashed line",
                    align_y + 4,
                )
            writer.add_page(_tile(page, scale=scale, tx=tx, ty=ty, overlay=overlay))
    with dst.open("wb") as out:
        writer.write(out)
    return len(writer.pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument(
        "--margins",
        default="11,12",
        help="Source page margins to crop, 'vertical,horizontal' in mm (default: 11,12).",
    )
    args = parser.parse_args()
    margin_v, margin_h = (float(value) * MM for value in args.margins.split(","))
    sheets = tile_pdf(args.src, args.dst, margin_v, margin_h)
    print(f"{args.dst}: {sheets} A4 landscape sheets")


if __name__ == "__main__":
    main()
