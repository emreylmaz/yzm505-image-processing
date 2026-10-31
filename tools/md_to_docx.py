"""pandoc yoksa yedek: basit Markdown → .docx (başlık, paragraf, liste, kod, tablo, görsel)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt


def convert(md_path: Path, out_path: Path) -> None:
    doc = Document()
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            p = doc.add_paragraph()
            run = p.add_run("\n".join(lines[i + 1 : j]))
            run.font.name, run.font.size = "Consolas", Pt(8)
            i = j + 1
            continue
        if m := re.match(r"^(#{1,4})\s+(.*)", line):
            doc.add_heading(m.group(2), level=len(m.group(1)))
        elif m := re.match(r"^!\[(.*?)\]\((.*?)\)", line):
            img = (md_path.parent / m.group(2)).resolve()
            if img.is_file():
                doc.add_picture(str(img), width=Cm(16))
            doc.add_paragraph(m.group(1)).italic = True
        elif (
            line.startswith("|")
            and i + 1 < len(lines)
            and set(lines[i + 1].replace("|", "").strip()) <= set("-: ")
        ):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            rows.pop(1)
            t = doc.add_table(rows=len(rows), cols=len(rows[0]))
            t.style = "Table Grid"
            for r, row in enumerate(rows):
                for c, cell in enumerate(row[: len(rows[0])]):
                    t.cell(r, c).text = cell.replace("**", "")
            continue
        elif re.match(r"^\s*[-*]\s+", line):
            doc.add_paragraph(
                re.sub(r"^\s*[-*]\s+", "", line).replace("**", ""), style="List Bullet"
            )
        elif line.strip():
            doc.add_paragraph(line.replace("**", ""))
        i += 1
    doc.save(str(out_path))


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"report.docx (python-docx) yazıldı: {sys.argv[2]}")
