from pathlib import Path

import bibtexparser

# directory containing this file
HERE = Path(__file__).resolve().parent

# full path to references.bib
BIB_PATH = HERE / "references.bib"

with open(BIB_PATH, encoding="utf-8") as bibfile:
    bib_database = bibtexparser.load(bibfile)

PAPERS = {entry["ID"]: entry for entry in bib_database.entries}


def paper_link(paper_id: str) -> str:
    paper = PAPERS[paper_id]

    author = paper.get("author")
    title = paper.get("title")
    url = paper.get("url")
    year = paper.get("year")

    text = f"{author}, {title}, {year}"

    return f"[{text}]({url})"
