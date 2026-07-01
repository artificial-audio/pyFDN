PAPERS = {
    "Concert_Hall_Impulse_Responses": {
        "title": "legacy.spa.aalto.fi/projects/poririrs",
        "url": "http://legacy.spa.aalto.fi/projects/poririrs/",
    }
}

def paper_link(paper_id: str) -> str:
    paper = PAPERS[paper_id]
    return f"[{paper['title']}]({paper['url']})"