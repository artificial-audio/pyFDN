# DAFx 2026 tutorial — source

Quarto sources for the 90-minute pyFDN tutorial at DAFx 2026 (Cambridge, MA,
1–4 September 2026).

```
_quarto.yml               project config; output goes to docs/_static/tutorials/dafx2026/
index.qmd                 participant landing page (setup, schedule, notebook list)
slides.qmd                the reveal.js deck
theme/pyfdn-slides.scss   deck styling (pyFDN logo palette)
theme/pyfdn-page.scss     landing-page styling
figures/make_figures.py   regenerates every figure in the deck
figures/out/*.svg         the figures, checked in
```

## Rendering

Requires [Quarto](https://quarto.org) (developed against 1.8). No Python is
needed to render — see below.

```bash
cd docs/tutorials/dafx2026
quarto render                 # both pages → docs/_static/tutorials/dafx2026/
quarto preview slides.qmd     # live reload while editing
```

Use `make publish`, not bare `quarto render`: it renders into `_site/`, syncs that
into `docs/_static/tutorials/dafx2026/`, and records a digest of the sources so a
forgotten publish fails CI (see below).

## Why the rendered output is committed

Neither docs build runs Quarto. `.github/workflows/docs.yml` (→ GitHub Pages) and
`.readthedocs.yaml` both just run Sphinx, which *copies* `docs/_static` verbatim.
So the committed HTML is what the website serves.

Rendering in CI instead was considered and rejected:

- **It defeats the freeze.** The whole point of this deck is to be a snapshot of
  what was presented. A frozen historical document re-rendered on every push by
  whatever Quarto version CI resolves is not frozen.
- **There are two build systems, not one.** Quarto would have to be wired into
  the GitHub Actions workflow *and* Read the Docs — and RTD can't do it with
  `apt_packages`, since Quarto isn't in Ubuntu's repos, so it needs a hand-rolled
  `build.jobs` download step.
- **It adds a failure mode to the workflow that deploys the site.** Today a
  Quarto bug or upgrade cannot break the docs deploy.

The cost is ~6 MB in the repository, of which ~5 MB is the reveal.js runtime that
only changes when Quarto is upgraded. Day-to-day edits rewrite ~100 KB of HTML.

The real risk of committing is staleness — editing a slide and forgetting to
publish. `make publish` writes `sources.sha256` next to the output, and
`tests/test_tutorial_dafx2026.py::test_published_output_is_not_stale` recomputes
it from `sources_digest.py`, so that mistake fails CI instead of silently
shipping the old deck. It is content-based, not mtime-based, because a git
checkout does not preserve modification times.

To regenerate the figures after a pyFDN change:

```bash
python docs/tutorials/dafx2026/figures/make_figures.py           # all
python docs/tutorials/dafx2026/figures/make_figures.py poles ir  # a subset
```

For a single-file copy to present from offline (a USB stick, a borrowed laptop):

```bash
quarto render slides.qmd --to revealjs -M embed-resources:true -o slides-standalone.html
```

That one is deliberately **not** committed — it is ~2 MB and would be rewritten
on every edit.

## Why no executed code?

`_quarto.yml` sets `execute: enabled: false`. Every code listing in the deck is
display-only, and every figure is a pre-rendered SVG.

This is the answer to the obvious risk with a tutorial about a 0.x toolbox: the
deck is a snapshot of an API that will keep moving. If the slides executed
`pyFDN` at render time, a signature change a year from now would either break
the render or silently change the figures. As written, the deck renders from
Markdown alone — forever, on any machine, with no Python environment at all.

The cost is that a listing can go stale without anything failing loudly, so:

- `tests/test_tutorial_dafx2026.py` checks that every `pyFDN.<name>` mentioned
  in the deck still exists in the public API. A rename breaks CI, with the
  slide's own text in the failure message.
- The figures stay reproducible: `make_figures.py` is real code against the real
  API, and it runs in CI for the same reason.

Neither guard catches a changed *argument* name. That is accepted — see freezing.

## Freezing after the conference

Once the tutorial has been given, the deck stops being a living document and
becomes a record of what was said. At that point:

1. `make figures && make publish` one final time.
2. Note the pyFDN version the deck was built against, on the landing page.
3. Tag the repository, e.g. `git tag dafx2026-tutorial`.
4. Delete `tests/test_tutorial_dafx2026.py`, so later API changes are free to
   move on without breaking CI for a frozen document.

Step 4 is the point of the whole arrangement: the drift guards are useful while
the deck is live and actively wrong to keep once it is frozen — including the
staleness check, which would otherwise demand a re-render (with a future Quarto)
of a document that is supposed to be fixed. The rendered HTML in
`docs/_static/tutorials/dafx2026/` keeps working either way, because it depends
on nothing.

## Conventions in the deck

Three custom divs carry meaning, styled in `theme/pyfdn-slides.scss`:

| Div | Renders as | Use for |
| --- | --- | --- |
| `::: {.live}` | orange **LIVE** callout | switch to marimo, a notebook, or the website |
| `::: {.exercise}` | green **YOUR TURN** callout | something the room does, not the presenter |
| `::: {.todo}` | red **DRAFT** callout | decisions still open; must be empty before September |

Speaker notes go in `::: notes` blocks and are visible with `S` in the browser.
They carry the timing plan and what to actually say, so the deck is presentable
by either author.

Grep for the open decisions:

```bash
grep -n 'todo}' slides.qmd
```
