# Build and publish the DAFx 2026 tutorial. See README.md.
#
# `publish` is the one that matters: the Sphinx docs build only copies
# docs/_static, it never runs Quarto, so the committed copy under
# docs/_static/tutorials/dafx2026/ is what the website actually serves.

TUTORIAL_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT    := $(abspath $(TUTORIAL_DIR)../../..)
PUBLISH_DIR  := $(REPO_ROOT)/docs/_static/tutorials/dafx2026

.PHONY: help figures render publish preview standalone clean

help:
	@echo "figures     regenerate figures/out/*.svg from the current pyFDN"
	@echo "render      quarto render into ./_site"
	@echo "publish     render, then sync ./_site into docs/_static (commit this)"
	@echo "preview     quarto preview the slides with live reload"
	@echo "standalone  single-file deck for presenting offline (not committed)"
	@echo "clean       remove ./_site"

figures:
	python $(REPO_ROOT)/docs/tutorials/dafx2026/figures/make_figures.py

render:
	quarto render

# --exclude='*.map': reveal.js ships 832 KB of source maps for debugging
# reveal.js itself. Browsers only fetch them with devtools open, and a 404 there
# is harmless. Excluding them here (rather than in .gitignore) keeps the
# committed tree byte-identical to what this target produces.
publish: render
	mkdir -p "$(PUBLISH_DIR)"
	rsync -a --delete --delete-excluded --exclude='*.map' _site/ "$(PUBLISH_DIR)/"
	python sources_digest.py > "$(PUBLISH_DIR)/sources.sha256"
	@echo "published to $(PUBLISH_DIR)"

preview:
	quarto preview slides.qmd

standalone:
	quarto render slides.qmd --to revealjs -M embed-resources:true -o slides-standalone.html

clean:
	rm -rf _site
