# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
PYTHON        ?= python
SPHINXOPTS    =
SPHINXBUILD   ?= $(PYTHON) -msphinx
SPHINXPROJ    = pyFDN
SOURCEDIR     = .
BUILDDIR      = _build
SPHINXAPIDOC  ?= $(PYTHON) -msphinx.ext.apidoc

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean clean-generated generate-rst html

# API stubs and autosummary pages are reproducible build inputs. Remove them
# before every build so deleted or renamed public objects cannot leave stale
# documentation behind.
clean-generated:
	$(RM) modules.rst pyFDN.rst pyFDN.*.rst
	$(RM) -r generated

generate-rst: clean-generated
	$(SPHINXAPIDOC) -f -o "$(SOURCEDIR)" ../src/pyFDN
	$(PYTHON) patch_rst.py

clean: clean-generated
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: generate-rst
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
