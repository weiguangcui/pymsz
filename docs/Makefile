# Minimal makefile for the Sphinx documentation.
#   make html       build docs/_build/html (open _build/html/index.html)
#   make clean      remove the build and the generated API pages
#   make linkcheck  check the external links

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" api/generated reference/_generated

.PHONY: help clean Makefile

# Route all other targets to Sphinx ("make mode").
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
