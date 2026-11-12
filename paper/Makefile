SECTIONS := $(wildcard sections/*.tex)

main.pdf: main.tex references.bib $(SECTIONS)
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

# Single pass, for fast iteration on prose (cross-refs may be stale).
quick: main.tex $(SECTIONS)
	pdflatex -interaction=nonstopmode main

# Surface unresolved references and overfull boxes after a build.
check:
	@grep -n "LaTeX Warning:" main.log | grep -v "Font\|Rerun\|rerun" || echo "no LaTeX warnings"
	@echo "overfull boxes: $$(grep -c Overfull main.log)"

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc

distclean: clean
	rm -f main.pdf

.PHONY: quick check clean distclean
