MDS=$(wildcard *.md)
PDFS=$(MDS:.md=.pdf)

all: $(PDFS)

%.pdf: %.md
	pandoc -o $@ $^

clean:
	rm -rf *.pdf