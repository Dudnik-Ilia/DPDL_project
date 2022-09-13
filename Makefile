MD = $(wildcard *.md)
PDF = $(MD:%.md=%.pdf)

.PHONY: all clean

all: $(PDF)

%.pdf: %.md
	pandoc --pdf-engine=xelatex -o $@ $<

$(PDF): $(MD)

clean:
	rm -f $(PDF)

 
