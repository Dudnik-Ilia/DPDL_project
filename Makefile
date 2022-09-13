MD = $(wildcard *.md)
PDF = $(MD:%.md=%.pdf)

.PHONY: all clean

all: $(PDF)

%.pdf: %.md
	pandoc -o $@ $<

$(PDF): $(MD)

clean:
	rm -f $(PDF)

 