SRCS=$(wildcard *.md)
OBJS=$(SRCS:.md=.pdf)

all: $(OBJS)

%.pdf: %.md
	pandoc -o $@ $^

clean:
	rm -rf *.pdf