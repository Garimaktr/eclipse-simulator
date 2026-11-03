CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c99 -pedantic
LIBS = -lglut -lGLU -lGL -lm

eclipse_sim: eclipse_sim.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f eclipse_sim
