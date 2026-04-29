CC	= cc
INCS	= -I/usr/local/include -L/usr/local/lib
CFLAGS	= -Ofast -g -Wall -Wextra -pipe -fPIE -fPIC -fopenmp -std=c17 -pedantic $(INCS)
LDLIBS	= -lgmp -lmpfr -lm
EXE	= epoch hex_epoch neodate ackermann \
	  prime fastfib approx circle simpleturtle pi-omp pi-fast us ns notfib \
	  e e-normal e-pipeline logfactorial rand_blast
.PHONY: all

all: ${EXE}

countline:
	wc -l *.c
clean:
	-rm -f ${EXE} *.o
