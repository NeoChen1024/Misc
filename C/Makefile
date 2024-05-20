CC	= cc
INCS	= -I/usr/local/include -L/usr/local/lib
CFLAGS	= -Ofast -g -Wall -Wextra -pipe -fPIE -fPIC -fopenmp -std=c17 -pedantic $(INCS)
LDLIBS	= -lgmp -lmpfr -lm
EXE	= epoch hex_epoch neodate ackermann \
	  prime fastfib approx circle simpleturtle pi-omp pi-fast us ns notfib \
	  e e-normal e-pipeline logfactorial rand_blast nvidia-set-fanspeed
.PHONY: all

all: ${EXE}

nvidia-set-fanspeed: nvidia-set-fanspeed.c
	${CC} ${CFLAGS} -I/opt/cuda/targets/x86_64-linux/include -o $@ $< /opt/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so

countline:
	wc -l *.c
clean:
	-rm -f ${EXE} *.o
