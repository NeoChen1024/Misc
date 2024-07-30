/* OpenMP program to generate a random file or crack the random seed of a random file */

#include <bits/types/timer_t.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <signal.h>
#include <omp.h>

#define MEGA (1024ULL * 1024)

#ifndef NO_EXPECT
#  define likely(x)	__builtin_expect(!!(x), 1)
#  define unlikely(x)	__builtin_expect(!!(x), 0)
#else
#  define likely(x)	(x)
#  define unlikely(x)	(x)
#endif

// Linear congruential generator used by glibc
// substitute with target's RNG if known
static inline unsigned int weak_rand_r(unsigned int *seed)
{
	*seed = *seed * 1103515245 + 12345;
	return (unsigned int)(*seed / 65536) % 32768;
}

static inline int match(unsigned int seed, const uint8_t * restrict bin, const size_t len)
{
	for (size_t i = 0; i < len; i++)
	{
		if (likely(weak_rand_r(&seed) % 256 != bin[i]))
			return 0;
	}
	return len;
}

static volatile unsigned long long ctr = 0;
static volatile unsigned long long last = 0;
static volatile unsigned long long secs = 0;

void display(union sigval sigval)
{
	// silence the compiler
	(void)sigval;

	secs += 1;
	fprintf(stderr, ">%7.3f%% (%llu/%u) @ %.5f Mop/s\n",
	ctr / (double)UINT_MAX * 100, ctr, UINT_MAX, ((double)(ctr - last) / 1000.0 / 1000.0));
	
	last = ctr;
}

int main(int argc, char *argv[])
{
	uint8_t *bin = NULL;
	FILE *fp;

	if(argc < 2)
	{
		printf("Usage: %s <file> [seed]\n", argv[0]);
		return 1;
	}

	// Generate random file
	if(argc == 3)
	{
		unsigned int seed = atol(argv[2]);
		unsigned int temp = seed;

		bin = malloc(sizeof(char) * MEGA);
		for(size_t i = 0; i < MEGA; i++)
			bin[i] = weak_rand_r(&temp) % 256;
		fp = fopen(argv[1], "wb");
		if(fp == NULL)
		{
			printf("Error: could not open file %s\n", argv[2]);
			return 1;
		}
		fwrite(bin, sizeof(char), MEGA, fp);
		printf("File %s generated with seed %u\n", argv[1], seed);
		return 0;
	}

	if(strcmp(argv[1], "-") == 0)
	{
		fp = stdin;
	}
	else
	{
		fp = fopen(argv[1], "rb");
		if(fp == NULL)
		{
			printf("Error: could not open file %s\n", argv[1]);
			return 1;
		}
	}

	// read file
	bin = malloc(sizeof(char) * MEGA);
	if(bin == NULL)
	{
		printf("Error: could not allocate memory\n");
		return 1;
	}
	size_t len = fread(bin, sizeof(char), MEGA, fp);
	if(len == 0)
	{
		printf("Error: could not read file\n");
		return 1;
	}
	
	// set-up timer for progress display
	timer_t timer;
	struct sigevent ev =
	{
		.sigev_notify = SIGEV_THREAD,
		.sigev_notify_function = display,
		.sigev_notify_attributes = NULL
	};
	timer_create(CLOCK_MONOTONIC, &ev, &timer);

	struct itimerspec period =
	{
		.it_value.tv_sec=1,
		.it_interval.tv_sec=1,
		.it_interval.tv_nsec=000000000L
	};

	ctr = 0;
	timer_settime(timer, TIMER_ABSTIME, &period, NULL);

	#pragma omp parallel for shared(bin, len)
	for(int64_t i = 0; i < (1ULL<<32); i+= (1<<16))
	{
		for(int64_t j = 0; j < (1<<16); j++)
		{
			if(unlikely(match(i + j, bin, len)))
			{
				printf("Seed found: %u\n", (unsigned int)i);
				// exit(0);
			}
		}
		#pragma omp atomic
		ctr += (1<<16);
	}
	printf("Seed not found\n");
	timer_delete(timer);
	return 0;
}
