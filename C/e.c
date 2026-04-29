/*
 * Optimized e calculator with tiled decimal conversion and getopt
 * Based on Neo_Chen's original e.c
 *
 * Credits:
 * - Original algorithm by Steve Wozniak (1980)
 * - Parallel pipeline by Neo_Chen
 * - Futex-based lockless pipeline + 32-bit division optimization by mxi-box
 * - Tiled decimal conversion + getopt by Lena Lamport
 *
 * Compile: gcc -O3 -march=native -fopenmp e_optimized.c -o e_optimized
 * Usage: ./e_optimized [-t terms] [-i intensity] [-T tile_size] [-o file] [-q] [-h]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <signal.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#define true	1
#define false	0

#define WORD_SIZE (64)
#define DEFAULT_TILE_WORDS 4096

typedef uint64_t word_t;
#if __SIZEOF_INT128__ != 16
	#error "No 128-bit integer support"
#endif
typedef __uint128_t dword_t;
typedef uint8_t byte;

/* Global options */
struct {
    word_t terms;
    word_t intensity;
    size_t tile_words;
    int verbose;
    const char *output_file;
} opts = {
    .terms = 5,
    .intensity = 1,
    .tile_words = DEFAULT_TILE_WORDS,
    .verbose = 1,
    .output_file = NULL
};

/* Progress tracking */
volatile word_t current_progress = 0;
volatile word_t secs = 0;
volatile word_t end_progress = 0;

void display(union sigval sigval)
{
    (void)sigval;
    static word_t last = 0;
    if(last > end_progress)
        last = 0;

    secs += 1;
    fprintf(stderr, ">%7.3f%% (%" PRIu64 "/%" PRIu64 ") @ %zu op/s (%zu op/s avg.)\n",
        (float)current_progress * 100 / end_progress,
        current_progress,
        end_progress,
        current_progress - last,
        current_progress / secs
    );
    last = current_progress;
}

double log2fractorial(word_t n)
{
    return log2(2 * M_PI)/2 + log2(n) * (n + 0.5) - n / log(2);
}

size_t to_decimal_precision(size_t n, size_t word_size)
{
    const size_t digits = floor(log(2)/log(10) * n * word_size);
    return digits;
}

static inline word_t intpow10(byte n)
{
    word_t r = 1;
    assert(n <= 19);
    for(word_t i = 0; i < n; i++)
        r *= 10;
    return r;
}

#define GROUP_SIZE (19)
const word_t pow10_19 = 10000000000000000000ULL;

static inline word_t lfixmul(word_t *frac, word_t mul, word_t carry_in)
{
    dword_t wide = (dword_t)(*frac) * mul + carry_in;
    (*frac) = (word_t)wide;
    return (word_t)(wide >> WORD_SIZE);
}

/*
 * Tiled decimal conversion - optimized for cache locality
 * Progress tracking included
 */
static inline word_t process_tile(
    const word_t * restrict input,
    word_t * restrict output,
    size_t start, size_t end,
    word_t carry_in
) {
    word_t carry = carry_in;
    for (ssize_t i = (ssize_t)end - 1; i >= (ssize_t)start; i--) {
        dword_t prod = (dword_t)input[i] * pow10_19 + carry;
        output[i] = (word_t)prod;
        carry = (word_t)(prod >> WORD_SIZE);
    }
    return carry;
}

void print_fraction_tiled(word_t *frac, size_t n, size_t digits, FILE *out)
{
    size_t num_tiles = (n + opts.tile_words - 1) / opts.tile_words;
    word_t *tile_carries = calloc(num_tiles, sizeof(word_t));
    word_t *temp = calloc(n, sizeof(word_t));
    word_t *src = frac;
    word_t *dst = temp;

    size_t total_groups = digits / GROUP_SIZE;
    end_progress = total_groups;
    current_progress = 0;

    fprintf(out, "e = 2.");

    for (size_t g = 0; g < total_groups; g++) {
        /* Phase 1: Process all tiles in parallel */
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < num_tiles; t++) {
            size_t start = t * opts.tile_words;
            size_t end = (t + 1 == num_tiles) ? n : (t + 1) * opts.tile_words;
            tile_carries[t] = process_tile(src, dst, start, end, 0);
        }

        /* Phase 2: Sequential carry propagation between tiles */
        word_t running_carry = 0;
        for (ssize_t t = (ssize_t)num_tiles - 1; t >= 0; t--) {
            size_t start = t * opts.tile_words;
            size_t tile_end = (t + 1 == num_tiles) ? n : (t + 1) * opts.tile_words;
            
            dword_t sum = (dword_t)dst[start] + running_carry;
            dst[start] = (word_t)sum;
            running_carry = (word_t)(sum >> WORD_SIZE);

            for (size_t i = start + 1;
                 i < tile_end && running_carry;
                 i++) {
                dword_t sum = (dword_t)dst[i] + running_carry;
                dst[i] = (word_t)sum;
                running_carry = (word_t)(sum >> WORD_SIZE);
            }

            running_carry += tile_carries[t];
        }

        fprintf(out, "%019" PRIu64, running_carry);
        current_progress++;

        /* Swap src/dst for next iteration */
        word_t *swap = src;
        src = dst;
        dst = swap;
        
        /* Clear dst before next use to avoid stale data */
        memset(dst, 0, n * sizeof(word_t));
    }

    /* If result ended up in temp, copy back */
    if (src != frac) {
        memcpy(frac, src, n * sizeof(word_t));
    }

    /* Handle remaining digits */
    size_t rest = digits % GROUP_SIZE;
    if (rest > 0) {
        word_t carry = 0;
        word_t pow10 = intpow10(rest);
        for (ssize_t j = n - 1; j >= 0; j--) {
            carry = lfixmul(&frac[j], pow10, carry);
        }
        char fmtspec[32];
        sprintf(fmtspec, "%%0%zu" PRIu64, rest);
        fprintf(out, fmtspec, carry);
    }

    fputc('\n', out);

    free(tile_carries);
    free(temp);
}

/* Core e calculation - unchanged from original */
static inline word_t lfixdiv(word_t * restrict efrac, size_t current, word_t divisor, word_t remainder)
{
    if(divisor <= 1)
        return 0;

    dword_t tmp_partial_dividend = ((dword_t)remainder << WORD_SIZE) | efrac[current];
    efrac[current] = tmp_partial_dividend / divisor;
    return tmp_partial_dividend % divisor;
}

static inline void efrac_calc(word_t * restrict efrac, size_t start, size_t end, word_t divisor, word_t * restrict remainders, word_t intensity)
{
    if(divisor <= 1)
        return;
    for(size_t i = start; i < end; i++)
    {
        for(word_t j = 0; j < intensity; j++)
        {
            remainders[j] = lfixdiv(efrac, i, divisor - j, remainders[j]);
        }
    }
}

static inline void ecalc_parallel(size_t efrac_size, word_t * restrict efrac, const word_t intensity, word_t remainders[][intensity], word_t * restrict divisors_pipeline)
{
    for(size_t i = 0; i < intensity; i++)
    {
        remainders[0][i] = 1;
    }

    #pragma omp parallel shared(remainders, efrac, efrac_size, divisors_pipeline, intensity)
    {
        int t = omp_get_thread_num();
        size_t chunk_size = efrac_size / omp_get_num_threads();
        size_t start = t * chunk_size;
        size_t end = (t + 1) * chunk_size;
        if(t == omp_get_num_threads() - 1)
            end = efrac_size;
        efrac_calc(efrac, start, end, divisors_pipeline[t], remainders[t], intensity);
    }

    #pragma omp barrier
    for(int i = omp_get_max_threads() - 1; i > 0; i--)
    {
        memcpy(remainders[i], remainders[i - 1], sizeof(remainders[0]));
    }
}

static inline void ecalc(word_t *efrac, size_t efrac_size, word_t terms, word_t intensity)
{
    int maxt = omp_get_max_threads();
    word_t divisors_pipeline[maxt];
    word_t remainders[maxt][intensity];

    for(int i = 0; i < maxt; i++)
        divisors_pipeline[i] = 0;

    memset(remainders, 0, sizeof(remainders));

    if(efrac_size < (size_t)maxt)
    {
        fprintf(stderr, "calculating e with 1 thread\n");
        for(word_t divisor = terms; divisor >= intensity; divisor -= intensity)
        {
            for(size_t i = 0; i < intensity; i++)
            {
                remainders[0][i] = 1;
            }
            efrac_calc(efrac, 0, efrac_size, divisor, remainders[0], intensity);
            current_progress += intensity;
        }

        word_t remaining_divisor = terms % intensity;

        for(size_t i = 0; i < intensity; i++)
        {
            remainders[0][i] = 1;
        }
        efrac_calc(efrac, 0, efrac_size, remaining_divisor, remainders[0], remaining_divisor);
        current_progress += remaining_divisor;

    }
    else
    {
        fprintf(stderr, "calculating e with %d threads, intensity = %zd\n", maxt, intensity);
        for(word_t divisor = terms; divisor >= intensity; divisor -= intensity)
        {
            for(size_t i = maxt - 1; i > 0; i--)
                divisors_pipeline[i] = divisors_pipeline[i - 1];
            divisors_pipeline[0] = divisor;

            ecalc_parallel(efrac_size, efrac, intensity, remainders, divisors_pipeline);
            current_progress += intensity;
        }

        for(int i = 0; i < maxt; i++)
        {
            for(int i = maxt - 1; i > 0; i--)
                divisors_pipeline[i] = divisors_pipeline[i - 1];
            divisors_pipeline[0] = 0;
            ecalc_parallel(efrac_size, efrac, intensity, remainders, divisors_pipeline);
        }

        word_t remaining_intensity = terms % intensity;
        for(size_t i = maxt - 1; i > 0; i--)
            divisors_pipeline[i] = divisors_pipeline[i - 1];
        divisors_pipeline[0] = remaining_intensity;

        ecalc_parallel(efrac_size, efrac, remaining_intensity, remainders, divisors_pipeline);
        current_progress += remaining_intensity;

        fprintf(stderr, "Finalizing...\n");
        for(int i = 0; i < maxt; i++)
        {
            for(int i = maxt - 1; i > 0; i--)
                divisors_pipeline[i] = divisors_pipeline[i - 1];
            divisors_pipeline[0] = 0;
            ecalc_parallel(efrac_size, efrac, remaining_intensity, remainders, divisors_pipeline);
        }

    }
}

void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  -t TERMS      Number of terms (default: 5)\n"
        "  -i INTENSITY  Intensity for core calculation (default: 1)\n"
        "  -T TILE       Tile size in words for decimal conversion (default: 4096)\n"
        "  -o FILE       Output to file instead of stdout\n"
        "  -q            Quiet mode (no progress output)\n"
        "  -h            Show this help\n"
        "\n"
        "Examples:\n"
        "  %s -t 1000000 -i 256 -T 4096          # Large computation to stdout\n"
        "  %s -t 100000 -i 64 -o e_100k.txt      # Output to file\n"
        "  %s -t 1000000 -i 256 -q -o e.txt      # Quiet mode, output to file\n",
        prog, prog, prog, prog
    );
}

int main(int argc, char **argv)
{
    int opt;

    while ((opt = getopt(argc, argv, "t:i:T:o:qh")) != -1) {
        switch (opt) {
            case 't':
                opts.terms = strtoull(optarg, NULL, 10);
                break;
            case 'i':
                opts.intensity = strtoull(optarg, NULL, 10);
                break;
            case 'T':
                opts.tile_words = strtoull(optarg, NULL, 10);
                break;
            case 'o':
                opts.output_file = optarg;
                break;
            case 'q':
                opts.verbose = 0;
                break;
            case 'h':
                usage(argv[0]);
                return 0;
            default:
                usage(argv[0]);
                return 1;
        }
    }

    if (opts.terms == 0) {
        fprintf(stderr, "Error: Invalid term count\n");
        return 1;
    }

    if (opts.intensity == 0 || opts.intensity > opts.terms) {
        fprintf(stderr, "Error: Invalid intensity\n");
        return 1;
    }

    end_progress = opts.terms;

    /* Estimate required precision */
    double precision = log2fractorial(opts.terms);
    fprintf(stderr, "estimated required precision: log2(%" PRIu64 "!) ~= %lf bits\n",
            opts.terms, precision);

    size_t efrac_size = ceil(precision / WORD_SIZE);
    word_t *efrac = calloc(efrac_size, sizeof(word_t));
    fprintf(stderr, "allocated %zd %dbit words (%zu bit)\n",
            efrac_size, WORD_SIZE, efrac_size * WORD_SIZE);

    size_t digits = to_decimal_precision(efrac_size, WORD_SIZE);
    fprintf(stderr, "will print %zu digits\n", digits);

    /* Set up timer for progress display */
    timer_t timer;
    struct sigevent ev = {
        .sigev_notify = SIGEV_THREAD,
        .sigev_notify_function = display,
        .sigev_notify_attributes = NULL
    };
    timer_create(CLOCK_MONOTONIC, &ev, &timer);

    struct itimerspec period = {
        .it_value.tv_sec = 1,
        .it_interval.tv_sec = 1,
        .it_interval.tv_nsec = 0
    };

    current_progress = 0;
    if (opts.verbose) {
        timer_settime(timer, TIMER_ABSTIME, &period, NULL);
    }

    /* Calculate e */
    ecalc(efrac, efrac_size, opts.terms, opts.intensity);

    putc('\n', stderr);

    /* Print the result */
    secs = 0;
    current_progress = 0;

    fprintf(stderr, "Printing...\n");
    
    /* Set up output stream */
    FILE *out = stdout;
    if (opts.output_file != NULL) {
        out = fopen(opts.output_file, "w");
        if (out == NULL) {
            perror("fopen");
            free(efrac);
            return 1;
        }
    }
    
    /* Set up progress timer for printing */
    if (opts.verbose) {
        timer_settime(timer, TIMER_ABSTIME, &period, NULL);
    }
    
    print_fraction_tiled(efrac, efrac_size, digits, out);
    
    if (opts.output_file != NULL) {
        fclose(out);
        fprintf(stderr, "\nOutput written to %s\n", opts.output_file);
    }

    timer_delete(timer);
    free(efrac);
    return 0;
}
