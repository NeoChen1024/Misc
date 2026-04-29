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
#include <getopt.h>
#include <unistd.h>
#include <omp.h>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#ifndef SYS_futex
#ifdef __NR_futex
#define SYS_futex __NR_futex
#endif
#endif
#define HAVE_MXI_BACKEND 1
#else
#define HAVE_MXI_BACKEND 0
#endif

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

typedef enum {
    CALC_BACKEND_LEGACY = 0,
    CALC_BACKEND_MXI = 1,
} calc_backend_t;

#if HAVE_MXI_BACKEND
#define DEFAULT_CALC_BACKEND CALC_BACKEND_MXI
#else
#define DEFAULT_CALC_BACKEND CALC_BACKEND_LEGACY
#endif

/* Global options */
struct {
    word_t terms;
    word_t intensity;
    size_t tile_words;
    int verbose;
    const char *output_file;
    calc_backend_t backend;
} opts = {
    .terms = 5,
    .intensity = 1,
    .tile_words = DEFAULT_TILE_WORDS,
    .verbose = 1,
    .output_file = NULL
    , .backend = DEFAULT_CALC_BACKEND
};

/* Progress tracking */
volatile word_t current_progress = 0;
volatile word_t secs = 0;
volatile word_t end_progress = 0;

static const char *backend_name(calc_backend_t backend)
{
    switch (backend) {
        case CALC_BACKEND_LEGACY:
            return "legacy";
        case CALC_BACKEND_MXI:
            return "mxi";
        default:
            return "unknown";
    }
}

static int backend_supported(calc_backend_t backend)
{
    switch (backend) {
        case CALC_BACKEND_LEGACY:
            return 1;
        case CALC_BACKEND_MXI:
            return HAVE_MXI_BACKEND;
        default:
            return 0;
    }
}

static calc_backend_t parse_backend(const char *value, int *ok)
{
    if (strcmp(value, "legacy") == 0) {
        *ok = 1;
        return CALC_BACKEND_LEGACY;
    }
    if (strcmp(value, "mxi") == 0) {
        *ok = 1;
        return CALC_BACKEND_MXI;
    }
    *ok = 0;
    return CALC_BACKEND_LEGACY;
}

static calc_backend_t resolve_backend(calc_backend_t requested)
{
    if (requested == CALC_BACKEND_MXI && !backend_supported(CALC_BACKEND_MXI)) {
        fprintf(stderr, "mxi backend is unavailable on this platform; falling back to legacy\n");
        return CALC_BACKEND_LEGACY;
    }
    return requested;
}

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
            size_t tile_index = (size_t)t;
            size_t start = tile_index * opts.tile_words;
            size_t tile_end = (tile_index + 1 == num_tiles) ? n : (tile_index + 1) * opts.tile_words;

            if (running_carry != 0) {
                ssize_t i = (ssize_t)tile_end - 1;
                while (1) {
                    dword_t sum = (dword_t)dst[i] + running_carry;
                    dst[i] = (word_t)sum;
                    running_carry = (word_t)(sum >> WORD_SIZE);

                    if (running_carry == 0 || i == (ssize_t)start) {
                        break;
                    }
                    i--;
                }
            }

            running_carry += tile_carries[t];
        }

        fprintf(out, "%019" PRIu64, running_carry);
        /* add newline every 4 groups */
        if((current_progress & 0x3) == 0)
        {
            fputc('\n', out);
        }
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

#if HAVE_MXI_BACKEND
static inline uint64_t computeM_u32(uint32_t d)
{
    return UINT64_C(0xFFFFFFFFFFFFFFFF) / d + 1;
}

static inline uint32_t lfixdiv_2mul(uint32_t * restrict efrac, size_t current, uint64_t reciprocal, uint32_t divisor, uint32_t remainder)
{
    uint64_t tmp_partial_dividend = ((uint64_t)remainder << 32) | efrac[current];
    uint32_t quotient = (uint32_t)(((__uint128_t)reciprocal * tmp_partial_dividend) >> 64);
    efrac[current] = quotient;
    return (uint32_t)(tmp_partial_dividend - (uint64_t)quotient * divisor);
}

static inline void init_mxi_batch(
    uint32_t * restrict divisors,
    uint64_t * restrict reciprocals,
    uint32_t * restrict remainders,
    word_t divisor_start,
    word_t intensity
)
{
    for (size_t i = 0; i < intensity; i++) {
        remainders[i] = 1;
        divisors[i] = (uint32_t)(divisor_start - i);
        reciprocals[i] = computeM_u32(divisors[i]);
    }
}

static void efrac_calc_2mul(
    uint32_t * restrict efrac,
    size_t start,
    size_t end,
    const uint64_t * restrict reciprocals,
    const uint32_t * restrict divisors,
    uint32_t * restrict remainders,
    word_t intensity
)
{
    if (intensity == 0 || reciprocals == NULL || divisors == NULL || end <= start)
        return;

    size_t i;
    for (i = start; i + 1 < end; i += 2)
    {
        remainders[0] = lfixdiv_2mul(efrac, i, reciprocals[0], divisors[0], remainders[0]);
        for (word_t j = 1; j < intensity; j++)
        {
            remainders[j - 1] = lfixdiv_2mul(efrac, i + 1, reciprocals[j - 1], divisors[j - 1], remainders[j - 1]);
            remainders[j + 0] = lfixdiv_2mul(efrac, i + 0, reciprocals[j + 0], divisors[j + 0], remainders[j + 0]);
        }
        remainders[intensity - 1] = lfixdiv_2mul(efrac, i + 1, reciprocals[intensity - 1], divisors[intensity - 1], remainders[intensity - 1]);
    }
    if (i != end) {
        for (word_t j = 0; j < intensity; j++)
        {
            remainders[j] = lfixdiv_2mul(efrac, i, reciprocals[j], divisors[j], remainders[j]);
        }
    }
}

#define SPIN_TIMES (1000)
static inline void wait_pipeline(volatile uint32_t *a, int64_t b)
{
    uint32_t val = *a;
    for (size_t i = SPIN_TIMES; val <= b; i--) {
        if (i == 0) {
            i = SPIN_TIMES;
            syscall(SYS_futex, a, FUTEX_WAIT, val, NULL);
        }
        val = *a;
    }
}

static inline void notify_pipeline(uint32_t * restrict a, volatile uint32_t * restrict b, int64_t offset)
{
    if (*b == (offset + (*a)++))
        syscall(SYS_futex, a, FUTEX_WAKE, 1, NULL);
}

static inline int ecalc_mxi(uint32_t *efrac, size_t limb_count, word_t terms, word_t intensity)
{
    if (terms > UINT32_MAX) {
        return 1;
    }

    size_t maxt = omp_get_max_threads();
    const size_t interthread_buffer = 4;
    int64_t buffer_size = (int64_t)maxt * (int64_t)interthread_buffer;
    uint32_t (*remainders)[intensity] = calloc((size_t)buffer_size, sizeof(*remainders));
    uint64_t (*reciprocals)[intensity] = calloc((size_t)buffer_size, sizeof(*reciprocals));
    uint32_t (*divisors)[intensity] = calloc((size_t)buffer_size, sizeof(*divisors));
    uint32_t *sync = calloc(maxt, sizeof(*sync));

    if (remainders == NULL || reciprocals == NULL || divisors == NULL || sync == NULL) {
        fprintf(stderr, "Error: allocation failed in mxi backend\n");
        free(remainders);
        free(reciprocals);
        free(divisors);
        free(sync);
        exit(1);
    }

    memset(remainders, 0, (size_t)buffer_size * sizeof(*remainders));

    if (maxt == 1 || limb_count < maxt)
    {
        fprintf(stderr, "calculating e with 1 thread (mxi)\n");
        word_t divisor;
        for (divisor = terms; divisor > intensity; divisor -= intensity)
        {
            init_mxi_batch(divisors[0], reciprocals[0], remainders[0], divisor, intensity);
            efrac_calc_2mul(efrac, 0, limb_count, reciprocals[0], divisors[0], remainders[0], intensity);
            current_progress += intensity;
        }

        init_mxi_batch(divisors[0], reciprocals[0], remainders[0], divisor, divisor - 1);
        efrac_calc_2mul(efrac, 0, limb_count, reciprocals[0], divisors[0], remainders[0], divisor - 1);
        current_progress += divisor - 1;
    }
    else
    {
        fprintf(stderr, "calculating e with %zu threads (mxi), intensity = %zu\n", maxt, (size_t)intensity);
        memset(sync, 0, maxt * sizeof(*sync));

        size_t chunk_size = limb_count / maxt;
        #pragma omp parallel default(shared)
        {
            int t = omp_get_thread_num();
            size_t start = (size_t)t * chunk_size;
            size_t end = (size_t)(t + 1) * chunk_size;
            if (t == (int)maxt - 1)
                end = limb_count;
            uint64_t divisor;
            uint64_t buf_idx = 0;

            if (t == 0) {
                for (divisor = terms; divisor > intensity; divisor -= intensity)
                {
                    wait_pipeline(sync + maxt - 1, ((int64_t)sync[t]) - buffer_size);
                    init_mxi_batch(divisors[buf_idx], reciprocals[buf_idx], remainders[buf_idx], divisor, intensity);
                    efrac_calc_2mul(efrac, start, end, reciprocals[buf_idx], divisors[buf_idx], remainders[buf_idx], intensity);
                    buf_idx = ((buf_idx + 1) == (uint64_t)buffer_size) ? 0 : buf_idx + 1;
                    notify_pipeline(sync + t, sync + t + 1, 0);
                }
                init_mxi_batch(divisors[buf_idx], reciprocals[buf_idx], remainders[buf_idx], divisor, divisor - 1);
                wait_pipeline(sync + maxt - 1, (int64_t)sync[t] - buffer_size);
                    efrac_calc_2mul(efrac, start, end, reciprocals[buf_idx], divisors[buf_idx], remainders[buf_idx], divisor - 1);
                notify_pipeline(sync + t, sync + t + 1, 0);
            } else if (t == (int)maxt - 1) {
                for (divisor = terms; divisor > intensity; divisor -= intensity)
                {
                    wait_pipeline(sync + t - 1, sync[t]);
                    efrac_calc_2mul(efrac, start, end, reciprocals[buf_idx], divisors[buf_idx], remainders[buf_idx], intensity);
                    buf_idx = ((buf_idx + 1) == (uint64_t)buffer_size) ? 0 : buf_idx + 1;
                    notify_pipeline(sync + t, sync + 0, buffer_size);
                    #pragma omp atomic
                    current_progress += intensity;
                }
                wait_pipeline(sync + t - 1, sync[t]);
                    efrac_calc_2mul(efrac, start, end, reciprocals[buf_idx], divisors[buf_idx], remainders[buf_idx], divisor - 1);
                notify_pipeline(sync + t, sync + 0, buffer_size);
                #pragma omp atomic
                current_progress += divisor - 1;
            } else {
                for (divisor = terms; divisor > intensity; divisor -= intensity)
                {
                    wait_pipeline(sync + t - 1, sync[t]);
                    efrac_calc_2mul(efrac, start, end, reciprocals[buf_idx], divisors[buf_idx], remainders[buf_idx], intensity);
                    buf_idx = ((buf_idx + 1) == (uint64_t)buffer_size) ? 0 : buf_idx + 1;
                    notify_pipeline(sync + t, sync + t + 1, 0);
                }
                wait_pipeline(sync + t - 1, sync[t]);
                efrac_calc_2mul(efrac, start, end, reciprocals[buf_idx], divisors[buf_idx], remainders[buf_idx], divisor - 1);
                notify_pipeline(sync + t, sync + t + 1, 0);
            }
        }
    }

    free(remainders);
    free(reciprocals);
    free(divisors);
    free(sync);
    return 0;
}

static void pack_u32_fraction(const uint32_t *src, word_t *dst, size_t words)
{
    for (size_t i = 0; i < words; i++) {
        dst[i] = ((word_t)src[i * 2] << 32) | (word_t)src[i * 2 + 1];
    }
}
#endif

void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  -t TERMS      Number of terms (default: 5)\n"
        "  -i INTENSITY  Intensity for core calculation (default: 1)\n"
        "  -T TILE       Tile size in words for decimal conversion (default: 4096)\n"
        "  --impl=NAME   Calculation backend: legacy or mxi (default: %s)\n"
        "  -o FILE       Output to file instead of stdout\n"
        "  -q            Quiet mode (no progress output)\n"
        "  -h            Show this help\n"
        "\n"
        "Examples:\n"
        "  %s -t 1000000 -i 256 -T 4096          # Large computation to stdout\n"
        "  %s -t 100000 -i 64 -o e_100k.txt      # Output to file\n"
        "  %s -t 1000000 -i 256 -q -o e.txt      # Quiet mode, output to file\n",
        prog, backend_name(DEFAULT_CALC_BACKEND), prog, prog, prog
    );
}

int main(int argc, char **argv)
{
    int opt;
    int option_index = 0;
    static const struct option long_options[] = {
        {"impl", required_argument, NULL, 1000},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "t:i:T:o:qh", long_options, &option_index)) != -1) {
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
            case 1000: {
                int ok = 0;
                opts.backend = parse_backend(optarg, &ok);
                if (!ok) {
                    fprintf(stderr, "Error: invalid backend '%s'\n", optarg);
                    return 1;
                }
                break;
            }
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

    opts.backend = resolve_backend(opts.backend);
    fprintf(stderr, "using calculation backend: %s\n", backend_name(opts.backend));

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
    if (opts.backend == CALC_BACKEND_MXI) {
#if HAVE_MXI_BACKEND
        size_t efrac32_size = efrac_size * 2;
        uint32_t *efrac32 = calloc(efrac32_size, sizeof(uint32_t));
        if (efrac32 == NULL) {
            perror("calloc");
            free(efrac);
            return 1;
        }

        if (ecalc_mxi(efrac32, efrac32_size, opts.terms, opts.intensity) == 0) {
            word_t *packed = calloc(efrac_size, sizeof(word_t));
            if (packed == NULL) {
                perror("calloc");
                free(efrac32);
                free(efrac);
                return 1;
            }
            pack_u32_fraction(efrac32, packed, efrac_size);
            free(efrac32);
            free(efrac);
            efrac = packed;
        } else {
            fprintf(stderr, "mxi backend cannot handle this input; falling back to legacy calculation core\n");
            free(efrac32);
            ecalc(efrac, efrac_size, opts.terms, opts.intensity);
        }
#else
        fprintf(stderr, "mxi backend is unavailable on this platform; falling back to legacy calculation core\n");
        ecalc(efrac, efrac_size, opts.terms, opts.intensity);
#endif
    } else {
        ecalc(efrac, efrac_size, opts.terms, opts.intensity);
    }

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
