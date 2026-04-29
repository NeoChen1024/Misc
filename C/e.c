/*
 * ============================================================
 *  Optimized e calculator with tiled decimal conversion
 *  ============================================================
 *
 *  Computes e = 2.71828... to arbitrary precision using a
 *  fixed-point fraction representation of the Taylor series.
 *
 *  Algorithm (Steve Wozniak, 1980):
 *    e = sum_{k=0}^{n} 1/k! is computed as a continued-fraction
 *    style recurrence on a big fixed-point array. Each term
 *    divides the fraction by the divisor and adds the carry.
 *
 *  Credits:
 *    - Original algorithm by Steve Wozniak (1980)
 *    - Parallel pipeline by Neo_Chen
 *    - Futex-based lockless pipeline + 32-bit division
 *      optimization by mxi-box
 *    - Tiled decimal conversion + getopt by Lena Lamport
 *
 *  Compile: gcc -O3 -march=native -fopenmp e.c -o e
 *  Usage:   ./e [-t terms] [-i intensity] [-T tile_size]
 *                [-o file] [--impl=legacy|mxi] [-q] [-h]
 * ============================================================
 */

// =========================== Headers ==========================

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
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

// ====================== Platform Detection ====================

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

// ====================== Type Definitions ======================

#define WORD_SIZE         (64)
#define DEFAULT_TILE_WORDS 4096

typedef uint64_t     word_t;
typedef __uint128_t  dword_t;
typedef uint8_t      byte;

#if __SIZEOF_INT128__ != 16
#error "No 128-bit integer support"
#endif

typedef enum {
    CALC_BACKEND_LEGACY = 0,
    CALC_BACKEND_MXI    = 1,
} calc_backend_t;

#if HAVE_MXI_BACKEND
#define DEFAULT_CALC_BACKEND CALC_BACKEND_MXI
#else
#define DEFAULT_CALC_BACKEND CALC_BACKEND_LEGACY
#endif

// ====================== Global Options ========================

struct {
    word_t        terms;
    word_t        intensity;
    size_t        tile_words;
    int           verbose;
    const char   *output_file;
    calc_backend_t backend;
} opts = {
    .terms       = 5,
    .intensity   = 1,
    .tile_words  = DEFAULT_TILE_WORDS,
    .verbose     = 1,
    .output_file = NULL,
    .backend     = DEFAULT_CALC_BACKEND,
};

// ====================== Progress Tracking =====================

volatile word_t current_progress = 0;
volatile word_t secs             = 0;
volatile word_t end_progress     = 0;

// ====================== Backend Helpers =======================

static const char *backend_name(calc_backend_t backend)
{
    switch (backend) {
        case CALC_BACKEND_LEGACY: return "legacy";
        case CALC_BACKEND_MXI:    return "mxi";
        default:                  return "unknown";
    }
}

static bool backend_supported(calc_backend_t backend)
{
    switch (backend) {
        case CALC_BACKEND_LEGACY: return true;
        case CALC_BACKEND_MXI:    return HAVE_MXI_BACKEND != 0;
        default:                  return false;
    }
}

static calc_backend_t parse_backend(const char *value, bool *ok)
{
    if (strcmp(value, "legacy") == 0) { *ok = true;  return CALC_BACKEND_LEGACY; }
    if (strcmp(value, "mxi")    == 0) { *ok = true;  return CALC_BACKEND_MXI;    }
    *ok = false;
    return CALC_BACKEND_LEGACY;
}

static calc_backend_t resolve_backend(calc_backend_t requested)
{
    if (requested == CALC_BACKEND_MXI && !backend_supported(CALC_BACKEND_MXI)) {
        fprintf(stderr, "mxi backend is unavailable on this platform; "
                        "falling back to legacy\n");
        return CALC_BACKEND_LEGACY;
    }
    return requested;
}

// ============ Precision Estimation & Utilities ================

/*
 * log2(n!) via Stirling's approximation.
 * Used to estimate the number of bits needed for the fraction.
 */
static double log2_factorial(word_t n)
{
    return log2(2 * M_PI) / 2
         + log2(n) * (n + 0.5)
         - n / log(2);
}

/* Estimate how many decimal digits a word-array of size `n` can hold */
static size_t estimate_decimal_digits(size_t n, size_t word_size)
{
    return (size_t)floor(log(2) / log(10) * n * word_size);
}

/* Compute 10^n for small n (n <= 19 so it fits in uint64_t) */
static inline word_t int_pow10(byte n)
{
    word_t r = 1;
    assert(n <= 19);
    for (word_t i = 0; i < n; i++)
        r *= 10;
    return r;
}

#define GROUP_SIZE          (19)   /* digits per group (max for uint64_t) */
const word_t POW10_19 = 10000000000000000000ULL;

// ======== Fixed-Point Arithmetic Primitives (Legacy) ==========

/*
 * Multiply a fixed-point word by 'mul' and add a carry-in.
 * Returns the new carry (high word).
 */
static inline word_t frac_mul(word_t *fraction, word_t mul, word_t carry_in)
{
    dword_t wide = (dword_t)(*fraction) * mul + carry_in;
    *fraction = (word_t)wide;
    return (word_t)(wide >> WORD_SIZE);
}

/*
 * Divide a fixed-point word by 'divisor' (given a remainder from
 * the previous word).  Returns the new remainder.
 */
static inline word_t frac_div(
    word_t *restrict fraction,
    size_t  index,
    word_t  divisor,
    word_t  remainder
) {
    if (divisor <= 1)
        return 0;

    dword_t partial = ((dword_t)remainder << WORD_SIZE) | fraction[index];
    fraction[index]  = (word_t)(partial / divisor);
    return (word_t)(partial % divisor);
}

// ========== Tiled Decimal Conversion ==========================

/*
 * Multiply every word in [start, end) by POW10_19, propagating
 * carry from least-significant to most-significant.
 * Returns the final carry.
 */
static inline word_t process_tile_mul(
    const word_t *restrict input,
    word_t       *restrict output,
    size_t start,
    size_t end,
    word_t carry_in
) {
    word_t carry = carry_in;
    for (ssize_t i = (ssize_t)end - 1; i >= (ssize_t)start; i--) {
        dword_t prod = (dword_t)input[i] * POW10_19 + carry;
        output[i]    = (word_t)prod;
        carry        = (word_t)(prod >> WORD_SIZE);
    }
    return carry;
}

/*
 * Compute the start and end indices for a tile, given the total
 * array length.
 */
static inline void tile_bounds(
    size_t  tile_idx,
    size_t  num_tiles,
    size_t  array_len,
    size_t *start_out,
    size_t *end_out
) {
    *start_out = tile_idx * opts.tile_words;
    *end_out   = (tile_idx + 1 == num_tiles)
                     ? array_len
                     : (tile_idx + 1) * opts.tile_words;
}

/*
 * Propagate a running carry through a contiguous range of a
 * word array (big-endian order).  Stops early if carry
 * disappears or we reach `start`.
 */
static inline void propagate_carry(
    word_t *array,
    size_t  start,
    size_t  end,
    word_t *carry
) {
    if (*carry == 0)
        return;

    ssize_t i = (ssize_t)end - 1;
    while (1) {
        dword_t sum = (dword_t)array[i] + *carry;
        array[i]    = (word_t)sum;
        *carry      = (word_t)(sum >> WORD_SIZE);

        if (*carry == 0 || i == (ssize_t)start)
            break;
        i--;
    }
}

/*
 * Print 'e = 2.' followed by the fractional part, using a
 * tile-based parallel multiplication method.
 *
 * The tiling strategy improves cache locality: the fraction is
 * split into chunks ("tiles") that are multiplied in parallel
 * by POW10_19.  An outer sequential pass then finishes carry
 * propagation across tile boundaries.
 */
static void print_fraction_tiled(
    word_t *fraction,
    size_t  word_count,
    size_t  digit_count,
    FILE   *out
) {
    size_t num_tiles = (word_count + opts.tile_words - 1)
                       / opts.tile_words;
    word_t *tile_carries = calloc(num_tiles, sizeof(word_t));
    word_t *temp         = calloc(word_count, sizeof(word_t));
    word_t *src          = fraction;
    word_t *dst          = temp;

    if (!tile_carries || !temp) {
        fprintf(stderr, "Error: allocation failed in print_fraction_tiled\n");
        exit(1);
    }

    size_t total_groups = digit_count / GROUP_SIZE;
    end_progress = total_groups;
    current_progress = 0;

    fprintf(out, "e = 2.");

    for (size_t group = 0; group < total_groups; group++) {
        /* ---- Phase 1: multiply all tiles in parallel --------- */
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < num_tiles; t++) {
            size_t start, end;
            tile_bounds(t, num_tiles, word_count, &start, &end);
            tile_carries[t] = process_tile_mul(src, dst, start, end, 0);
        }

        /* ---- Phase 2: sequential carry propagation ---------- */
        word_t running_carry = 0;
        for (ssize_t t = (ssize_t)num_tiles - 1; t >= 0; t--) {
            size_t start, end;
            tile_bounds((size_t)t, num_tiles, word_count, &start, &end);
            propagate_carry(dst, start, end, &running_carry);
            running_carry += tile_carries[t];
        }

        fprintf(out, "%019" PRIu64, running_carry);

        /* Insert newline every 4 groups for readability */
        if ((current_progress & 0x3) == 0)
            fputc('\n', out);

        current_progress++;

        /* Swap src/dst for next iteration (ping-pong buffer) */
        word_t *swap = src;
        src = dst;
        dst = swap;

        /* Clear the recycled buffer to avoid stale data */
        memset(dst, 0, word_count * sizeof(word_t));
    }

    /* If result ended up in temp, copy back to fraction */
    if (src != fraction)
        memcpy(fraction, src, word_count * sizeof(word_t));

    /* Handle remaining digits that don't fill a full group */
    size_t remaining = digit_count % GROUP_SIZE;
    if (remaining > 0) {
        word_t carry = 0;
        word_t pow10 = int_pow10((byte)remaining);
        for (ssize_t j = (ssize_t)word_count - 1; j >= 0; j--)
            carry = frac_mul(&fraction[j], pow10, carry);

        char fmt[32];
        snprintf(fmt, sizeof(fmt), "%%0%zu" PRIu64, remaining);
        fprintf(out, fmt, carry);
    }

    fputc('\n', out);

    free(tile_carries);
    free(temp);
}

// ========== Progress Display ==================================

static void progress_report(union sigval sv)
{
    (void)sv;
    static word_t last = 0;

    if (last > end_progress)
        last = 0;

    secs += 1;
    fprintf(stderr,
        ">%7.3f%% (%" PRIu64 "/%" PRIu64 ") @ %zu op/s (%zu op/s avg.)\n",
        (float)current_progress * 100 / end_progress,
        current_progress,
        end_progress,
        current_progress - last,
        current_progress / secs
    );
    last = current_progress;
}

// ================== Pipeline Shift Helper =====================

/*
 * Shift values along a pipeline (right-to-left) and insert a
 * new value at position 0.  This is how each thread receives
 * the "stale" divisor from its neighbour.
 */
static inline void shift_pipeline(
    word_t *pipeline,
    size_t  count,
    word_t  new_value
) {
    for (size_t i = count - 1; i > 0; i--)
        pipeline[i] = pipeline[i - 1];
    pipeline[0] = new_value;
}

// =================== Core e Calculation =======================

/*
 * Compute one pass over the fraction: divide each limb by a
 * contiguous range of divisors tracked by the remainders array.
 */
static inline void fraction_compute_pass(
    word_t *restrict fraction,
    size_t  start,
    size_t  end,
    word_t  divisor,
    word_t *restrict remainders,
    word_t  intensity
) {
    if (divisor <= 1)
        return;

    for (size_t i = start; i < end; i++)
        for (word_t j = 0; j < intensity; j++)
            remainders[j] = frac_div(fraction, i, divisor - j, remainders[j]);
}

/*
 * Perform a single parallelised e-calculation step:
 *   -  initialise remainders for thread 0
 *   -  split the fraction among threads
 *   -  propagate remainders from thread i-1 to i
 */
static inline void parallel_division_pass(
    size_t   fraction_size,
    word_t *restrict fraction,
    word_t  intensity,
    word_t  remainders[][intensity],
    word_t *restrict divisors_pipeline
) {
    for (size_t i = 0; i < intensity; i++)
        remainders[0][i] = 1;

    #pragma omp parallel shared(remainders, fraction, fraction_size,        \
                                divisors_pipeline, intensity)
    {
        int thread     = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        size_t chunk   = fraction_size / (size_t)num_threads;
        size_t start   = (size_t)thread * chunk;
        size_t end     = (size_t)(thread + 1) * chunk;
        if (thread == num_threads - 1)
            end = fraction_size;

        fraction_compute_pass(fraction, start, end,
                              divisors_pipeline[thread],
                              remainders[thread],
                              intensity);
    }

    #pragma omp barrier
    /* Propagate remainders from thread i-1 to thread i */
    for (int i = omp_get_max_threads() - 1; i > 0; i--)
        memcpy(remainders[i], remainders[i - 1],
               sizeof(remainders[0]));
}

/* ---- Legacy backend (generic 64-bit division) -------------- */

static void compute_e_legacy(
    word_t *fraction,
    size_t  fraction_size,
    word_t  terms,
    word_t  intensity
) {
    int num_threads = omp_get_max_threads();
    word_t divisors_pipeline[num_threads];
    word_t remainders[num_threads][intensity];

    memset(divisors_pipeline, 0, sizeof(divisors_pipeline));
    memset(remainders,        0, sizeof(remainders));

    if (fraction_size < (size_t)num_threads) {
        /* Not enough data to parallelise – single thread */
        fprintf(stderr, "calculating e with 1 thread\n");
        for (word_t divisor = terms; divisor >= intensity; divisor -= intensity) {
            for (size_t i = 0; i < intensity; i++)
                remainders[0][i] = 1;
            fraction_compute_pass(fraction, 0, fraction_size,
                                  divisor, remainders[0], intensity);
            current_progress += intensity;
        }

        word_t remaining = terms % intensity;
        for (size_t i = 0; i < intensity; i++)
            remainders[0][i] = 1;
        fraction_compute_pass(fraction, 0, fraction_size,
                              remaining, remainders[0], remaining);
        current_progress += remaining;
    } else {
        fprintf(stderr, "calculating e with %d threads, intensity = %zu\n",
                num_threads, (size_t)intensity);

        /* Main loop: feed divisors through the pipeline */
        for (word_t divisor = terms; divisor >= intensity; divisor -= intensity) {
            shift_pipeline(divisors_pipeline, (size_t)num_threads, divisor);
            parallel_division_pass(fraction_size, fraction, intensity,
                                   remainders, divisors_pipeline);
            current_progress += intensity;
        }

        /* Drain the pipeline (flush stale divisors) */
        for (int j = 0; j < num_threads; j++) {
            shift_pipeline(divisors_pipeline, (size_t)num_threads, 0);
            parallel_division_pass(fraction_size, fraction, intensity,
                                   remainders, divisors_pipeline);
        }

        /* Handle the final partial-intensity batch */
        word_t remaining = terms % intensity;
        if (remaining > 0) {
            shift_pipeline(divisors_pipeline, (size_t)num_threads, remaining);
            parallel_division_pass(fraction_size, fraction, remaining,
                                   remainders, divisors_pipeline);
            current_progress += remaining;

            fprintf(stderr, "Finalizing...\n");
            /* Another drain to finish off */
            for (int j = 0; j < num_threads; j++) {
                shift_pipeline(divisors_pipeline, (size_t)num_threads, 0);
                parallel_division_pass(fraction_size, fraction, remaining,
                                       remainders, divisors_pipeline);
            }
        }
    }
}

// ============= MXI Backend (32-bit, futex-based) ==============

#if HAVE_MXI_BACKEND

/* Compute the "magic" 64-bit reciprocal for a 32-bit divisor */
static inline uint64_t compute_reciprocal_u32(uint32_t d)
{
    return UINT64_MAX / d + 1;
}

/*
 * Divide a 32-bit limb using pre-computed reciprocal.
 * Much faster than hardware division on many CPUs.
 */
static inline uint32_t frac_div_reciprocal(
    uint32_t *restrict fraction,
    size_t   index,
    uint64_t reciprocal,
    uint32_t divisor,
    uint32_t remainder
) {
    uint64_t partial  = ((uint64_t)remainder << 32) | fraction[index];
    uint32_t quotient = (uint32_t)(((__uint128_t)reciprocal * partial) >> 64);
    fraction[index]   = quotient;
    return (uint32_t)(partial - (uint64_t)quotient * divisor);
}

/* Initialise one batch of divisor / reciprocal / remainder arrays */
static inline void init_mxi_batch(
    uint32_t *restrict divisors,
    uint64_t *restrict reciprocals,
    uint32_t *restrict remainders,
    word_t   divisor_start,
    word_t   intensity
) {
    for (size_t i = 0; i < intensity; i++) {
        remainders[i]  = 1;
        divisors[i]    = (uint32_t)(divisor_start - i);
        reciprocals[i] = compute_reciprocal_u32(divisors[i]);
    }
}

/*
 * Fixed-point division pass using 32-bit limbs and
 * reciprocal-based division.  Processes two limbs per
 * iteration for improved instruction-level parallelism.
 */
static void fraction_compute_pass_reciprocal(
    uint32_t *restrict fraction,
    size_t   start,
    size_t   end,
    const uint64_t *restrict reciprocals,
    const uint32_t *restrict divisors,
    uint32_t *restrict remainders,
    word_t   intensity
) {
    if (intensity == 0 || reciprocals == NULL || divisors == NULL || end <= start)
        return;

    /* Process two limbs at a time when possible */
    size_t i;
    for (i = start; i + 1 < end; i += 2) {
        remainders[0] = frac_div_reciprocal(fraction, i,
                                            reciprocals[0], divisors[0],
                                            remainders[0]);
        for (word_t j = 1; j < intensity; j++) {
            remainders[j - 1] = frac_div_reciprocal(fraction, i + 1,
                                                    reciprocals[j - 1],
                                                    divisors[j - 1],
                                                    remainders[j - 1]);
            remainders[j]     = frac_div_reciprocal(fraction, i,
                                                    reciprocals[j],
                                                    divisors[j],
                                                    remainders[j]);
        }
        remainders[intensity - 1] = frac_div_reciprocal(fraction, i + 1,
                                                        reciprocals[intensity - 1],
                                                        divisors[intensity - 1],
                                                        remainders[intensity - 1]);
    }
    /* Handle the last odd limb if any */
    if (i != end) {
        for (word_t j = 0; j < intensity; j++)
            remainders[j] = frac_div_reciprocal(fraction, i,
                                                reciprocals[j], divisors[j],
                                                remainders[j]);
    }
}

/* ---- Futex-based pipeline synchronisation ----------------- */
#define SPIN_TIMES 1000

static inline void pipeline_wait(volatile uint32_t *sync, int64_t expected)
{
    uint32_t val = *sync;
    for (size_t i = SPIN_TIMES; (int64_t)val <= expected; i--) {

        if (i == 0) {
            i = SPIN_TIMES;
            syscall(SYS_futex, sync, FUTEX_WAIT, val, NULL);
        }
        val = *sync;
    }
}

static inline void pipeline_notify(
    uint32_t *restrict sync,
    volatile uint32_t *restrict next_sync,
    int64_t offset
) {
    if (*next_sync == (offset + (uint32_t)(*sync)++))
        syscall(SYS_futex, sync, FUTEX_WAKE, 1, NULL);
}

/* ---- MXI backend entry point ------------------------------ */

/*
 * The MXI backend distributes divisor batches across threads
 * using a futex-based ring pipeline:
 *
 *   Thread 0  –  master: initialises divisors & reciprocals
 *   Thread N-1 –  tail:   commits progress atomically
 *   Threads 1..N-2 – relay: just compute
 *
 * Each thread waits for its predecessor, computes, then notifies
 * its successor.  A ring buffer (interthread_buffer) decouples
 * the threads so bursts don't deadlock.
 */
static int compute_e_mxi(
    uint32_t *fraction,
    size_t    limb_count,
    word_t    terms,
    word_t    intensity
) {
    if (terms > UINT32_MAX) {
        return 1;   /* MXI cannot handle more than 2^32 terms */
    }

    size_t num_threads = (size_t)omp_get_max_threads();
    const size_t INTERBUFFER = 4;   /* ring-buffer depth */
    int64_t buffer_size = (int64_t)num_threads * (int64_t)INTERBUFFER;

    uint32_t (*remainders)[intensity]  = calloc((size_t)buffer_size, sizeof(*remainders));
    uint64_t (*reciprocals)[intensity] = calloc((size_t)buffer_size, sizeof(*reciprocals));
    uint32_t (*divisors)[intensity]    = calloc((size_t)buffer_size, sizeof(*divisors));
    uint32_t *sync                     = calloc(num_threads, sizeof(*sync));

    if (!remainders || !reciprocals || !divisors || !sync) {
        fprintf(stderr, "Error: allocation failed in mxi backend\n");
        free(remainders);
        free(reciprocals);
        free(divisors);
        free(sync);
        exit(1);
    }

    memset(remainders, 0, (size_t)buffer_size * sizeof(*remainders));

    if (num_threads == 1 || limb_count < num_threads) {
        /* Single-thread fallback */
        fprintf(stderr, "calculating e with 1 thread (mxi)\n");
        word_t divisor;

        for (divisor = terms; divisor > intensity; divisor -= intensity) {
            init_mxi_batch(divisors[0], reciprocals[0], remainders[0],
                           divisor, intensity);
            fraction_compute_pass_reciprocal(fraction, 0, limb_count,
                                             reciprocals[0], divisors[0],
                                             remainders[0], intensity);
            current_progress += intensity;
        }

        word_t last_count = divisor - 1;
        if (last_count > 0) {
            init_mxi_batch(divisors[0], reciprocals[0], remainders[0],
                           divisor, last_count);
            fraction_compute_pass_reciprocal(fraction, 0, limb_count,
                                             reciprocals[0], divisors[0],
                                             remainders[0], last_count);
        }
        current_progress += last_count;
    } else {
        fprintf(stderr, "calculating e with %zu threads (mxi), intensity = %zu\n",
                num_threads, (size_t)intensity);
        memset(sync, 0, num_threads * sizeof(*sync));

        size_t chunk_size = limb_count / num_threads;

        #pragma omp parallel default(shared)
        {
            int    thread_id   = omp_get_thread_num();
            size_t start       = (size_t)thread_id * chunk_size;
            size_t end         = (size_t)(thread_id + 1) * chunk_size;
            if (thread_id == (int)num_threads - 1)
                end = limb_count;

            uint64_t divisor;
            uint64_t buf_idx = 0;

            /*
             * Each iteration through the pipeline does:
             *  1. Wait for predecessor  (or, for thread 0, the pipeline slot)
             *  2. Compute (or, for thread 0, init+compute)
             *  3. Notify successor
             *  4. Advance buffer index
             */

            /* ---- Main loop: feed divisors through the pipeline ----- */
            for (divisor = terms; divisor > intensity; divisor -= intensity)
            {
                if (thread_id == 0) {
                    pipeline_wait(sync + num_threads - 1,
                                  (int64_t)sync[thread_id] - buffer_size);
                    init_mxi_batch(divisors[buf_idx], reciprocals[buf_idx],
                                   remainders[buf_idx], divisor, intensity);
                    fraction_compute_pass_reciprocal(
                        fraction, start, end,
                        reciprocals[buf_idx], divisors[buf_idx],
                        remainders[buf_idx], intensity);
                    pipeline_notify(sync + thread_id, sync + thread_id + 1, 0);
                } else if (thread_id == (int)num_threads - 1) {
                    pipeline_wait(sync + thread_id - 1, sync[thread_id]);
                    fraction_compute_pass_reciprocal(
                        fraction, start, end,
                        reciprocals[buf_idx], divisors[buf_idx],
                        remainders[buf_idx], intensity);
                    pipeline_notify(sync + thread_id, sync + 0, buffer_size);
                    #pragma omp atomic
                    current_progress += intensity;
                } else {
                    pipeline_wait(sync + thread_id - 1, sync[thread_id]);
                    fraction_compute_pass_reciprocal(
                        fraction, start, end,
                        reciprocals[buf_idx], divisors[buf_idx],
                        remainders[buf_idx], intensity);
                    pipeline_notify(sync + thread_id, sync + thread_id + 1, 0);
                }
                buf_idx = (buf_idx + 1 == (uint64_t)buffer_size)
                              ? 0 : buf_idx + 1;
            }

            /* ---- Handle the final partial-intensity batch -------- */
            word_t last_count = divisor - 1;
            if (last_count > 0)
            {
                if (thread_id == 0) {
                    pipeline_wait(sync + num_threads - 1,
                                  (int64_t)sync[thread_id] - buffer_size);
                    init_mxi_batch(divisors[buf_idx], reciprocals[buf_idx],
                                   remainders[buf_idx], divisor, last_count);
                    fraction_compute_pass_reciprocal(
                        fraction, start, end,
                        reciprocals[buf_idx], divisors[buf_idx],
                        remainders[buf_idx], last_count);
                    pipeline_notify(sync + thread_id, sync + thread_id + 1, 0);
                } else if (thread_id == (int)num_threads - 1) {
                    pipeline_wait(sync + thread_id - 1, sync[thread_id]);
                    fraction_compute_pass_reciprocal(
                        fraction, start, end,
                        reciprocals[buf_idx], divisors[buf_idx],
                        remainders[buf_idx], last_count);
                    pipeline_notify(sync + thread_id, sync + 0, buffer_size);
                    #pragma omp atomic
                    current_progress += last_count;
                } else {
                    pipeline_wait(sync + thread_id - 1, sync[thread_id]);
                    fraction_compute_pass_reciprocal(
                        fraction, start, end,
                        reciprocals[buf_idx], divisors[buf_idx],
                        remainders[buf_idx], last_count);
                    pipeline_notify(sync + thread_id, sync + thread_id + 1, 0);
                }
                buf_idx = (buf_idx + 1 == (uint64_t)buffer_size)
                              ? 0 : buf_idx + 1;
            }
        }

    }

    free(remainders);
    free(reciprocals);
    free(divisors);
    free(sync);
    return 0;
}

/* Pack a 32-bit fraction array back into 64-bit words */
static void pack_u32_to_u64_fraction(
    const uint32_t *src,
    word_t         *dst,
    size_t          words
) {
    for (size_t i = 0; i < words; i++)
        dst[i] = ((word_t)src[i * 2] << 32) | (word_t)src[i * 2 + 1];
}

#endif  /* HAVE_MXI_BACKEND */

// ====================== Usage / Help =========================

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  -t TERMS      Number of terms (default: 5)\n"
        "  -i INTENSITY  Intensity for core calculation (default: 1)\n"
        "  -T TILE       Tile size in words for decimal conversion "
                          "(default: 4096)\n"
        "  --impl=NAME   Calculation backend: legacy or mxi "
                          "(default: %s)\n"
        "  -o FILE       Output to file instead of stdout\n"
        "  -q            Quiet mode (no progress output)\n"
        "  -h            Show this help\n"
        "\n"
        "Examples:\n"
        "  %s -t 1000000 -i 256 -T 4096        "
          "# Large computation to stdout\n"
        "  %s -t 100000 -i 64 -o e_100k.txt    "
          "# Output to file\n"
        "  %s -t 1000000 -i 256 -q -o e.txt    "
          "# Quiet mode, output to file\n",
        prog,
        backend_name(DEFAULT_CALC_BACKEND),
        prog, prog, prog);
}

// =========================== main =============================

int main(int argc, char **argv)
{
    /* ---- Parse arguments ----------------------------------- */
    int opt;
    int option_index = 0;
    static const struct option long_options[] = {
        {"impl", required_argument, NULL, 1000},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "t:i:T:o:qh",
                              long_options, &option_index)) != -1)
    {
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
            bool ok = false;
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
    fprintf(stderr, "using calculation backend: %s\n",
            backend_name(opts.backend));

    end_progress = opts.terms;

    /* ---- Estimate precision & allocate fraction ------------ */
    double precision = log2_factorial(opts.terms);
    fprintf(stderr,
            "estimated required precision: log2(%" PRIu64 "!) ~= %lf bits\n",
            opts.terms, precision);

    size_t fraction_size = (size_t)ceil(precision / WORD_SIZE);
    word_t *fraction = calloc(fraction_size, sizeof(word_t));
    if (!fraction) {
        perror("calloc");
        return 1;
    }
    fprintf(stderr, "allocated %zu %d-bit words (%zu bit)\n",
            fraction_size, WORD_SIZE, fraction_size * WORD_SIZE);

    size_t digits = estimate_decimal_digits(fraction_size, WORD_SIZE);
    fprintf(stderr, "will print %zu digits\n", digits);

    /* ---- Set up progress timer ----------------------------- */
    timer_t timer;
    struct sigevent ev = {
        .sigev_notify          = SIGEV_THREAD,
        .sigev_notify_function = progress_report,
        .sigev_notify_attributes = NULL
    };
    if (timer_create(CLOCK_MONOTONIC, &ev, &timer) != 0) {
        perror("timer_create");
        free(fraction);
        return 1;
    }

    struct itimerspec period = {
        .it_value.tv_sec  = 1,
        .it_interval.tv_sec  = 1,
        .it_interval.tv_nsec = 0
    };

    current_progress = 0;
    if (opts.verbose)
        timer_settime(timer, TIMER_ABSTIME, &period, NULL);

    /* ---- Compute e ----------------------------------------- */
    if (opts.backend == CALC_BACKEND_MXI) {
#if HAVE_MXI_BACKEND
        size_t frac32_size = fraction_size * 2;
        uint32_t *frac32 = calloc(frac32_size, sizeof(uint32_t));
        if (!frac32) {
            perror("calloc");
            free(fraction);
            return 1;
        }

        if (compute_e_mxi(frac32, frac32_size,
                          opts.terms, opts.intensity) == 0)
        {
            word_t *packed = calloc(fraction_size, sizeof(word_t));
            if (!packed) {
                perror("calloc");
                free(frac32);
                free(fraction);
                return 1;
            }
            pack_u32_to_u64_fraction(frac32, packed, fraction_size);
            free(frac32);
            free(fraction);
            fraction = packed;
        } else {
            fprintf(stderr,
                    "mxi backend cannot handle this input; "
                    "falling back to legacy calculation core\n");
            free(frac32);
            compute_e_legacy(fraction, fraction_size,
                             opts.terms, opts.intensity);
        }
#else
        fprintf(stderr,
                "mxi backend is unavailable on this platform; "
                "falling back to legacy calculation core\n");
        compute_e_legacy(fraction, fraction_size,
                         opts.terms, opts.intensity);
#endif
    } else {
        compute_e_legacy(fraction, fraction_size,
                         opts.terms, opts.intensity);
    }

    fputc('\n', stderr);

    /* ---- Print result -------------------------------------- */
    secs             = 0;
    current_progress = 0;
    fprintf(stderr, "Printing...\n");

    FILE *out = stdout;
    if (opts.output_file) {
        out = fopen(opts.output_file, "w");
        if (!out) {
            perror("fopen");
            free(fraction);
            return 1;
        }
    }

    if (opts.verbose)
        timer_settime(timer, TIMER_ABSTIME, &period, NULL);

    print_fraction_tiled(fraction, fraction_size, digits, out);

    if (opts.output_file) {
        fclose(out);
        fprintf(stderr, "\nOutput written to %s\n", opts.output_file);
    }

    /* ---- Cleanup ------------------------------------------- */
    timer_delete(timer);
    free(fraction);
    return 0;
}
