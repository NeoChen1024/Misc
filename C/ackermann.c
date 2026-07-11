#include <errno.h>
#include <gmp.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
	mpz_t *values;
	size_t size;
	size_t capacity;
} virtual_stack;

static atomic_ullong count = 0;
static clock_t start;

static void die(const char *message)
{
	perror(message);
	exit(EXIT_FAILURE);
}

static void stack_init(virtual_stack *stack)
{
	stack->values = NULL;
	stack->size = 0;
	stack->capacity = 0;
}

static void stack_grow(virtual_stack *stack)
{
	size_t old_capacity = stack->capacity;
	size_t new_capacity = old_capacity == 0 ? 16 : old_capacity * 2;

	if (new_capacity < old_capacity ||
	    new_capacity > SIZE_MAX / sizeof(*stack->values)) {
		errno = ENOMEM;
		die("virtual stack");
	}

	mpz_t *new_values = realloc(stack->values,
	                            new_capacity * sizeof(*stack->values));
	if (new_values == NULL)
		die("virtual stack");

	stack->values = new_values;
	for (size_t i = old_capacity; i < new_capacity; ++i)
		mpz_init(stack->values[i]);
	stack->capacity = new_capacity;
}

static void stack_push(virtual_stack *stack, const mpz_t value)
{
	if (stack->size == stack->capacity)
		stack_grow(stack);
	mpz_set(stack->values[stack->size++], value);
}

static void stack_pop(virtual_stack *stack, mpz_t value)
{
	mpz_set(value, stack->values[--stack->size]);
}

static void stack_clear(virtual_stack *stack)
{
	for (size_t i = 0; i < stack->capacity; ++i)
		mpz_clear(stack->values[i]);
	free(stack->values);
}

/*
 * Evaluate Ackermann without using the C call stack.  Each saved m is a
 * pending invocation.  For A(m, n) with m,n > 0, pushing m-1 followed by m
 * evaluates the inner A(m, n-1) before its outer continuation A(m-1, ...).
 */
static void ackermann(mpz_t result, const mpz_t input_m, const mpz_t input_n)
{
	virtual_stack stack;
	mpz_t m;

	stack_init(&stack);
	mpz_init(m);
	mpz_set(result, input_n);
	stack_push(&stack, input_m);

	while (stack.size != 0) {
		stack_pop(&stack, m);
		atomic_fetch_add_explicit(&count, 1, memory_order_relaxed);

		if (mpz_sgn(m) == 0) {
			mpz_add_ui(result, result, 1);
		} else if (mpz_sgn(result) == 0) {
			mpz_set_ui(result, 1);
			mpz_sub_ui(m, m, 1);
			stack_push(&stack, m);
		} else {
			mpz_sub_ui(result, result, 1);
			mpz_sub_ui(m, m, 1);
			stack_push(&stack, m);
			mpz_add_ui(m, m, 1);
			stack_push(&stack, m);
		}
	}

	mpz_clear(m);
	stack_clear(&stack);
}

static void display(union sigval sigval)
{
	(void)sigval;
	clock_t ticks = clock() - start;
	unsigned long long calls =
	    atomic_load_explicit(&count, memory_order_relaxed);
	double seconds = (double)ticks / CLOCKS_PER_SEC;

	if (ticks > 0)
		fprintf(stderr, "\rC=%5.3fk\tT=%5.3fk\t@%5.3f MT/s",
		        (double)calls / 1000.0, seconds,
		        (double)calls / (double)ticks);
}

static int parse_nonnegative(mpz_t value, const char *text)
{
	return text[0] != '\0' && text[0] != '-' && mpz_set_str(value, text, 10) == 0;
}

int main(int argc, char **argv)
{
	mpz_t arg_m, arg_n, result;
	timer_t timer;
	struct sigevent ev = {
		.sigev_notify = SIGEV_THREAD,
		.sigev_notify_function = display,
		.sigev_notify_attributes = NULL,
	};
	struct itimerspec period = {
		.it_value.tv_sec = 1,
		.it_interval.tv_nsec = 50000000L,
	};

	if (argc != 3) {
		fprintf(stderr, "Usage: %s M N\n", argv[0]);
		return EXIT_FAILURE;
	}

	mpz_inits(arg_m, arg_n, result, NULL);
	if (!parse_nonnegative(arg_m, argv[1]) ||
	    !parse_nonnegative(arg_n, argv[2])) {
		fprintf(stderr, "M and N must be non-negative decimal integers.\n");
		mpz_clears(arg_m, arg_n, result, NULL);
		return EXIT_FAILURE;
	}

	if (timer_create(CLOCK_MONOTONIC, &ev, &timer) == -1)
		die("timer_create");
	if (timer_settime(timer, 0, &period, NULL) == -1)
		die("timer_settime");

	start = clock();
	ackermann(result, arg_m, arg_n);
	clock_t end = clock();
	timer_delete(timer);

	gmp_printf("\nTotal %.6f sec, %llu time(s) called, "
	           "ACKERMANN(%Zd, %Zd) = %Zd\n",
	           (double)(end - start) / CLOCKS_PER_SEC,
	           atomic_load_explicit(&count, memory_order_relaxed), arg_m, arg_n,
	           result);
	if (end != start)
		printf("%.2f c/usec\n",
		       (double)atomic_load_explicit(&count, memory_order_relaxed) /
		           (double)(end - start));

	mpz_clears(arg_m, arg_n, result, NULL);
	return EXIT_SUCCESS;
}
