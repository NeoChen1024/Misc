#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef long long int num_t;

double distance(double a, double b)
{
	return sqrt(a*a + b*b);
}

int main(int argc, char **argv)
{
	double x, y, d;
	int size;

	if(argc > 2)
		goto error;

	size = atoi(argv[1]);

	if(size < 0)
		goto error;
	/* plot size: size * 2 + 1 */

	for(y = size; y >= -size; y--)
	{
		for(x = -size; x <= size; x += 0.5)
		{
			d = distance(x, y);

			if(d <= size)
				putchar('X');
			else if(d - size <= 0.125)
				putchar('=');
			else if(d - size <= 0.25)
				putchar('-');
			else
				putchar(' ');
		}

		putchar('\n');
	}
	exit(0);
error:
	exit(1);
}
