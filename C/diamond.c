/* Diamond of "*" */
/* Size will be 2*input+1 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void print_head(int now, int all);
void print_body(int now);
void print_tail(int count);

int main(int argc, char **argv)
{
	int lines=1;
	int current=1;
	if(argc == 2)
		sscanf(argv[1], "%d", &lines);
	else
		return 2;

	for(current = 1; current <= (lines + 1); current++)
	{
		print_head(current, lines);
		print_body(current);
	}
	for(current = lines; current >= 1; current--)
	{
		print_head(current, lines);
		print_body(current);
	}
	return 0;
}

void print_head(int now, int all)
{
	int count=0;
	for(count = 0; count <= (all - now); count++)
		putchar(' ');
}

void print_body(int now)
{
	int count=1;

	for(count=1; count <= now; count++)
	{
		if(count == 1)
			putchar('*');
		else if(count == now)
		{
			printf(" *");
		}
		else
			printf("  ");
	}
	putchar('\n');
}
