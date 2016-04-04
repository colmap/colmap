/* 
 *  Copyright (c) 2008  Noah Snavely (snavely (at) cs.washington.edu)
 *    and the University of Washington
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 */

/* util.c */
/* Various utility functions */

#include <stdlib.h>

/* Returns the floor of the log (base 2) of the given number */
int ilog2(int n) {
    if (n == 0)
	return 0;
    if (n == 1)
	return 0;
    else
	return ilog2(n >> 1) + 1;
}

/* Returns a random double between 0.0 and 1.0 */
double rand_unit() {
    return (((double) rand()) / RAND_MAX);
}

/* Returns a random double between min and max */
double rand_double(double min, double max) {
    return min + rand_unit() * (max - min);
}

/* Clamps the value x to lie within min and max */
double clamp(double x, double min, double max) {
    if (x < min)
	return min;
    else if (x > max) 
	return max;
    
    return x;
}

/* Returns true if n is a power of two */
int is_power_of_two(int n) {
    return ((n & (n - 1)) == 0);
}

/* Returns the smallest power of two no smaller than the input */
int least_larger_power_of_two(int n) {
    int i;

    if (n < 0)
	return 1;

    if (is_power_of_two(n))
	return n;
    
    i = 0;
    while ((n >> i) != 0)
	i++;
    
    return (1 << i);
}

/* Return the closest integer to x, rounding up */
int iround(double x) {
    if (x < 0.0) {
	return (int) (x - 0.5);
    } else {
	return (int) (x + 0.5);
    }
}
