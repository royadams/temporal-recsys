// The MIT License (MIT)
//
// Copyright (c) 2013 Roy Adams
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <cmath>
#include <cstdlib>
#include "ran_sampler.h"

using namespace std;


// Seed the C random number generator
void seeder(float seed)
{
	srand(seed);
}

// Generate a U(0,1) RV
double unif_01()
{
	return rand()/double(RAND_MAX);
}

// Generate a single standard normal sample
// From:
// Joseph L. Leva: A Fast Normal Random Number Generator
double normal(double mean, double std_dev)
{
	double u,v,x,y,Q;
	do
	{
		do
		{
			u = unif_01();
		}while(u == 0.0);
		v = 1.7156*(unif_01() - 0.5);
		x = u - 0.449871;
		y = abs(v) + 0.386595;
		Q = x*x + y*(0.19600 * y - 0.25472 * x);
		if(Q < 0.27597) break;
	}while((Q > 0.27846) || ((v*v) > -4.0*(u*u)*(log(u))));
	
	return mean + std_dev*(v/u);
}

// Generate and array of normal RVs
void normal_array(double * arr, int len, double mean, double std_dev)
{
	for( int i = 0; i < len; i++)
		arr[i] = normal(mean,std_dev);
}

void normal_matrix(double** M,int nrows,int ncols,double mean,double std_dev)
{
	for( int r = 0; r < nrows; r++)
		normal_array(M[r],ncols,mean,std_dev);
}
