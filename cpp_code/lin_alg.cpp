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
#include "lin_alg.h"
using namespace std;

bool invert_matrix(double ** &M, int m)
{
	// declarations
	int r,c,k,r_max;
	double m_val,mult;
	double * tmp_row;
	
	// Initialize M_inv as I
	double ** M_inv = new double*[m];
	for(r = 0; r < m; r++)
	{
		M_inv[r] = new double[m]();
		M_inv[r][r] = 1.0;
	}
	
	for(k = 0; k < m; k++)
	{
		m_val = 0;
		r_max = 0;
		for(r = k; r < m; r++)
		{
			if(abs(M[r][k]) > m_val)
			{
				m_val = abs(M[r][k]);
				r_max = r;
			}
		}
		
		if(m_val == 0)
			return(false);
		
		//swap rows k and r_max
		tmp_row = M[k];
		M[k] = M[r_max];
		M[r_max] = tmp_row;
		tmp_row = M_inv[k];
		M_inv[k] = M_inv[r_max];
		M_inv[r_max] = tmp_row;
		
		for(r = 0; r < m; r++)
		{
			if(r == k)
				continue;
			mult = M[r][k]/M[k][k];
			for(c = (k+1); c < m; c++)
				M[r][c] = M[r][c] - M[k][c]*mult;
			M[r][k] = 0;
			for(c = 0; c < m; c++)
				M_inv[r][c] = M_inv[r][c] - M_inv[k][c]*mult;
		}
	}
	
	for(r = 0; r < m; r++)
		for(c = 0; c < m; c++)
			M_inv[r][c] /= M[r][r];
			
	for(r = 0; r < m; r++)
	{
		delete[] M[r];
		M[r] = M_inv[r];
	}
	delete[] M_inv;
	
	return(true);
}

double dot(double * a, double * b, int len)
{
	double res = 0;
	for(int i = 0; i < len; i++)
	{
		res += a[i]*b[i];
	}
	return(res);
}

void mat_vec_mult(double ** M, double * a, int m, int n, double * target)
{
	for(int i = 0; i < m; i++)
		target[i] = dot(M[i],a,n);
}
	
	
	