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
	
	
	