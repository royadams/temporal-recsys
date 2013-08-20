#ifndef RANDOM_SAMPLER
#define RANDOM_SAMPLER

void seeder(float);
double unif_01(void);
double normal(double,double);
void normal_array(double*,int,double,double);
void normal_matrix(double**,int,int,double,double);

#endif