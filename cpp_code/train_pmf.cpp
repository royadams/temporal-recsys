#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include "pmf_recommender.h"
#include "ran_sampler.h"
using namespace std;
using namespace pmf;

int main(int argc, char ** argv)
{
	time_t bef,aft;
	string model_fn = string(argv[1]);
	string test_data_fn = string(argv[2]);
	int n_test_samples = atoi(argv[3]);
	bool save_model = atoi(argv[4]);
	PMF_recommender model(model_fn);
	time(&bef);
	model.fit_sgd(false);
	time(&aft);
	cout << "Train time was " << difftime(aft,bef) << " seconds" << endl;
	cout << "Test RMSE is: " << model.RMSE(test_data_fn,n_test_samples) << endl;
	
	if(save_model)
	{
		string model_out_fn = string(argv[5]);
		string out_fldr = "/home/rjadams/YahooMusic/output/";
		model.save_model(out_fldr,model_out_fn);
	}
	
	return 0;
}