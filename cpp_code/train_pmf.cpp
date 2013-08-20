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