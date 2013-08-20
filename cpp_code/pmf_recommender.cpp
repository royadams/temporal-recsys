#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include "ran_sampler.h"
#include "lin_alg.h"
#include "pmf_recommender.h"
#include "/home/rjadams/cpp_utils/cpp_utils.h"

using namespace std;
using namespace pmf;

int sign(double x) 
{
	if (x > 0) return 1;
	if (x < 0) return -1;
	return 0;
}

PMF_recommender::PMF_recommender(	string config_fn):
									rank(0),
									lr(.001),
									rmax(9999),
									rmin(-9999)
{
	int c = 0;
	nb = 0;
	nuff = 0;
	nvff = 0;
	nw = 0;
	ifstream file(config_fn.c_str(),ios::in);
	string line;
	string arg,val;
	size_t pos;
	n_disc_features=0;
	n_cont_features=0;
	string train_data_fn,test_data_fn;
	string train_feature_fn,test_feature_fn;
	string bias_combo_str,bias_reg_str;
	string uff_str = "";
	string uff_reg_str = "";
	string vff_str = ""; 
	string vff_reg_str = "";
	string w_str = "";
	vector<string> uff_names,vff_names;
	
	while(getline(file,line))
	{
		if( line[0] == '#')
			continue;
			
		pos = line.find("=");
		arg = line.substr(0,pos);
		val = line.substr(pos+1);
			
		if(arg == "n_users")
			nu = long(atoi(val.c_str()));
		else if(arg == "n_items")
			ni = long(atoi(val.c_str()));
		else if(arg == "n_biases")
			nb = long(atoi(val.c_str()));
		else if( arg == "mf_rank")
			rank =  long(atoi(val.c_str()));
		else if(arg == "bias_combos")
			bias_combo_str = val;
		else if(arg == "n_epochs")
			n_epochs = long(atoi(val.c_str()));
		else if(arg == "train_data_fn")
			train_data_fn = val;
		else if(arg == "test_data_fn")
			test_data_fn = val;
		else if(arg == "train_feature_fn")
			train_feature_fn = val;
		else if(arg == "test_feature_fn")
			test_feature_fn = val;
		else if(arg == "n_train_samples")
			n_samples = int(atoi(val.c_str()));
		else if(arg == "n_test_samples")
			n_test_samples = int(atoi(val.c_str()));
		else if(arg == "init_stdev")
			sd = double(atof(val.c_str()));
		else if(arg == "seed")
			seed = atoi(val.c_str());
		else if(arg == "average_last_pass")
			avg = (val == "true");
		else if(arg == "verbose")
			verbose = (val == "true");
		else if(arg == "track_train_error")
			track_train_errs = (val == "true");
		else if(arg == "track_test_error")
			track_test_errs = (val == "true");
		else if(arg == "u_reg")
			ur = double(atof(val.c_str()));
		else if(arg == "v_reg")
			vr = double(atof(val.c_str()));
		else if(arg == "learning_rate")
			lr = double(atof(val.c_str()));
		else if(arg == "bias_regs")
			bias_reg_str = val;
		else if(arg == "user_feature_factors")
			uff_str = val;
		else if(arg == "item_feature_factors")
			vff_str = val;
		else if(arg == "user_feature_factor_regs")
			uff_reg_str = val;
		else if(arg == "item_feature_factor_regs")
			vff_reg_str = val;
		else if(arg == "linear_factors")
			w_str = val;
		else if(arg == "linear_factor_regs")
			w_mult = double(1-atof(val.c_str()));
		else if(arg == "rmax")
			rmax = int(atoi(val.c_str()));
		else if(arg == "rmin")
			rmin = int(atoi(val.c_str()));
		else
		{
			cout << "Invalid config line." << endl;
			cout << line << endl;
			exit(1);
		}
	}
	
	file.close();
	
	read_init_data_file(&train_data,&train_cont_features,train_data_fn,n_samples);
	
	if(rank > 0)
	{
		U = new double*[nu];
		for(int u =0; u<nu;u++)
			U[u] = new double[rank];
		V = new double*[ni];
		for(int i=0;i<ni;i++)
			V[i] = new double[rank];
		u_mult = 1 - lr*ur;
		v_mult = 1 - lr*vr;
		
		istringstream uff_stream(uff_str);
		istringstream vff_stream(vff_str);
		istringstream uff_reg_stream(uff_reg_str);
		istringstream vff_reg_stream(vff_reg_str);
		string feat,reg;
		while(getline(uff_stream,feat,';'))
		{
			cout << "in uff" << endl;
			for(c = 0; c < n_disc_features; c++)
				if(feat == disc_feature_names[c])
					break;
			uff_cols.push_back(c);
			getline(uff_reg_stream,reg,';');
			uff_mults.push_back(1 - lr*atof(reg.c_str()));
		}
		while(getline(vff_stream,feat,';'))
		{
			cout << "in vff" << endl;
			for(c = 0; c < n_disc_features; c++)
				if(feat == disc_feature_names[c])
					break;
			vff_cols.push_back(c);
			getline(vff_reg_stream,reg,';');
			vff_mults.push_back(1 - lr*atof(reg.c_str()));
		}
		
		nuff = uff_mults.size();
		nvff = vff_mults.size();
		Uff = new double***[nuff];
		Vff = new double***[nvff];
		uff_maxes = new int *[nuff];
		vff_maxes = new int *[nvff];
		
		cout << "nuff = " << nuff << " and nvff = " << nvff << endl;
		
		for(int ff = 0; ff < nuff; ff++)
		{
			Uff[ff] = new double**[nu];
			max_by_bin(train_data,0,uff_cols[ff],nu,n_samples,uff_maxes[ff]);
			for(int u = 0; u < nu; u++)
			{
				Uff[ff][u] = new double*[uff_maxes[ff][u]+1];
				for(int t = 0; t < uff_maxes[ff][u]+1; t++)
					Uff[ff][u][t] = new double[rank];
			}
		}
		for(int ff = 0; ff < nvff; ff++)
		{
			Vff[ff] = new double**[ni];
			max_by_bin(train_data,1,vff_cols[ff],ni,n_samples,vff_maxes[ff]);
			for(int i = 0; i < ni; i++)
			{
				Vff[ff][i] = new double*[vff_maxes[ff][i]+1];
				for(int t = 0; t < vff_maxes[ff][i]+1; t++)
					Vff[ff][i][t] = new double[rank];
			}
		}
	}
	else
	{
		U = NULL;
		V = NULL;
	}
	
	if(nb > 0)
	{
		brs = new double**[nb];
		// b_mults = new float[nb];
		d0s = new int[nb];
		d1s = new int[nb];
		B = new double**[nb];
		b_buff = new double*[nb];
			
		string feat;
		string reg;
		istringstream bc_stream(bias_combo_str);
		istringstream br_stream(bias_reg_str);
		for(int b = 0; b < nb; b++)
		{
			getline(bc_stream,feat,',');
			for(c = 0; c < n_disc_features; c++)
				if(feat == disc_feature_names[c])
					break;
			d0s[b] = c;
			
			getline(bc_stream,feat,';');
			for(c = 0; c < n_disc_features; c++)
				if(feat == disc_feature_names[c])
					break;
			d1s[b] = c;
			
			getline(br_stream,reg,':');
			if(reg == "rbf")
			{
				bias_prior_types.push_back(RBF);
				getline(br_stream,reg,',');
				a_inv.push_back(double(atof(reg.c_str())));
				getline(br_stream,reg,';');
				b_inv.push_back(double(atof(reg.c_str())));
				brs[b] = new double*[disc_feature_ranges[d1s[b]]];
				b_buff[b] = new double[disc_feature_ranges[d1s[b]]];
				for(int t = 0; t < disc_feature_ranges[d1s[b]]; t++)
				{
					brs[b][t] = new double[disc_feature_ranges[d1s[b]]];
					for(int tp = 0; tp < disc_feature_ranges[d1s[b]]; tp++)
						brs[b][t][tp] = exp(-1*pow((t-tp),2)/b_inv[b])/a_inv[b];
				}
				invert_matrix(brs[b],disc_feature_ranges[d1s[b]]);
			}
			else if(reg == "ind")
			{
				bias_prior_types.push_back(IND);
				getline(br_stream,reg,';');
				a_inv.push_back(double(1-atof(reg.c_str())*lr));
				b_inv.push_back(0);
			}
			else
			{
				cout << "Invalid bias prior type: " << disc_feature_names[d0s[b]] << "," << disc_feature_names[d1s[b]] << endl;
				exit(1);
			}
				
			// brs[b] = double(atof(reg.c_str()));
			// b_mults[b] = float(1-brs[b]*lr);
					
			B[b] = new double*[disc_feature_ranges[d0s[b]]];
			for(int d0 = 0; d0 < disc_feature_ranges[d0s[b]]; d0++)
				B[b][d0] = new double[disc_feature_ranges[d1s[b]]];
		}
	}
	else
	{
		B = NULL;
		brs = NULL;
		d0s = NULL;
		d1s = NULL;
		nb = 0;
	}
	
	if(w_str != "")
	{
		istringstream wss(w_str);
		string feat;
		while(getline(wss,feat,';'))
		{
			for(c = 0; c < n_cont_features; c++)
				if(feat == cont_feature_names[c])
					break;
			if(c == n_cont_features)
			{
				cout << feat << " not in data file." << endl;
				exit(0);
			}
			w_cols.push_back(c);
		}
		nw = w_cols.size();
		
		W = new double[nw];
	}
	
	if(track_test_errs)
	{
		read_data_file(&test_data,&test_cont_features,test_data_fn,n_test_samples);
	}
	else
	{
		test_data = NULL;
	}
}// Constructor

PMF_recommender::~PMF_recommender()
{
	if(rank > 0)
	{
		for(int u = 0; u < nu; u++)
			delete[] U[u];
		delete[] U;
		for(int i = 0; i < ni; i++)
			delete[] V[i];
		delete[] V;
		
		if(nuff > 0)
		{
			for(int ff = 0; ff < nuff; ff++)
			{
				for(int u = 0; u < nu; u++)
				{
					for(int t = 0; t < uff_maxes[ff][u]+1; t++)
					{
						delete[] Uff[ff][u][t];
					}
					delete[] Uff[ff][u];
				}
				delete[] Uff[ff];
			}
			delete[] Uff;
		}
		if(nvff > 0)
		{
			for(int ff = 0; ff < nvff; ff++)
			{
				for(int i = 0; i < ni; i++)
				{
					for(int t = 0; t < vff_maxes[ff][i]+1; t++)
					{
						delete[] Vff[ff][i][t];
					}
					delete[] Vff[ff][i];
				}
				delete[] Vff[ff];
			}
			delete[] Vff;
		}
		
	}
	
	if(nb > 0)
	{
		for(int b = 0; b < nb; b++)
		{
			for(int d0 = 0; d0 < disc_feature_ranges[d0s[b]]; d0++)
				delete[] B[b][d0];
			if(bias_prior_types[b] == RBF)
			{
				for(int t = 0; t < disc_feature_ranges[d1s[b]]; t++)
					delete[] brs[b][t];
				delete[] brs[b];
			}
			delete[] B[b];
		}
		delete[] B;
		delete[] d0s;
		delete[] d1s;
		delete[] brs;
	} 
	
	
	for(int s = 0; s < n_samples; s++)
		delete[] train_data[s];
	delete[] train_data;
	
	if(track_test_errs)
	{
		for(int s = 0; s < n_test_samples; s++)
			delete[] test_data[s];
		delete[] test_data;
	}	
	
}// Destructor

void PMF_recommender::max_by_bin(int** M,int bin_col,int val_col,int n_bins,int M_len, int * &maxes)
{
	int r,bin;
	maxes = new int[n_bins];
	for(bin = 0; bin < n_bins; bin++)
		maxes[bin] = -9999999;
	
	for(r = 0; r < M_len; r++)
	{
		if(M[r][val_col] > maxes[M[r][bin_col]])
			maxes[M[r][bin_col]] = M[r][val_col];
	}
}


double PMF_recommender::predict(int u, int i, int * cur_samp, float * c_feats)
{
	int ff;
	double pred = mu;
	double uuk,vik;
	for(int b = 0; b < nb; b++)
		pred += B[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]];
	
	for(int w = 0; w < nw; w++)
		pred += W[w] * c_feats[w_cols[w]];
		
	if(nuff > 0 || nvff > 0)
	{
		for(int k = 0; k < rank; k++)
		{
			uuk = U[u][k];
			vik = V[i][k];
			for(ff = 0; ff < nuff; ff++)
				if(cur_samp[uff_cols[ff]] <= uff_maxes[ff][u])
					uuk += Uff[ff][u][cur_samp[uff_cols[ff]]][k];
			for(ff = 0; ff < nvff; ff++)
				if(cur_samp[vff_cols[ff]] <= vff_maxes[ff][i])
					vik += Vff[ff][i][cur_samp[vff_cols[ff]]][k];
			pred += uuk*vik;
		}
	}
	else
	{
		for(int k = 0; k < rank; k++)
			pred += U[u][k]*V[i][k];
	}
	
		
	return(pred);
}

double PMF_recommender::test_pred(int u, int i, int * cur_samp, float * c_feats)
{
	int ff;
	double pred = mu;
	double uuk,vik;
	// double *** Uffut;
	// double *** Vffit;
	
	for(int b = 0; b < nb; b++)
		if(B[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]] > -999998)
			pred += B[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]];
		
	
	for(int w = 0; w < nw; w++)
		if(W[w] > -999998)
			pred += W[w] * c_feats[w_cols[w]];
		
	if(nuff > 0 || nvff > 0)
	{
		for(int k = 0; k < rank; k++)
		{
			uuk = U[u][k];
			vik = V[i][k];
			for(ff = 0; ff < nuff; ff++)
				if(cur_samp[uff_cols[ff]] <= uff_maxes[ff][u] && Uff[ff][u][cur_samp[uff_cols[ff]]][k] > -999998)
					uuk += Uff[ff][u][cur_samp[uff_cols[ff]]][k];
			for(ff = 0; ff < nvff; ff++)
				if(cur_samp[vff_cols[ff]] <= vff_maxes[ff][i] && Vff[ff][i][cur_samp[vff_cols[ff]]][k] > -999998)
					vik += Vff[ff][i][cur_samp[vff_cols[ff]]][k];
			pred += uuk*vik;
		}
		
	}
	else
	{
		for(int k = 0; k < rank; k++)
			pred += U[u][k]*V[i][k];
	}
		
	if(pred > rmax)
		pred = rmax;
	if(pred < rmin)
		pred = rmin;
	return(pred);
}

void PMF_recommender::update_parameters(const int u,const int i,const double err_ui, int * cur_samp, float * c_feats)
{
	int ff;
	double u_buff;
	double v_buff;
	double errlr = lr*err_ui;
	// double reglr;
	
	mu += errlr;
	
	for(int b = 0; b < nb; b++)
	{
		if(bias_prior_types[b] == IND)
			B[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]] = errlr + a_inv[b]*B[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]];
		else if(bias_prior_types[b] == RBF)
		{
			mat_vec_mult(brs[b],B[b][cur_samp[d0s[b]]],disc_feature_ranges[d1s[b]],disc_feature_ranges[d1s[b]],b_buff[b]);
			B[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]] += errlr;
			for(int t = 0; t < disc_feature_ranges[d1s[b]]; t++)
				B[b][cur_samp[d0s[b]]][t] -= lr*b_buff[b][t];
			// reglr = lr*dot(brs[b][d1s[b]],B[b][cur_samp[d0s[b]]],disc_feature_ranges[d1s[b]]);
			// B[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]] += errlr - reglr;
		}
	}
	
	for(int w = 0; w < nw; w++)
		W[w] = errlr*c_feats[w_cols[w]] + w_mult*W[w];
		
	if(nuff > 0 || nvff > 0)
	{
		for( int k = 0; k < rank; k++)
		{
			u_buff = U[u][k];
			v_buff = V[i][k];
			for(ff = 0; ff < nuff; ff++)
				u_buff += Uff[ff][u][cur_samp[uff_cols[ff]]][k];
			for(ff = 0; ff < nvff; ff++)
				v_buff += Vff[ff][i][cur_samp[vff_cols[ff]]][k];
			U[u][k] = errlr*v_buff + u_mult*U[u][k];
			V[i][k] = errlr*u_buff + v_mult*V[i][k];
			for(ff = 0; ff < nuff; ff++)
				Uff[ff][u][cur_samp[uff_cols[ff]]][k] = errlr*v_buff + uff_mults[ff]*Uff[ff][u][cur_samp[uff_cols[ff]]][k];
			for(ff = 0; ff < nvff; ff++)
				Vff[ff][i][cur_samp[vff_cols[ff]]][k] = errlr*u_buff + vff_mults[ff]*Vff[ff][i][cur_samp[vff_cols[ff]]][k];
		}
	}
	else
	{	
		for( int k = 0; k < rank; k++)
		{
			u_buff = U[u][k];
			U[u][k] = errlr*V[i][k] + u_mult*U[u][k];
			V[i][k] = errlr*u_buff + v_mult*V[i][k];
		}
	}
}

void PMF_recommender::read_init_data_file(int *** target, float *** c_feats, string fn,int n_samples)
{
	int dc = 0;
	int cc = 0;
	int c = 0;
	int range = 0;
	int nf = 0;
	vector<bool> disc;
	ifstream file(fn.c_str(),ios::in);
	string line,names,cards;
	string feat,card;
	getline(file,names);
	getline(file,cards);
	istringstream name_str(names);
	istringstream card_str(cards);
	n_disc_features = 1;
	while(getline(name_str,feat,',') && getline(card_str,card,','))
	{
		range = atoi(card.c_str());
		if(range == -1)
		{
			n_cont_features++;
			cont_feature_names.push_back(feat);
			disc.push_back(false);
		}
		else
		{
			n_disc_features++;
			disc_feature_names.push_back(feat);
			disc_feature_ranges.push_back(range);
			disc.push_back(true);
		}
	}
	nf = n_disc_features+n_cont_features;
	disc_feature_names.push_back("const");
	disc_feature_ranges.push_back(1);
	(*target) = new int*[n_samples];
	if(n_cont_features > 0)
		(*c_feats) = new float*[n_samples];
	for(int s = 0; s < n_samples; s++)
	{
		(*target)[s] = new int[n_disc_features];
		if(n_cont_features > 0)
			(*c_feats)[s] = new float[n_cont_features];
		dc = 0;
		cc = 0;
		for(c = 0; c < (nf-2); c++)
		{
			getline(file,line,',');
			if(disc[c])
			{
				(*target)[s][dc] = atoi(line.c_str());
				dc++;
			}
			else
			{
				(*c_feats)[s][cc] = atof(line.c_str());
				cc++;
			}
		}
		getline ( file, line);
		if(disc[c])
		{
			(*target)[s][dc] = atoi(line.c_str());
			dc++;
		}
		else
		{
			(*c_feats)[s][cc] = atof(line.c_str());
		}
		(*target)[s][dc] = 0;
	}
	file.close();
}

void PMF_recommender::read_data_file(int *** target,float *** c_feats, string fn,int n_samples)
{
	int c = 0;
	int dc = 0;
	int cc = 0;
	int range = 0;
	int nf = n_disc_features + n_cont_features;
	vector<bool> disc;
	ifstream file(fn.c_str(),ios::in);
	string line,names,cards;
	string card;
	getline(file,names);
	getline(file,cards);
	istringstream card_str(cards);
	while(getline(card_str,card,','))
	{
		range = atoi(card.c_str());
		if(range == -1)
			disc.push_back(false);
		else
			disc.push_back(true);
	}
	(*target) = new int*[n_samples];
	if(n_cont_features > 0)
		(*c_feats) = new float*[n_samples];
	for(int s = 0; s < n_samples; s++)
	{
		(*target)[s] = new int[n_disc_features];
		if(n_cont_features > 0)
			(*c_feats)[s] = new float[n_cont_features];
		dc = 0;
		cc = 0;
		for(c = 0; c < (nf-2); c++)
		{
			getline(file,line,',');
			if(disc[c])
			{
				(*target)[s][dc] = atoi(line.c_str());
				dc++;
			}
			else
			{
				(*c_feats)[s][cc] = atof(line.c_str());
				cc++;
			}
		}
		getline ( file, line);
		if(disc[c])
		{
			(*target)[s][dc] = atoi(line.c_str());
			dc++;
		}
		else
		{
			(*c_feats)[s][cc] = atof(line.c_str());
		}
		(*target)[s][dc] = 0;
	}
	file.close();
}

void PMF_recommender::init_params()
{
	seeder(seed);
	normal_array(W,nw,0,sd);
	for(int b = 0; b < nb; b++)
		normal_matrix(B[b],disc_feature_ranges[d0s[b]],disc_feature_ranges[d1s[b]],0,sd);
	if(rank > 0) 
	{
		normal_matrix(U,nu,rank,0,sd);
		normal_matrix(V,ni,rank,0,sd);
		for(int ff = 0; ff<nuff; ff++)
		{
			for(int u = 0; u < nu; u++)
			{
				normal_matrix(Uff[ff][u],uff_maxes[ff][u]+1,rank,0,sd);
			}
		}
		for(int ff = 0; ff<nvff; ff++)
		{
			for(int i = 0; i < ni; i++)
			{
				normal_matrix(Vff[ff][i],vff_maxes[ff][i]+1,rank,0,sd);
			}
		}
	}
	
}

double PMF_recommender::RMSE(	int** data, float ** c_feats, long ns)
{
	double err;
	double rmse = 0.0;
	float * cfp = NULL;
	for(int s = 0; s < ns; s++)
	{
		if(nw > 0)
			cfp = c_feats[s];
		err = double(data[s][2]) - test_pred(data[s][0],data[s][1],data[s],cfp);
		rmse += err*err;
	}
	return sqrt(rmse/double(ns));
}

double PMF_recommender::MAE(	int** data, float ** c_feats, long ns)
{
	double err;
	double rmse = 0.0;
	float * cfp = NULL;
	for(int s = 0; s < ns; s++)
	{
		if(nw > 0)
			cfp = c_feats[s];
		err = double(data[s][2]) - test_pred(data[s][0],data[s][1],data[s],cfp);
		rmse += abs(err);
	}
	return rmse/double(ns);
}

double PMF_recommender::kendalls_tau_b(int** data, float ** c_feats, long ns)
{
	float * cfp = NULL;
	vector<double> * user_ratings = new vector<double>[nu];
	vector<double> * user_preds = new vector<double>[nu];
	for(int s = 0; s < ns; s++)
	{
		if(nw > 0)
			cfp = c_feats[s];
		user_ratings[data[s][0]].push_back(double(data[s][2]));
		user_preds[data[s][0]].push_back(test_pred(data[s][0],data[s][1],data[s],cfp));
	}
	
	double kt = 0.0;
	int cnt = 0;
	
	for(int u = 0; u < nu; u++)
	{
		int nr = user_ratings[u].size();
		if(nr <= 1)
			continue;
		
		int n0 = nr*(nr-1)/2;
		int numer = 0;
		int sr,sp;
		bool flat_user = true;
		bool flat_preds = true;
		for(int r1 = 1; r1 < nr; r1++)
		{
			for(int r2 = 0; r2 < r1; r2++)
			{	
				sr = sign(user_ratings[u][r1] - user_ratings[u][r2]);
				sp = sign(user_preds[u][r1] - user_preds[u][r2]);
				if(sr != 0)
					flat_user = false;
				if(sp != 0)
					flat_preds = false;
				numer += sr*sp;
			}
		}
		
		if(flat_user)
			continue;
			
		if(flat_preds)
		{
			cnt++;
			continue;
		}
		
		user_ratings[u] = merge_sort<double>(user_ratings[u]);
		user_preds[u] = merge_sort<double>(user_preds[u]);
		
		double cur_pred = -9999;
		int cur_n_pred_ties = 1;
		double cur_rat = -9999;
		int cur_n_rat_ties = 1;
		double n1 = 0;
		double n2 = 0;
		
		for(int r = 0; r < nr; r++)
		{
			if(user_ratings[u][r] != cur_rat)
			{
				n1 += double(cur_n_rat_ties*(cur_n_rat_ties-1))/2;
				cur_rat = user_ratings[u][r];
				cur_n_rat_ties = 1;
			}
			else
				cur_n_rat_ties++;
			if(user_preds[u][r] != cur_pred)
			{
				n2 += double(cur_n_pred_ties*(cur_n_pred_ties-1))/2;
				cur_pred = user_preds[u][r];
				cur_n_pred_ties = 1;
			}
			else
				cur_n_pred_ties++;
		}
		
		n1 += double(cur_n_rat_ties*(cur_n_rat_ties-1))/2;
		n2 += double(cur_n_pred_ties*(cur_n_pred_ties-1))/2;
		
		
		kt += double(numer)/sqrt(double(n0-n1)*double(n0-n2));
		cnt++;
	}

	
	return(kt/double(cnt));
		
}
	
double PMF_recommender::NDCG(int** data, float ** c_feats, long ns)
{
	float * cfp = NULL;
	vector<double> * user_ratings = new vector<double>[nu];
	vector<double> * user_preds = new vector<double>[nu];
	for(int s = 0; s < ns; s++)
	{
		if(nw > 0)
			cfp = c_feats[s];
		user_ratings[data[s][0]].push_back(double(data[s][2]));
		user_preds[data[s][0]].push_back(test_pred(data[s][0],data[s][1],data[s],cfp));
	}
	
	double andcg = 0.0;
	double log2 = log(2);
	
	for(int u = 0; u < nu; u++)
	{
		int nr = user_ratings[u].size();
		if(nr <= 1)
			continue;
			
		
		vector<int> pred_order = merge_order<double>(user_preds[u]);
		vector<int> rat_order = merge_order<double>(user_ratings[u]);
		
		for(int r = 0; r < nr; r++)
		{
			user_preds[u][r] = 5 - user_preds[u][r];
			user_ratings[u][r] = 5 - user_ratings[u][r];
		}
		
		double udcg = user_ratings[u][pred_order[0]];
		double uidcg = user_ratings[u][rat_order[0]];
		for(int r = 1; r < nr; r++)
		{
			udcg += user_ratings[u][pred_order[r]]/(log(r+1)/log2);
			uidcg += user_ratings[u][rat_order[r]]/(log(r+1)/log2);
		}
		
		if((udcg + .0000001)/(uidcg + .0000001) > 1)
			cout << "here" << endl;
		
		andcg += (udcg + .0000001)/(uidcg + .0000001);
	}
	
	return(andcg/double(nu));
}		
	
double PMF_recommender::RMSE(string data_fn, long ns)
{
	int ** data;
	float ** c_feats = NULL;
	int s;
	double rmse;
	read_data_file(&data,&c_feats,data_fn,ns);
		
	rmse = RMSE(data,c_feats,ns);
	
	for(s = 0; s < ns; s++)
		delete[] data[s];
	delete data;
	return rmse;
}

void PMF_recommender::init_averages()
{
	mua = 0;

	if(rank > 0)
	{
		Ua = new double*[nu];
		Uc = new int[nu]();
		for(int u =0; u<nu;u++)
		{
			Ua[u] = new double[rank]();
		}
		Va = new double*[ni];
		Vc = new int[ni]();
		for(int i=0;i<ni;i++)
		{
			Va[i] = new double[rank]();
		}
		if(nuff > 0)
		{
			Uffa = new double***[nuff];
			Uffc = new int**[nuff];
			for(int ff = 0; ff < nuff; ff++)
			{
				Uffa[ff] = new double**[nu];
				Uffc[ff] = new int*[nu];
				for(int u = 0; u < nu; u++)
				{
					Uffa[ff][u] = new double*[uff_maxes[ff][u]+1];
					Uffc[ff][u] = new int[uff_maxes[ff][u]+1]();
					for(int t = 0; t < uff_maxes[ff][u]+1; t++)
					{
						Uffa[ff][u][t] = new double[rank]();
					}
				}
			}
		}
		if(nvff > 0)
		{
			Vffa = new double***[nvff];
			Vffc = new int**[nvff];
			for(int ff = 0; ff < nuff; ff++)
			{
				Vffa[ff] = new double**[ni];
				Vffc[ff] = new int*[ni];
				for(int i = 0; i < ni; i++)
				{
					Vffa[ff][i] = new double*[vff_maxes[ff][i]+1];
					Vffc[ff][i] = new int[vff_maxes[ff][i]+1]();
					for(int t = 0; t < vff_maxes[ff][i]+1; t++)
					{
						Vffa[ff][i][t] = new double[rank]();
					}
				}
			}
		}
	}
	if(nb > 0)
	{
		Ba = new double**[nb];
		Bc = new int**[nb];
		
		for(int b = 0; b < nb; b++)
		{
			Ba[b] = new double*[disc_feature_ranges[d0s[b]]];
			Bc[b] = new int*[disc_feature_ranges[d0s[b]]];
				for(int d0 = 0; d0 < disc_feature_ranges[d0s[b]]; d0++)
				{
					Ba[b][d0] = new double[disc_feature_ranges[d1s[b]]]();
					Bc[b][d0] = new int[disc_feature_ranges[d1s[b]]]();
				}
		}
	}
	if(nw > 0)
		Wa = new double[nw]();
}
		
void PMF_recommender::update_averages(int u, int i, int * cur_samp)
{
	mua += mu;
	if(rank > 0)
	{
		Uc[u] +=1;
		Vc[i] +=1;
		for(int k = 0; k < rank; k++)
		{
			Ua[u][k] += U[u][k];
			Va[i][k] += V[i][k];
		}
		for(int ff = 0; ff < nuff; ff++)
		{
			Uffc[ff][u][cur_samp[uff_cols[ff]]] += 1;
			for(int k = 0; k < rank; k++)
				Uffa[ff][u][cur_samp[uff_cols[ff]]][k] += Uff[ff][u][cur_samp[uff_cols[ff]]][k];
		}
		for(int ff = 0; ff < nvff; ff++)
		{
			Vffc[ff][i][cur_samp[vff_cols[ff]]] += 1;
			for(int k = 0; k < rank; k++)
				Vffa[ff][i][cur_samp[vff_cols[ff]]][k] += Vff[ff][i][cur_samp[vff_cols[ff]]][k];
		}
	}
	for(int b = 0; b < nb; b++)
	{
		Ba[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]] += B[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]];
		Bc[b][cur_samp[d0s[b]]][cur_samp[d1s[b]]] += 1;
	}	
	for(int w = 0; w < nw; w++)
		Wa[w] += W[w];
}

void PMF_recommender::finalize_averages(int ns)
{
	int u,i,k,b,d0,d1,ff,t;
	mu = mua/ns;
	if(rank > 0)
	{
		for(u = 0; u < nu; u++)
		{
			for(k = 0; k < rank; k++)
			{
				if(Uc[u] > 0)
					U[u][k] = Ua[u][k]/double(Uc[u]);
				else
					U[u][k] = -999999;
			}
			delete[] Ua[u];
		}
		delete[] Ua;
		delete[] Uc;
		for(i = 0; i < ni; i++)
		{
			for(k = 0; k < rank; k++)
			{
				if(Vc[i] > 0)
					V[i][k] = Va[i][k]/double(Vc[i]);
				else
					V[i][k] = -999999;
			}
			delete[] Va[i];
		}
		delete[] Va;
		delete[] Vc;
		if(nuff > 0)
		{
			for(ff = 0; ff < nuff; ff++)
			{
				for(u = 0; u < nu; u++)
				{
					for(t = 0; t < uff_maxes[ff][u]+1; t++)
					{
						for(k = 0; k < rank; k++)
						{
							if(Uffc[ff][u][t] > 0)
								Uff[ff][u][t][k] = Uffa[ff][u][t][k]/double(Uffc[ff][u][t]);
							else
								Uff[ff][u][t][k] = -999999;
						}
						delete[] Uffa[ff][u][t];
					}
					delete[] Uffa[ff][u];
					delete[] Uffc[ff][u];
				}
				delete[] Uffa[ff];
				delete[] Uffc[ff];
			}
			delete[] Uffa;
			delete[] Uffc;
		}
		if(nvff > 0)
		{
			for(ff = 0; ff < nvff; ff++)
			{
				for(i = 0; i < ni; i++)
				{
					for(t = 0; t < vff_maxes[ff][i]+1; t++)
					{
						for(k = 0; k < rank; k++)
						{
							if(Vffc[ff][i][t] > 0)
								Vff[ff][i][t][k] = Vffa[ff][i][t][k]/double(Vffc[ff][i][t]);
							else
								Vff[ff][i][t][k] = -999999;
						}
						delete[] Vffa[ff][i][t];
					}
					delete[] Vffa[ff][i];
					delete[] Vffc[ff][i];
				}
				delete[] Vffa[ff];
				delete[] Vffc[ff];
			}
			delete[] Vffa;
			delete[] Vffc;
		}
	}
	if(nb > 0)
	{
		for(b = 0; b < nb; b++)
		{
			for(d0 = 0; d0 < disc_feature_ranges[d0s[b]]; d0++)
			{
				for(d1 = 1; d1 < disc_feature_ranges[d1s[b]]; d1++)
				{
					if(Bc[b][d0][d1] > 0)
						B[b][d0][d1] = Ba[b][d0][d1]/double(Bc[b][d0][d1]);
					else
						B[b][d0][d1] = -999999;
				}
				delete[] Ba[b][d0];
				delete[] Bc[b][d0];
			}
			delete[] Ba[b];
			delete[] Bc[b];
		}
		delete[] Ba;
		delete[] Bc;
	}
	
	if(nw > 0)
	{	
		for(int w = 0; w < nw; w++)
			W[w] = Wa[w]/ns;
		delete[] Wa;
	}
}


void PMF_recommender::fit_sgd(	bool continue_fit = false)
{
	double err_ui;
	int u,i;
	double r;
	double train_err = 0.0;
	double test_err = 0.0;
	bool last_pass = false;
	bool using_w = false;
	float * cfp = NULL;

	if(verbose) cout << "*** Initializing ***" << endl;
	
	if(continue_fit){}
	else
		init_params();
	
	// Calculate mu
	mu = normal(0,sd);
	// for(int s = 0; s < n_samples; s++)
		// mu+=train_data[s][2];
	// mu/=n_samples;
	
	if(nw > 0)
		using_w = true;
	
	if(verbose) cout << "* The mean rating is: " << mu << endl;
	
	if(verbose) cout << "*** Training ***" << endl;
	for( int epoch = 0; epoch < n_epochs; epoch++)
	{
		if(avg && epoch == (n_epochs -1))
		{
			last_pass = true;
			init_averages();
		}
		if(verbose) cout << "* Epoch " << epoch << " *" << endl;
		for( int s = 0; s < n_samples; s++)
		{
			u = train_data[s][0];
			i = train_data[s][1];
			r = double(train_data[s][2]);
			
			// if(nuff > 0)
				// cout << "WHAAAAAAAAaaaAAAAAT!?!?" << endl;
			
			if(u >= nu || u < 0 || i < 0 || i >= ni)
			{
				cout << "u or i out of bounds: u=" << u << " i=" << i << endl;
				exit(1);
			}
			
			if(using_w)
				cfp = train_cont_features[s];
			
			err_ui = r - predict(u,i,train_data[s],cfp);
			
			update_parameters(u,i,err_ui,train_data[s],cfp);
			
			if(avg && last_pass)
				update_averages(u,i,train_data[s]);
			
			if(track_train_errs) train_err += err_ui*err_ui;		
			
		}
		
		if(track_train_errs)
		{
			cout << "Train err averaged over epoch " << epoch << " was: " << sqrt(train_err/double(n_samples)) << endl;
			train_err = 0.0;
		}
		if(track_test_errs)
		{
			test_err = RMSE(test_data,test_cont_features,long(n_test_samples));
			cout << "Test RMSE after " << epoch << " epochs is: " << test_err << endl;
		}
	}
	
	if(avg)
		finalize_averages(n_samples);
}

void PMF_recommender::fit_sgd(	int ** data, 
								float ** c_feats, 
								// vector<string> f_names, 
								int ns, 
								int ne, 
								double learning_rate = .001,
								double std_dev = 1,
								double ureg = 0,
								double vreg = 0,
								double wreg = 0,
								vector<double> bregs = vector<double>(),
								vector<double> uffregs = vector<double>(),
								vector<double> vffregs = vector<double>(),
								bool track_err = false,
								bool avg_last_pass = true,
								bool continue_fit = false)
{
	double err_ui;
	int u,i;
	double r;
	double train_err = 0.0;
	bool last_pass = false;
	bool using_w = false;
	float * cfp = NULL;

	// if(disc_feature_names.size() != f_names.size())
	// {
		// cout << "Invalid data file: ncols does not match" << endl;
		// exit(0);
	// }
	
	// for(int i = 0; i < int(f_names.size()); i++)
	// {
		// if(f_names[i] != disc_feature_names[i])
		// {
			// cout << "Invalid data file: f_names does not match" << endl;
			// exit(0);
		// }
	// }
	
	if(verbose) cout << "*** Initializing ***" << endl;
	
	lr = learning_rate;
	sd = std_dev;
	u_mult = 1-ureg;
	v_mult = 1-vreg;
	w_mult = 1-wreg;
	avg = avg_last_pass;
	track_train_errs = track_err;
	for(int b = 0; b < nb; b++)
		a_inv[b] = 1-bregs[b];
	for(int ff = 0; ff < nuff; ff++)
		uff_mults[ff] = 1-uffregs[ff];
	for(int ff = 0; ff < nvff; ff++)
		vff_mults[ff] = 1-vffregs[ff];
	
	if(continue_fit){}
	else
		init_params();
	
	// Calculate mu
	// mu = normal(0,sd);
	mu = 0.0;
	for(int s = 0; s < n_samples; s++)
		mu+=data[s][2];
	mu/=n_samples;
	
	if(nw > 0)
		using_w = true;
	
	if(verbose) cout << "* The mean rating is: " << mu << endl;
	
	if(verbose) cout << "*** Training ***" << endl;
	for( int epoch = 0; epoch < ne; epoch++)
	{
		if(avg && epoch == (ne - 1))
		{
			last_pass = true;
			init_averages();
		}
		if(verbose) cout << "* Epoch " << epoch << " *" << endl;
		for( int s = 0; s < ns; s++)
		{
			u = data[s][0];
			i = data[s][1];
			r = double(data[s][2]);
			
			if(u >= nu || u < 0 || i < 0 || i >= ni)
			{
				cout << "u or i out of bounds: u=" << u << " i=" << i << endl;
				exit(1);
			}
			
			if(using_w)
				cfp = c_feats[s];
				
			err_ui = r - predict(u,i,data[s],cfp);
			
			update_parameters(u,i,err_ui,data[s],cfp);
			
			if(avg && last_pass)
				update_averages(u,i,data[s]);
			
			if(track_train_errs) train_err += err_ui*err_ui;		
		}
		
		if(track_train_errs)
		{
			cout << "Train err averaged over epoch " << epoch << " was: " << sqrt(train_err/double(n_samples)) << endl;
			train_err = 0.0;
		}
	}
	
	if(avg)
		finalize_averages(ns);
}

void PMF_recommender::train_user(int u, int ** data, vector<string> & f_names, int ns, int ne)
{
	// double err_ui;
	// int i;
	// double r;
	// double train_err = 0.0;
	// bool last_pass = false;
	// vector<int> user_bs;
	// int nub;
	// double errlr;
	// double u_buff;
	// double v_buff;
	// int b;

	// if(disc_feature_names.size() != f_names.size())
	// {
		// cout << "Invalid data file: ncols does not match" << endl;
		// exit(0);
	// }
	
	// for(int i = 0; i < int(f_names.size()); i++)
	// {
		// if(f_names[i] != disc_feature_names[i])
		// {
			// cout << "Invalid data file: f_names does not match" << endl;
			// exit(0);
		// }
	// }
	
	// for(b = 0; b < nb; b++)
	// {
		// if(d0s[b] == 0 || d1s[b] == 0)
			// user_bs.push_back(b);
	// }
	// nub = user_bs.size();
	
	// // init user params
	// for(int ub = 0; ub < nub; ub++)
	// {
		// b = user_bs[ub];
		// if(d0s[b] == 0)
			// normal_array(B[b][u],disc_feature_ranges[d1s[b]],0,sd);
	// }
	// normal_array(U[u],rank,0,sd);
	// for(int ff = 0; ff < nuff; ff++)
	// {
		// for(int t = 0; t <= uff_maxes[ff][u]; t++)
		// {
			// normal_array(Uff[ff][u][t],rank,0,sd);
		// }
	// }
			
	// if(verbose) cout << "*** Training ***" << endl;
	// for( int epoch = 0; epoch < ne; epoch++)
	// {
		// if(avg && epoch == (ne - 1))
		// {
			// last_pass = true;
			// //TODO: init avgs
		// }
		// if(verbose) cout << "* Epoch " << epoch << " *" << endl;
		// for( int s = 0; s < ns; s++)
		// {
			// u = data[s][0];
			// if(data[s][0] != u)
				// continue;
			// i = data[s][1];
			// r = double(data[s][2]);
			
			// if(u >= nu || u < 0 || i < 0 || i >= ni)
			// {
				// cout << "u or i out of bounds: u=" << u << " i=" << i << endl;
				// exit(1);
			// }
			
			// err_ui = r - predict(u,i,data[s]);
			// errlr = lr*err_ui;
			// // update biases
			// for(int ub = 0; ub < nub; ub++)
			// {
				// b = user_bs[ub];
				// if(bias_prior_types[b] == IND)
					// B[b][data[s][d0s[b]]][data[s][d1s[b]]] = errlr + a_inv[b]*B[b][data[s][d0s[b]]][data[s][d1s[b]]];
				// else if(bias_prior_types[b] == RBF)
				// {
					// mat_vec_mult(brs[b],B[b][data[s][d0s[b]]],disc_feature_ranges[d1s[b]],disc_feature_ranges[d1s[b]],b_buff[b]);
					// B[b][data[s][d0s[b]]][data[s][d1s[b]]] += errlr;
					// for(int t = 0; t < disc_feature_ranges[d1s[b]]; t++)
						// B[b][data[s][d0s[b]]][t] -= lr*b_buff[b][t];
					// // reglr = lr*dot(brs[b][d1s[b]],B[b][data[s][d0s[b]]],disc_feature_ranges[d1s[b]]);
					// // B[b][data[s][d0s[b]]][data[s][d1s[b]]] += errlr - reglr;
				// }
			// }
				
			// // update U
			// // update UFF
			// if(nuff > 0)
			// {
				// for( int k = 0; k < rank; k++)
				// {
					// u_buff = U[u][k];
					// v_buff = V[i][k];
					// for(int ff = 0; ff < nuff; ff++)
						// u_buff += Uff[ff][u][data[s][uff_cols[ff]]][k];
					// for(int ff = 0; ff < nvff; ff++)
						// v_buff += Vff[ff][i][data[s][vff_cols[ff]]][k];
					// U[u][k] = errlr*v_buff + u_mult*U[u][k];
					// for(int ff = 0; ff < nuff; ff++)
						// Uff[ff][u][data[s][uff_cols[ff]]][k] = errlr*v_buff + uff_mults[ff]*Uff[ff][u][data[s][uff_cols[ff]]][k];
				// }
			// }
			// else
			// {	
				// for( int k = 0; k < rank; k++)
				// {
					// U[u][k] = errlr*V[i][k] + u_mult*U[u][k];
				// }
			// }
			
			// if(avg && last_pass)
				// //TODO: update averages
			
			// if(track_train_errs) train_err += err_ui*err_ui;		
		// }
		
		// if(track_train_errs)
		// {
			// cout << "Train err averaged over epoch " << epoch << " was: " << sqrt(train_err/double(n_samples)) << endl;
			// train_err = 0.0;
		// }
	// }
	
	// if(avg)
		// //TODO: finalize averages
		// cout << endl;
}

void PMF_recommender::write_double_csv(string fn, double ** M, int n_rows, int n_cols)
{
	int r,c;
	ofstream file(fn.c_str(),ios::out);
	file << n_rows << "," << n_cols << "\n";
	for(r = 0; r < n_rows; r++)
	{
		for(c = 0; c < (n_cols-1); c++)
			file << M[r][c] << ",";
		file << M[r][c] << "\n";
	}
	
	file.close();
}	

void PMF_recommender::save_model(string output_folder, string fn)
{
	stringstream num_str;
	string fldr = output_folder + fn + "/";
	system(("mkdir "+fldr).c_str());
	
	// write model file
	string mfn = fldr + fn + ".model";
	ofstream mfl(mfn.c_str(),ios::out);
	mfl << "rank=" << rank << endl;
	mfl << "nu=" << nu << endl;
	mfl << "ni=" << ni << endl;
	mfl << "nuff=" << nuff << endl;
	mfl << "nvff=" << nvff << endl;
	mfl << "nb=" << nb << endl;
	mfl << "nw=" << nw << endl;
	mfl << "mu=" << mu << endl;
	mfl << "uff_feats=";
	for(int i = 0; i < nuff; i++)
		mfl << disc_feature_names[uff_cols[i]] << ";";
	mfl << endl;
	mfl << "uff_ranges=";
	for(int i = 0; i < nuff; i++)
		mfl << disc_feature_ranges[uff_cols[i]] << ";";
	mfl << endl;
	mfl << "vff_feats=";
	for(int i = 0; i < nvff; i++)
		mfl << disc_feature_names[vff_cols[i]] << ";";
	mfl << endl;
	mfl << "vff_ranges=";
	for(int i = 0; i < nvff; i++)
		mfl << disc_feature_ranges[vff_cols[i]] << ";";
	mfl << endl;
	mfl << "d0_feats=";
	for(int i = 0; i < nb; i++)
		mfl << disc_feature_names[d0s[i]] << ";";
	mfl << endl;
	mfl << "d0_ranges=";
	for(int i = 0; i < nb; i++)
		mfl << disc_feature_ranges[d0s[i]] << ";";
	mfl << endl;
	mfl << "d1_feats=";
	for(int i = 0; i < nb; i++)
		mfl << disc_feature_names[d1s[i]] << ";";
	mfl << endl;
	mfl << "d1_ranges=";
	for(int i = 0; i < nb; i++)
		mfl << disc_feature_ranges[d1s[i]] << ";";
	mfl << endl;
	mfl << "w_feats=";
	for(int i = 0; i < nw; i++)
		mfl << cont_feature_names[w_cols[i]] << ";";
	mfl << endl;
	// mfl << "u_mult=" << u_mult << endl;
	// mfl << "v_mult=" << v_mult << endl;
	// mfl << "uff_multss=";
	// for(int i = 0; i < nuff; i++)
		// mfl << uff_mults[i] << ";";
	// mfl << endl;
	// mfl << "vff_mults=";
	// for(int i = 0; i < nvff; i++)
		// mfl << vff_mults[i] << ";";
	// mfl << endl;
	// mfl << "bias_prior_types=";
	// for(int i = 0; i < nb; i++)
		// mfl << bias_prior_types[i] << ";";
	// mfl << endl;
	// mfl << "a_inv=";
	// for(int i = 0; i < nb; i++)
		// mfl << a_inv[i] << ";";
	// mfl << endl;
	// mfl << "b_inv=";
	// for(int i = 0; i < nb; i++)
		// mfl << b_inv[i] << ";";
	// mfl << endl;
	mfl.close();
	
	// write U,V,UFF,VFF
	if(rank > 0)
	{
		string u_fn = fldr + fn+".U";
		string v_fn = fldr + fn+".V";
		
		ofstream uf(u_fn.c_str(),ios::out|ios::binary);
		ofstream vf(v_fn.c_str(),ios::out|ios::binary);
		
		for(int u=0; u < nu; u++)
			uf.write((char*)U[u], rank*sizeof(double));
		for(int i=0;i<ni;i++)
			vf.write((char*)V[i], rank*sizeof(double));
			
		uf.close();
		vf.close();
			
		string uff_fn = fldr + fn + ".UFF.";
		for(int ff = 0; ff < nuff; ff++)
		{
			num_str.str("");
			num_str << ff;
			ofstream ufff((uff_fn+num_str.str()).c_str(),ios::out|ios::binary);
			ofstream uffmf((uff_fn+num_str.str()+".maxes").c_str(),ios::out|ios::binary);
			uffmf.write((char*)uff_maxes[ff],nu*sizeof(int));
			for(int u = 0; u < nu; u++)
			{
				for(int t = 0; t <= uff_maxes[ff][u]; t++)
					ufff.write((char*)Uff[ff][u][t],rank*sizeof(double));
			}
			ufff.close();
			uffmf.close();
		}
		string vff_fn = fldr + fn + ".VFF.";
		for(int ff = 0; ff < nvff; ff++)
		{
			num_str.str("");
			num_str << ff;
			ofstream vfff((vff_fn+num_str.str()).c_str(),ios::out|ios::binary);
			ofstream vffmf((vff_fn+num_str.str()+".maxes").c_str(),ios::out|ios::binary);
			vffmf.write((char*)vff_maxes[ff],ni*sizeof(int));
			for(int i = 0; i < ni; i++)
			{
				for(int t = 0; t <= vff_maxes[ff][i]+1; t++)
					vfff.write((char*)Vff[ff][i][t],rank*sizeof(double));
			}
			vfff.close();
			vffmf.close();
		}
	}
	
	// write B
	string bfn = fldr + fn + ".B.";
	for(int b = 0; b < nb; b++)
	{
		num_str.str("");
		num_str << b;
		ofstream bf((bfn+num_str.str()).c_str(),ios::out|ios::binary);
		for(int d0=0; d0 < disc_feature_ranges[d0s[b]]; d0++)
			bf.write((char*)B[b][d0],disc_feature_ranges[d1s[b]]*sizeof(double));
		bf.close();
	}
	
	string wfn = fldr + fn + ".W";
	ofstream wf(wfn.c_str(),ios::out|ios::binary);
	wf.write((char*)W,nw*sizeof(double));
	wf.close();
	
}