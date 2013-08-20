#include <string>
#include <vector>
using namespace std;

#ifndef PMF_RECOMMENDER
#define PMF_RECOMMENDER

namespace pmf
{
	const int IND = 0;
	const int RBF = 1;

	class PMF_recommender
	{
	private:
		double mu;
		double ** U; // user factors
		double ** V; // item factors
		double **** Uff;
		double **** Vff;
		double * W;
		vector<int> w_cols;
		double w_mult;
		vector<double> uff_mults;
		vector<double> vff_mults;
		vector<int> uff_cols;
		vector<int> vff_cols;
		int ** uff_maxes;
		int ** vff_maxes;
		int nuff;
		int nvff;
		double *** B; // array of bias matrices
		vector<string> disc_feature_names;
		vector<int> disc_feature_ranges;
		vector<string> cont_feature_names;
		long rank,nu,ni,nb,nw;
		double ur, vr, lr;
		double *** brs;
		vector<int> bias_prior_types;
		int n_epochs;
		bool avg,verbose,track_train_errs,track_test_errs;
		double seed;
		double sd;
		float u_mult,v_mult;
		float * b_mults;
		int n_samples,n_test_samples,n_disc_features,n_cont_features;
		int * d0s;
		int * d1s;
		int ** train_data;
		float ** train_cont_features;
		int ** test_data;
		float ** test_cont_features;
		vector<double> a_inv;
		vector<double> b_inv;
		double ** b_buff;
		int rmax;
		int rmin;
		
		// Variables for accumulating averages
		// Ua,Va, and Ba are for summing param values
		double ** Ua;
		double ** Va;
		double *** Ba;
		double **** Uffa;
		double **** Vffa;
		double * Wa;
		double mua;
		// Uc, Vc, and Bc are for counting the number of times each param is observed
		int * Uc;
		int * Vc;
		int *** Bc;
		int *** Uffc;
		int *** Vffc;
		
		void update_parameters(int,int,double,int*,float*);
		void init_params();
		void read_init_data_file(int ***, float ***, string, int);
		void read_data_file(int ***, float***,string, int);
		void init_averages();
		void update_averages(int, int, int *);
		void finalize_averages(int);
		void write_double_csv(string, double **, int, int);
		void max_by_bin(int**,int,int,int,int, int * &);
		
	public:
		PMF_recommender(string);
		~PMF_recommender();
		void fit_sgd(bool);
		double predict(int,int,int*,float*);
		double test_pred(int,int,int*,float*);
		double RMSE(string data_fn, long ns);
		double RMSE(int**,float**,long);
		double MAE(int**,float**,long);
		void save_model(string,string);
		void fit_sgd(int **, float**, int, int, double, double, double, double, double, vector<double>, vector<double>, vector<double>, bool, bool, bool);
		void train_user(int, int **, vector<string> &, int, int);
		double kendalls_tau_b(int** data, float ** c_feats, long ns);
		double NDCG(int** data, float ** c_feats, long ns);
	};
}
#endif // PMF_RECOMMENDER