data {
        int K; //Number of possible outcomes/levels
        int N; //Number of observations
        int D; //Number of predictors
        int n_cluster; //Number of clusters
        int y[N];
        matrix[N, D] x; //predictor matrix, maybe row_vector x...
        int<lower=1, upper=n_cluster> cluster[N]; //Vector with cluster indices
}

transformed data {
        row_vector[D] zeros = rep_row_vector(0, D); //Create zero for identification, will later be appended to beta_raw
}

parameters {
  
  matrix[K-1, D] mu;
  
  // cholesky_factor_corr[K-1] Lcorr_beta;  
  // vector<lower=0>[K-1] sigma_beta; 

  matrix[K-1,D] beta_raw[n_cluster];
  
  real<lower=0> sigma[D];
  real<lower=0> sigma_beta[D];
}
transformed parameters{
  matrix[K,D] beta[n_cluster];
  
  for (c in 1:n_cluster){
              beta[c,,] = append_row(zeros,beta_raw[c,,]); 
      }
}


model {
  matrix[N, K] x_beta;
// 
  // sigma_beta ~ cauchy(0,10);
  // Lcorr_beta ~ lkj_corr_cholesky(50);
  sigma~gamma(5,5);
  sigma_beta ~gamma(5,5);
  
  for (d in 1:D) {
    mu[,d] ~ normal(0, sigma[d]);
    // print("mu", mu[,d]); 
    for (c in 1:n_cluster){
       beta_raw[c, ,d] ~ normal(mu[,d], sigma_beta[d]);
      // beta[c, ,d] ~ normal(mu[,d], 10);
      // beta_raw[c,,d] ~ multi_normal_cholesky(mu[,d], diag_pre_multiply(sigma_beta, Lcorr_beta));
      // print(beta[c,,d]); 
    }
     
  }
for(n in 1:N){ 
          x_beta[n] = x[n,] * to_matrix(beta[cluster[n],,])';
          y[n] ~ categorical_logit(x_beta[n]');
 }
}

generated quantities {
        int y_pred_insample[N];
        vector[K] y_pred_soft[N];
        real log_lik[N];
        
        matrix[N, K] x_beta_train; 
        
        for (n in 1:N){
                x_beta_train[n] = x[n,] * (beta[cluster[n],,])';
                y_pred_insample[n] = categorical_logit_rng(x_beta_train[n]');
                y_pred_soft[n] = softmax(x_beta_train[n]');
                log_lik[n] = categorical_lpmf(y[n]|softmax(x_beta_train[n]'));
        }

}
