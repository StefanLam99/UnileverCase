data {
        int K; //Number of possible outcomes/levels
        int N; //Number of observations
        int D; //Number of predictors
        int n_cluster; //Number of clusters
        int y[N];
        matrix[N, D] x; //predictor matrix, maybe row_vector x...
        int<lower=1, upper=n_cluster> cluster[N]; //Vector with cluster indices
}

parameters {
  matrix[K, D] mu;
  
  cholesky_factor_corr[K] Lcorr_beta;  
  vector<lower=0>[K] sigma_beta; 

  matrix[K,D] beta[n_cluster];
}



model {
  matrix[N, K] x_beta;
// 
  sigma_beta ~ cauchy(0,100);
  Lcorr_beta ~ lkj_corr_cholesky(100);
  
  for (d in 1:D) {
    mu[,d] ~ normal(0, 20);
    // print("mu", mu[,d]); 
    for (c in 1:n_cluster){
       // beta[c, ,d] ~ normal(mu[,d], sigma[d]);
      // beta[c, ,d] ~ normal(mu[,d], 10);
      beta[c,,d] ~ multi_normal_cholesky(mu[,d], diag_pre_multiply(sigma_beta,Lcorr_beta));
      // print(beta[c,,d]); 
    }
     
  }
 for(n in 1:N){ 
          x_beta[n] = x[n,] * to_matrix(beta[cluster[n],,])';
          y[n] ~ categorical_logit(x_beta[n]');
 }
}
