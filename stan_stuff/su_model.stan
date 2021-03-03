data {
  int K; //Number of possible outcomes/levels
  int N; //Number of observations
  int D; //Number of predictors
  int n_cluster; //Number of clusters
  int y[N];
  matrix[N, D] x; //predictor matrix, maybe row_vector x...
  int<lower=1, upper=n_cluster> cluster[N]; //Vector with cluster indices
  
  real boolean_test; //test or not test
  int n_cluster_test; //number of test clusters
  int N_test; //Number test observations
  matrix[N_test, D] x_test; //test matrix
  int<lower=1, upper=n_cluster_test> cluster_test[N_test]; //Vector with cluster indices  
  
  real prior_set; //Whether or not prior_set is included
  matrix[K-1, D] prior_mean;
}

transformed data {
  row_vector[D] zeros = rep_row_vector(0, D); //Create zero for identification, will later be appended to beta_raw
}

parameters {
  
  matrix[K-1, D] mu;
  
  // cholesky_factor_corr[K-1] Lcorr_beta;  
  // vector<lower=0>[K-1] sigma_beta; 
  
  matrix[(K-1),n_cluster * D] beta_raw;
  
  real<lower=0> sigma[D];
  
  corr_matrix[n_cluster*D] Omega; // prior correlation
  vector<lower=0>[n_cluster*D] tau; // prior scale
}

transformed parameters{
  matrix[K,D] beta[n_cluster];
  
  
  for (c in 1:n_cluster){
    // print("beta raw: ", beta_raw);
    // print("beta raw slice: ",beta_raw[,((c-1)*D+1):D*c]);
    // print("beta raw slice zeros: ", append_row(zeros,beta_raw[,((c-1)*D+1):D*c]));
    beta[c,,] = append_row(zeros,beta_raw[,((c-1)*D+1):D*c]); 
  }
}


model {
  matrix[N, K] x_beta;
  matrix[n_cluster*D,n_cluster*D] Sigma_beta;
  
  Sigma_beta = quad_form_diag(Omega,tau);
  tau ~ cauchy(0,2.5);
  Omega ~ lkj_corr(0.5);
  
  sigma~gamma(2,1.0/10);
  
  for (d in 1:D) {
    
    if(prior_set){
      for(k in 1:K-1){
        mu[k, d] ~ normal(prior_mean[k,d], sigma); //Random prior, with more information we can specify this clearer.
      }
    }else{
      mu[,d] ~ normal(0, sigma[d]);
    }
  }
  // print("mu", mu[,d]); 
  
  for(k in 1:K-1){
    beta_raw[k, ] ~multi_normal(to_vector(append_row(mu[k,], mu[k,])), Sigma_beta);
  }

  for(n in 1:N){ 
    x_beta[n] = x[n,] * to_matrix(beta[cluster[n],,])';
    y[n] ~ categorical_logit(x_beta[n]');
  }
}

generated quantities {
  int y_pred_insample[N];
  int y_pred_outsample[N_test];
  // vector[K] y_pred_soft[N];
  // real log_lik[N];

  if(boolean_test){
    for (n in 1:N_test){
      y_pred_outsample[n] = categorical_logit_rng((x_test[n,] *(beta[cluster_test[n],,])')');
    }
  }
  
  for (n in 1:N){
    y_pred_insample[n] = categorical_logit_rng((x[n,] * (beta[cluster[n],,])')');
    // y_pred_soft[n] = softmax(x_beta_train[n]');
    // log_lik[n] = categorical_lpmf(y[n]|softmax(x_beta_train[n]'));
  }
  
}
