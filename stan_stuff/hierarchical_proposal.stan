data {
        int K; //Number of possible outcomes/levels
        int N; //Number of restaurant observations
        int D; //Number of restaurant predictors
        int Z; //Numbre of zipcode predictors
        int n_zip; //Number of zipcodes
        int n_cluster; //Number of clusters
        int y[N];
        matrix[N, D] x; //predictor matrix, maybe row_vector x...
        matrix[n_zip, Z] w; 
        int<lower=1, upper=n_cluster> cluster[N]; //Vector with cluster indices
        int<lower=1, upper=n_zip> zip[N]; //Vector wit zipcode indices
        int<lower=1, upper=n_zip> cluster_for_zip[n_zip]; //Vector with cluster index for zipcode
}

parameters {
  matrix[K,Z] alpha[n_cluster];
  
  matrix[K, D] beta_raw[n_cluster];
  
  matrix[Z,Z] ind_gamma; //Misschien een normale
  cov_matrix[D] ind_beta;
  
  real<lower=0> sigma[n_cluster];
  
  matrix[K, D] error_raw[n_cluster];
  
  cholesky_factor_corr[Z] Lcorr_gamma;  
  vector<lower=0>[Z] sigma_gamma; 
  
  cholesky_factor_corr[D] Lcorr_beta;  
  vector<lower=0>[Z] sigma_beta; 
}

transformed parameters{
  matrix[K, Z] gamma[n_cluster]; 
   
  matrix[K,D] beta[n_cluster];
  beta = beta_raw;
  
  //Identify mu and error. TODO try this with identifiable priors. 
  for(c in 1:n_cluster)
      for(d in 1:D){
          beta[c, 1, d] = 0;
          }
  // for (z in 1:n_zip){
  //   gamma[z,,] = alpha[cluster_for_zip[z],,] + ind_gamma;
  // }
  // for(c in 1:n_cluster)
  //     beta[c,,] = mu[c,,] + error[c,,];
  // // print("mu: ", mu);
  // // print("beta: ", beta);
}

model {
  matrix[N, K] x_beta;
  #Create cholesky matrices
  sigma_gamma ~ cauchy(0,10);
  Lcorr_gamma ~ lkj_corr_cholesky(10);
  
  sigma_beta ~ cauchy(0,10);
  Lcorr_beta ~ lkj_corr_cholesky(10);
  
  
  ##Sample
  #Sample alpha
  for(c in 1:n_cluster){
    to_vector(alpha[c,,]) ~ normal(0, 100);
  }
  #Sample Gamma
  for(z in 1:n_zip)
    for(k in 1:K){
      to_vector(gamma[z,k,]) ~ multi_normal(alpha[cluster_for_zip[z],k,], 
                                          diag_pre_multiply(sigma_gamma, Lcorr_gamma));
    }
  #Sample beta
  for(n in 1:N)
    for(k in 1:K){
      to_vector(beta_raw[n,k,]) ~ multi_normal(w[n]*gamma[zip[n]], 
                                    diag_pre_multiply(sigma_beta, Lcorr_beta));
    }
   
  for(n in 1:N)
          x_beta[n] = x[n,] * to_matrix(beta[zip[n],,])';

  for (n in 1:N)
          y[n] ~ categorical_logit(x_beta[n]');
}