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
  matrix[K,D] mu_raw[n_cluster];
  real<lower=0> sigma[n_cluster];

  matrix[K, D] error_raw[n_cluster];
}
transformed parameters{
  matrix[K,D] beta[n_cluster];
  matrix[K,D] mu[n_cluster];
  matrix[K,D] error[n_cluster];

  //Identify mu and error. TODO try this with identifiable priors. 
  mu = mu_raw;
  error = error_raw;
  for(c in 1:n_cluster)
      for(d in 1:D){
          error[c, 1, d] = 0;
          mu[c,1,d] = 0;
      }
      



  for(c in 1:n_cluster)
    beta[c,,] = mu[c,,] + error[c,,];
  // print("mu: ", mu);
  // print("beta: ", beta);
}

model {
  matrix[N, K] x_beta;
  for(n in 1:N)
          x_beta[n] = x[n,] * to_matrix(beta[cluster[n],,])';

  
  
  for(c in 1:n_cluster){
    to_vector(error_raw[c,,]) ~ normal(0, sigma[c]);
    to_vector(mu_raw[c,,]) ~ normal(0,10);
  }

  for (n in 1:N)
          y[n] ~ categorical_logit(x_beta[n]');
}