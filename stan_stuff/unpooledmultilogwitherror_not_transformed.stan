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

  real<lower=0> sigma[n_cluster];

  real mu_raw_hat[n_cluster];
  
  matrix[K,D] mu[n_cluster];
  matrix[K,D] error[n_cluster];
}
transformed parameters{
  matrix[K,D] beta[n_cluster];
  
  for(c in 1:n_cluster)
      beta[c,,] = mu[c,,] + mu_raw_hat[c] * error[c,,];
  // print("mu: ", mu);
  // print("beta: ", beta);
}

model {
  matrix[N, K] x_beta;
  for(n in 1:N)
          x_beta[n] = x[n,] * to_matrix(beta[cluster[n],,])';

  mu_raw_hat ~ normal(0,1);

  for(c in 1:n_cluster){
    to_vector(error[c,,]) ~ normal(0, sigma[c]);
    to_vector(mu[c,,]) ~ normal(0,10);}

  for (n in 1:N)
          y[n] ~ categorical_logit(x_beta[n]');
}
