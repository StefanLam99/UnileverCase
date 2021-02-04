data {
        int K; //Number of possible outcomes/levels
        int N; //Number of observations
        int D; //Number of predictors
        int n_cluster; //Number of clusters
        int y[N];
        matrix[N, D] x; //predictor matrix
        int<lower=1, upper=n_cluster> cluster[N]; //Vector with cluster indices
}


parameters {
        matrix[K,D] beta_raw[n_cluster]; //Beta coefficients for k groups (without the zeros)
}

transformed parameters {
        matrix[K, D] beta[n_cluster];
        beta = beta_raw;
        for (c in 1:n_cluster)
                for (d in 1:D)
                        beta[c, 1, d] = 0;
        // print(beta[cluster[20],,]);
}


model {
        matrix[N, K] x_beta;
        for(n in 1:N)
                x_beta[n] = x[n,] * (beta[cluster[n],,])';

        for(c in 1:n_cluster)
                to_vector(beta_raw[c,,]) ~ normal(0, 10); //Random prior, with more information we can specify this clearer.

        for (n in 1:N)
                y[n] ~ categorical_logit(x_beta[n]');
}
