data {
        int K; //Number of possible outcomes/levels
        int N; //Number of observations
        int D; //Number of predictors
        int n_cluster; //Number of clusters
        int y[N];
        matrix[N, D] x; //predictor matrix
        int<lower=1, upper=N> cluster[N]; //Vector with cluster indices
}

//ik doe nu eigenlijk niks met identifiability, not sure hoe dat in deze context moet

parameters {
        matrix[K,D] beta_raw[n_cluster]; //Beta coefficients for k groups (without the zeros)
        //level 2 errors
        matrix[K,D] error; //het random effect gedeelte
        //covariance matrix
        cov_matrix[D] sigma; //waar error mee wordt getrokken
        //intercept of the level
        matrix[K,D] beta_intercept; //voor elk cluster hetzelfde
}

transformed parameters {
        matrix[K, D] beta[n_cluster];
        beta = beta_raw;
        for (c in 1:n_cluster)
                for (d in 1:D)
                        beta[c, 1, d] = 0;
        // print(beta[cluster[20],,]);
        for(c in 1:n_cluster) {
          beta[c,,] = beta_intercept + error;
        }
          
}

model {
        matrix[N, K] x_beta;
        vector[D] zero;
        zero = rep_vector(0,D);
        for(i in 1:K){
          error[i,] ~ multi_normal(zero, sigma); // zoals in voorbeeld wat Bart stuurde, heel ff checken of het op deze manier dus wel andere waardes blijft trekken voor elk cluster, vgm wel
        }

        //moet vgm blijven staan, maar ik weet niet of 
        for(n in 1:N)
                x_beta[n] = x[n,] * to_matrix(beta[cluster[n],,])';

        //for(c in 1:n_cluster)
                //to_vector(beta_raw[c,,]) ~ normal(0, 10); //Random prior, with more information we can specify this clearer.

        for (n in 1:N)
                y[n] ~ categorical_logit(x_beta[n]');
}
