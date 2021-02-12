data {
        int K; //Number of possible outcomes/levels
        int N; //Number of observations
        int D; //Number of predictors
        int y[N];
        matrix[N, D] x; //predictor matrix
}


transformed data {
        row_vector[D] zeros = rep_row_vector(0, D); //Create zero for identification, will later be appended to beta_raw
}

parameters {
        matrix[K-1, D] beta_raw; //Beta coefficients for k-1 groups (without the zeros)
        real<lower=0> sigma;
}

transformed parameters {
        matrix[K, D] beta;
        beta = append_row(zeros,beta_raw); //Create final beta matrix
}

model {
        matrix[N, K] x_beta = x * beta';
        sigma ~ gamma(2,2);
        to_vector(beta_raw) ~ normal(0, sigma); //Random prior, with more information we can specify this clearer.
        for (n in 1:N)
                y[n] ~ categorical_logit(x_beta[n]');
}

generated quantities {
        int y_pred_insample[N];
        vector[K] y_pred_soft[N];
        real log_lik[N];
        
        matrix[N, K] x_beta_train = x * beta';
        for (n in 1:N){
          y_pred_insample[n] = categorical_logit_rng(x_beta_train[n]');
          y_pred_soft[n] = softmax(x_beta_train[n]');
          log_lik[n] = categorical_lpmf(y[n]|softmax(x_beta_train[n]'));
        }

}
