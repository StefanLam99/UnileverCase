data {
        int K; //Number of possible outcomes/levels
        int N; //Number of observations
        int D; //Number of predictors
        int y[N];
        matrix[N, D] x; //predictor matrix
        
        real boolean_test; //To test or not to test
        int N_test; //Number test observations
        matrix[N_test, D] x_test; //test matrix
        
        real prior_set; //Whether or not prior_set is included
        matrix[K-1, D] prior_mean;
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
        sigma ~ gamma(2,1.0/10);
        
        if(prior_set){
                for(k in 1:K-1){
                        for(d in 1:D){
                                beta_raw[k, d] ~ normal(prior_mean[k,d], sigma); //Random prior, with more information we can specify this clearer.
                        }
                }    
        }else{
                to_vector(beta_raw)~normal(0, sigma);
        }
        
        
        for (n in 1:N)
        y[n] ~ categorical_logit(x_beta[n]');
}

generated quantities {

                int y_pred_insample[N];
                int y_pred_outsample[N_test];
                
                matrix[N, K] x_beta_train = x * beta';
                
                if(boolean_test){
                        matrix[N_test, K] x_beta_test = x_test * beta';
                        for (n in 1:N_test){
                                y_pred_outsample[n] = categorical_logit_rng(x_beta_test[n]');
                        }
                }
                
                
                for (n in 1:N){
                        y_pred_insample[n] = categorical_logit_rng(x_beta_train[n]');
                }

        
}
