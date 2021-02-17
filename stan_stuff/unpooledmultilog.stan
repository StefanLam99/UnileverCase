data {
        int K; //Number of possible outcomes/levels
        int N; //Number of observations
        int D; //Number of predictors
        int n_cluster; //Number of clusters
        int y[N];
        matrix[N, D] x; //predictor matrix
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
        matrix[K-1,D] beta_raw[n_cluster]; //Beta coefficients for k groups (without the zeros)
        real<lower=0> sigma[n_cluster];
}

transformed parameters {
        matrix[K, D] beta[n_cluster];

        for (c in 1:n_cluster){
                    beta[c,,] = append_row(zeros,beta_raw[c,,]); 
                  }
        
}


model {
        matrix[N, K] x_beta;
        to_vector(sigma)~gamma(2,1.0/10);
        
        
        for(c in 1:n_cluster){
                if(prior_set){
                        for(k in 1:K-1){
                                for(d in 1:D){
                                        beta_raw[c,k, d] ~ normal(prior_mean[k,d], sigma[c]); //Random prior, with more information we can specify this clearer.
                                }
                        }    
                }else{
                        to_vector(beta_raw[c,,]) ~ normal(0, sigma[c]); //Random prior, with more information we can specify this clearer.
                }
                
                
                
                
                for (n in 1:N){
                        x_beta[n] = x[n,] * (beta[cluster[n],,])';
                        y[n] ~ categorical_logit(x_beta[n]');
                }
        }
        
}

generated quantities {
        int y_pred_insample[N];
        int y_pred_outsample[N_test];
        // vector[K] y_pred_soft[N];
        // real log_lik[N];
        
        matrix[N, K] x_beta_train; 
        matrix[N_test, K] x_beta_test;
        
        if(boolean_test){
                for (n in 1:N_test){
                        x_beta_test[n] = x_test[n,] *(beta[cluster_test[n],,])';
                        y_pred_outsample[n] = categorical_logit_rng(x_beta_test[n]');
                }
        }
        
        for (n in 1:N){
                x_beta_train[n] = x[n,] * (beta[cluster[n],,])';
                y_pred_insample[n] = categorical_logit_rng(x_beta_train[n]');
                // y_pred_soft[n] = softmax(x_beta_train[n]');
                // log_lik[n] = categorical_lpmf(y[n]|softmax(x_beta_train[n]'));
        }
        
}

