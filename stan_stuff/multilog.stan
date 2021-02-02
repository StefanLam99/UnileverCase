data {
        int K;
        int N;
        int D;
        int y[N];
        matrix[N, D] x;
}


transformed data {
        row_vector[D] zeros = rep_row_vector(0, D);
}

parameters {
        matrix[K-1, D] beta_raw;
}

transformed parameters {
        matrix[K, D] beta;
        beta = append_row(zeros,beta_raw);
}

model {
        matrix[N, K] x_beta = x * beta';
        
        to_vector(beta_raw) ~ normal(0, 10);
        
        for (n in 1:N)
                y[n] ~ categorical_logit(x_beta[n]');
}

