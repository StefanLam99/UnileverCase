library('bayesplot')
library('rstan')
library('ggplot2')
library('loo')

bml_nc <-readRDS("./stan_stuff/stan_model_output/bml_nc.rds")

fit <- b_equal.out
fit.ext <- rstan::extract(fit)
fit.mat <-as.matrix(fit)


p_bh_more <- mcmc_trace(as.matrix(fit), pars=vars( "sigma_beta[2]", "beta[3,1,1]", "beta[3,2,1]", "beta[3,3,1]",
                                              "beta[2,1,1]", "beta[2,2,1]", "beta[2,3,1]"))

lp_cp <- log_posterior(fit)
head(lp_cp)

np_cp <- nuts_params(fit)
head(np_cp)

color_scheme_set("darkgray")
mcmc_parcoord(fit.mat, np = np_cp, pars = colnames(fit.mat)[1:39]) + ggplot2::coord_flip()

print(fit,pars = "beta")
plot(fit, pars=  "beta")

log_lik_1 <- extract_log_lik(fit,
                             parameter_name = "log_lik",
                             merge_chains = FALSE)

r_eff1 <- relative_eff(exp(log_lik_1)) 
loo_1 <- loo(log_lik_1, r_eff = r_eff1, cores = 4)

print(loo_1)
plot(loo_1)

loo_function <-function(fit){
  log_lik <- extract_log_lik(fit,
                               parameter_name = "log_lik",
                               merge_chains = FALSE)
  
  r_eff <- relative_eff(exp(log_lik)) 
  loo <- loo(log_lik, r_eff = r_eff, cores = 4)
  loo
}
loo_1 <- loo_function(b_equal.out)
loo_2 <- loo_function(bun.out)
comp <- loo_compare(x = list(loo_1, loo_2))


print(comp)

####predictions

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

prediction_median <- apply(fit.ext$y_pred_insample, 2, median)
prediction_mean <- round(apply(fit.ext$y_pred_insample, 2, mean))
prediction_mode <- apply(fit.ext$y_pred_outsample, 2, getmode)

print(mean(prediction_mode == y_test))
cont_table <- table(prediction_mode, y_test)
cont_table <- (rbind(cont_table, apply(cont_table, 2, sum)))
cont_table <- cbind(cont_table, apply(cont_table,1, sum))
cont_table








fit_summary <-summary(fit)
print(names(fit_summary))
print(fit_summary$summary)

beta_summary <-summary(fit, pars = c("beta"), probs = c(0.1, 0.9))$summary
print(beta_summary)

sampler_params <- get_sampler_params(fit, inc_warmup = FALSE)
sampler_params_chain1 <- sampler_params[[1]]

#Obtain mean acceptance rate
mean_accept_stat_by_chain <- sapply(sampler_params, function(x) mean(x[, "accept_stat__"]))
print(mean_accept_stat_by_chain)

#Obtain max tree depth
max_treedepth_by_chain <- sapply(sampler_params, function(x) max(x[, "treedepth__"]))
print(max_treedepth_by_chain)

#Initial values
inits <- get_inits(fit)
inits_chain1 <- inits[[1]]
print(inits_chain1)

print(get_elapsed_time(fit))

fit <- sampling(fit, data = datlist.unpooled, iter = 1000, chains = 1)


