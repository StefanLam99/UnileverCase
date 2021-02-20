library('rstan')
library('shinystan')
library('nnet')
library('tidyr')
library("dplyr")
library('notifier')
library('caret')


rstan_options(auto_write = TRUE)
options(mc.cores = 4)

setwd("C:/Users/bartd/Erasmus/Erasmus_/Jaar 4/Master Econometrie/Seminar/UnileverCase_Conda/")

####DATA####
##LOAD DATA
WX_df_train <- read.csv("./Data/preprocessedData/WX_train.csv")
y_df_train <- read.csv("./Data/preprocessedData/y_train.csv", header = FALSE)
WX_df_test <- read.csv("./Data/preprocessedData/WX_test.csv")
y_df_test <- read.csv("./Data/preprocessedData/y_test.csv", header = FALSE)


##Choose columns of interest (Can also be done in python)
colnames(y_df_train) <- c('name', 'DV')
colnames(y_df_test)<-c('name', 'DV')

X_interest <- c("globalChannel_fastfood", "globalChannel_other", "rating")
WX_interest <- c(X_interest, "INWONER","P_MAN","P_VROUW","P_INW_1524", "P_INW_2544", "P_INW_4564",
                 "P_INW_65PL","AV1_FOOD","AV3_FOOD", "AV5_FOOD","OAD", "P_WE_MIG_A", 
                 "P_NW_MIG_A","GEM_HH_GR", "UITKMINAOW",  "P_HINK_HH", "log_median_inc", "AFS_TREINS","AFS_TRNOVS","AFS_OPRIT" )

table(y_df_train$DV) #1 = OPERATIONAl, 2= PERMANENTLY CLOSED, 3 = TEMPORARILY CLOSED
y_df_train$DV <- factor(y_df_train$DV)

#FUNCTION TO CREATE DATALIST
create_datlist <-  function(subset, subset_prior_mean=c(), with_zip = TRUE, oversample = FALSE){
  y <-  factor(y_df_train$DV)
  y <- y[subset]
  
  if(with_zip){
    WX <- WX_df_train[subset,WX_interest]
    WX_prior <- WX_df_train[subset_prior_mean, WX_interest]
    #Test dataframes
    y_test <- factor(y_df_test$DV)
    WX_test <- WX_df_test[, WX_interest]
  }else{
    WX <- WX_df_train[subset,X_interest]
    WX_prior <- WX_df_train[subset_prior_mean, X_interest]
    #Test dataframes
    y_test <- factor(y_df_test$DV)
    WX_test <- WX_df_test[, X_interest]
  }
  
  
  cluster <- WX_df_train$labels[subset]
  cluster_test <- WX_df_test$labels
  
  if(oversample){
    oversampled<- upSample(x = cbind(cluster, WX), 
                          y = y)
    cluster <- oversampled$cluster
    y <- oversampled$Class
    WX <- oversampled[,!(names(oversampled) %in% c('Class', 'cluster'))]
  }
  
  datlist <- list(N=nrow(WX),           #Nr of obs
                  K=length(unique(y)),  #Possible outcomes
                  D=ncol(WX)+1,            #NR of predictors
                  x=cbind(1,WX),                 #Predictor Matrix
                  y=as.numeric(y),  #Dependent Variable)
                  n_cluster = length(unique(cluster)), #Length of cluster
                  cluster = cluster+1, #Cluster
                  
                  boolean_test = 0, #Whether to test or not
                  generate = 0,
                  
                  y_test = y_test, #dependent test variable (probably not needed for stan but easy for prediction)
                  x_test = cbind(1,WX_test), #Test set
                  N_test = nrow(WX_test), #observations in testset
                  cluster_test = cluster_test+1, #clusters in test set
                  n_cluster_test = length(unique(cluster_test)), #No.  cluster in testset.
                  
                  y_prior = y[subset_prior_mean],
                  WX_prior = WX_prior,
                  prior_set = 0
  )
  
  if(length(subset_prior_mean)>0){
    mean <- coef(multinom(datlist$y_prior~., data = datlist$WX_prior))
    datlist$prior_mean <- (mean)
    
  }else{
    datlist$prior_mean <- matrix(rep( 0, len=(datlist$K-1)*datlist$D), nrow = (datlist$K-1))
  }
  
  datlist
}
#FUNCTION TO ESTIMATE MODELS
#MODEL TYPE: 1 = multilog pooled, 2 = multilog unpooled, 3 = multilog with mu
estimate_model <- function(datlist, model_type, gqs = TRUE, test = TRUE, prior_set =TRUE, save = FALSE, iter = 1000, chains = 4){
  if(test){
    datlist$boolean_test <- 1
  }
  if(gqs){
    datlist$generate <- 1
  }
  if(prior_set){
    datlist$prior_set <- 1
  }
  
  #Run model depending on model type. Optionally save the model
  #1 = multilog pooled, 2 = multilog unpooled, 3 = multilog with mu
  if(model_type == 1){
    b.out <- stan(file='./stan_stuff/multilog.stan',
                  data=datlist,
                  iter = iter,
                  chains = chains,
                  seed = 12591)
    if(save){
      saveRDS(b.out, "./stan_stuff/stan_model_output/multilog.rds")
    }
  }else if (model_type==2){
    b.out <- stan(file='./stan_stuff/unpooledmultilog.stan',
                  data=datlist,
                  iter = iter,
                  chains = chains,
                  seed = 12591)
    if(save){
      saveRDS(b.out, "./stan_stuff/stan_model_output/unpooledmultilog.rds")
    }
  }else if(model_type==3){
    b.out <- stan(file='./stan_stuff/multilog_based_on_mcstan.stan',
                  data=datlist,
                  iter = iter,
                  chains = chains,
                  seed = 12591)
    if(save){
      saveRDS(b.out, "./stan_stuff/stan_model_output/unpooledhierarchical.rds")
    }
  }else if(model_type==4){
    b.out <- stan(file='./stan_stuff/multilog_based_on_mcstan_multinormal.stan',
                  data=datlist,
                  iter = iter,
                  chains = chains,
                  seed = 12591)
  }
  b.out
}

####PREDICTION EVALUATION####
#FUNCTION FOR CONTINGENCY MODEL
#Average type: 1 = median, 2 = mean, 3 = mode
contingency_table <- function(b.out, datlist, average_type = 1, insample=TRUE){
  fit <- b.out
  fit.ext <- rstan::extract(fit)
  
  #In OutSample Prediction
  if(insample){
    prediction <-fit.ext$y_pred_insample
    true_value <-datlist$y
  }else{
    prediction <-fit.ext$y_pred_outsample
    true_value <- datlist$y_test
  }
  
  #Type of meaning prediction
  if(average_type==1){
    prediction <- apply(prediction, 2, median)
  }else if(average_type==2){
    prediction <- round(apply(prediction, 2, mean))
  }else{
    getmode <- function(v) {
      uniqv <- unique(v)
      uniqv[which.max(tabulate(match(v, uniqv)))]
    }
    prediction<- apply(prediction, 2, getmode)
  }

  prediction <- as.factor(prediction)
  true_value <- as.factor(true_value)
  
  (confusionMatrix(prediction, reference=true_value))
}

#FUNCTION FOR Parameter MODEL
#Obtain Parameter tables
parameter_table <- function(b.out, par_interest, clusters = TRUE, var_names){
  fit.ext <- rstan::extract(b.out)
  if(par_interest =="beta"){
    mcmc_int <- fit.ext$beta
  }else if(par_interest=="mu"){
    mcmc_int <- fit.ext$mu
  }else{
    stop("implement this one")
  }
  
  #CLUSTER BETA
  if(clusters & par_interest=='beta'){
    print(dim(mcmc_int))
    ind_coeff <- apply(mcmc_int,c(2,3,4), quantile, probs=c(0.025, 0.5, 0.975))
    
    ind_coeff_2 <- ind_coeff[,,2,]
    ind_coeff_3 <- ind_coeff[,,3,]
    
    df_ind_unpooled <-function(ind_coeff, outcome){
      df_ind_coeff <- data.frame(Coeff=rep(var_names,each=2), 
                                 LI=c(ind_coeff[1,,1:length(var_names)]),
                                 Median=c(ind_coeff[2,,1:length(var_names)]),
                                 HI=c(ind_coeff[3,,1:length(var_names)])
      )
      
      out<-paste(outcome)
      df_ind_coeff$Outcome<-factor(out,levels=out)
      
      gr<-paste("Gr",1:2)
      
      df_ind_coeff$Group<-factor(gr,levels=gr)
      df_ind_coeff
    }
    
    df_beta_2 <- df_ind_unpooled(ind_coeff_2, 2)
    df_beta_3 <- df_ind_unpooled(ind_coeff_3, 3)
    df_ind <- rbind(df_beta_2, df_beta_3) 
  }
  #CLUSTER MU
  else if (par_interest == "mu"){
    print('else if')
    ind_coeff<-apply(mcmc_int,c(2,3), quantile, probs=c(0.025,0.5,0.975))
    df_ind_mu <-function(ind_coeff){
      df_ind_coeff <- data.frame(Coeff=rep(var_names,each=2),
                                 LI=c(ind_coeff[1,,1:length(var_names)]),
                                 Median=c(ind_coeff[2,,1:length(var_names)]),
                                 HI=c(ind_coeff[3,,1:length(var_names)]))
      
      out <- paste(rep(c(2:3), times=length(var_names)/2))
      df_ind_coeff$Outcome <- factor(out, levels=unique(out))
      
      df_ind_coeff
    }
    df_ind <- df_ind_mu(ind_coeff)
  }
  #NO CLUSTER BETA
  else{
    print('else')
    ind_coeff<-apply(mcmc_int,c(2,3), quantile, probs=c(0.025,0.5,0.975))
    
    df_ind_pooled <-function(ind_coeff){
      df_ind_coeff <- data.frame(Coeff=rep(var_names,each=1),
                                 LI=c(ind_coeff[1,2:3,1:length(var_names)]),
                                 Median=c(ind_coeff[2,2:3,1:length(var_names)]),
                                 HI=c(ind_coeff[3,2:3,1:length(var_names)]))
      
      out <- paste(rep(c(2:3), each=length(var_names)))
      df_ind_coeff$Outcome <- factor(out, levels=unique(out))
      
      df_ind_coeff
    }
    
    df_ind <- df_ind_pooled(ind_coeff)
  }
  df_ind
}

obtain_output <- function(n_prior = 3000, n_observations =1000, restaurant_only = TRUE, oversample = FALSE, prior_set=FALSE,
                          model_num = 1, iter = 2000, chains = 4){
  output <- list()
  
  #Equal sample
  set.seed(5)
  sample_length = round(n_observations/3)
  ind_operational <- which(y_df_train$DV==1, arr.ind=TRUE)
  ind_temp <- which(y_df_train$DV==3, arr.ind=TRUE)
  ind_perm <- which(y_df_train$DV==2, arr.ind=TRUE)
  
  #Indices for other kind of datsets
  small_subset <- c(n_prior+1:n_prior+n_observations)
  if (prior_set){
    subset_prior_mean <- c(1:n_prior)
  }else{
    subset_prior_mean <-c()
  }
  
  small_subset_over <- c(sample(ind_perm, sample_length), 
                             ind_temp,
                             sample(ind_operational, sample_length))
  small_subset_undersampling <- c(sample(ind_perm, length(ind_temp)), 
                                  ind_temp,
                                  sample(ind_operational, length(ind_temp)))
  # small_subset_oversampling #TODO
  full_dataset <- c(ind_operational, ind_temp, ind_perm)
  
  if(oversample){
    subset <- small_subset_over
  }else if(n_observations == nrow(WX_df_train)){
    subset <- full_dataset
  }else{
    subset <- small_subset
  }
  
  if(restaurant_only){
    output$varnames <- c("INTERCEPT", X_interest)
    output$datlist <- create_datlist(subset, subset_prior_mean = subset_prior_mean, with_zip = FALSE, oversample)
    output$model <- estimate_model(output$datlist, model_type = model_num, prior_set=FALSE, iter=iter, chains=chains)
  }else{
    output$varnames <- c("INTERCEPT", WX_interest)
    output$datlist <- create_datlist(subset, subset_prior_mean = subset_prior_mean, with_zip = TRUE, oversample)
    output$model <- estimate_model(output$datlist, model_type = model_num, prior_set=FALSE, iter=iter, chains=chains)
  }
  
  output$cont_insample <- contingency_table(output$model, datlist = output$datlist, average_type = 3, insample=TRUE)
  output$cont_outsample <-contingency_table(output$model, datlist = output$datlist, average_type = 3, insample=FALSE)
  
  if(model_num==1){
    output$beta <- parameter_table(output$model, "beta", clusters=FALSE, var_names = output$varnames)
  }else if(model_num==2){
    output$beta <- parameter_table(output$model, "beta", clusters=TRUE, var_names = output$varnames)
  }else if (model_num==3){
    output$beta <- parameter_table(output$model, "beta", clusters=TRUE, var_names = output$varnames)
    output$mu <- parameter_table(output$model, "mu", clusters=TRUE, var_names = output$varnames)
  }else if (model_num==4){
    output$beta <- parameter_table(output$model, "beta", clusters=TRUE, var_names = output$varnames)
    output$mu <- parameter_table(output$model, "mu", clusters=TRUE, var_names = output$varnames)
  }
  output
}
# MAX
# rest_over_1.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = TRUE, oversample = TRUE, model_num = 1, iter = 2000, chains = 4)
# rest_over_2.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = TRUE, oversample = TRUE, model_num = 2, iter = 2000, chains = 4)
# # rest_over_3.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = TRUE, oversample = TRUE, model_num = 3, iter = 2000, chains = 4)
# rest_over_4.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = TRUE, oversample = TRUE, model_num = 4, iter = 2000, chains = 4)
# 
# 
# all_over_1.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = FALSE, oversample = TRUE, model_num = 1, iter = 2000, chains = 4)
# all_over_2.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = FALSE, oversample = TRUE, model_num = 2, iter = 2000, chains = 4)
# all_over_3.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = FALSE, oversample = TRUE, model_num = 3, iter = 2000, chains = 4)
# all_over_4.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = FALSE, oversample = TRUE, model_num = 4, iter = 2000, chains = 4)

#BART
print('rest1')
rest_norm_1.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = TRUE, oversample = FALSE, model_num = 1, iter = 2000, chains = 4)
print('rest2')
rest_norm_2.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = TRUE, oversample = FALSE, model_num = 2, iter = 2000, chains = 4)
# rest_norm_3.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = TRUE, oversample = FALSE, model_num = 3, iter = 2000, chains = 4)
print('rest4')
rest_norm_4.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = TRUE, oversample = FALSE, model_num = 4, iter = 2000, chains = 4)

print('all1')
all_norm_1.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = FALSE, oversample = FALSE, model_num = 1, iter = 2000, chains = 4)
print('all2')
all_norm_2.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = FALSE, oversample = FALSE, model_num = 2, iter = 2000, chains = 4)
# all_norm_3.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = FALSE, oversample = FALSE, model_num = 3, iter = 2000, chains = 4)
print('all4')
all_norm_4.output <- obtain_output(n_prior = 3000, n_observations = 3000, restaurant_only = FALSE, oversample = FALSE, model_num = 4, iter = 2000, chains = 4)


# #Additional possibly interesting diagnostics...
# temp <- rstan::extract(output_2$model)
# #If we Include a model about the data distribution corresponding to the plot below we have a nice argument to the 
# #Bootstrapping...
# 
# #Interesting plot of the distribution
# ppc_stat_grouped(output_2$datlist$y, temp$y_pred_insample, group = output_2$datlist$y )
# 
# #For some diagnostic plots
# launch_shinystan(output_2$model)

