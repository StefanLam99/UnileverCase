library('rstan')
library('shinystan')
library('nnet')
library('tidyr')
library("dplyr")
library('notifier')


rstan_options(auto_write = TRUE)
options(mc.cores = 4)

setwd("C:/Users/bartd/Erasmus/Erasmus_/Jaar 4/Master Econometrie/Seminar/UnileverCase_Conda/")

####DATA####
##LOAD DATA
X_df <- read.csv("./Data/preprocessedData/X.csv")
WX_df <- read.csv("./Data/preprocessedData/WX.csv")
W_df <- read.csv("./Data/preprocessedData/W.csv")
y_df <- read.csv("./Data/preprocessedData/y.csv", header = FALSE)

##Choose columns of interest (Can also be done in python)
colnames(y_df) <- c('name', 'DV')
W_interest <- c("INWONER", "higher_education")
X_interest <- c("globalChannel_fastfood", "globalChannel_other", "rating")
# WX_interest <- c(W_interest, X_interest)
WX_interest <- X_interest

small_subset <- c(1:200)
##Obtain dataframes that will be used in analysis, assume that missing values have
##been handled in python
y <- factor(y_df$DV)
y <- y[small_subset]
X <- X_df[small_subset,X_interest]
WX <- WX_df[small_subset,WX_interest]
W <- W_df[small_subset,W_interest]
cluster <- WX_df$cluster[small_subset]

cluster_for_zip <- W_df$cluster
zip_level <- X_df$level_zip[small_subset]

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

create_datlist(small_subset, with_zip=FALSE, oversample=FALSE)


W <- W_df[,W_interest]
datlist.proposal <- list(N=nrow(X),           #Nr of obs
                         K=length(unique(y)),  #Possible outcomes
                         D=ncol(X)+1,            #NR of predictors
                         Z=ncol(W)+1,
                         n_zip=nrow(W),
                         n_cluster=length(unique(cluster_for_zip)),
                         x=cbind(1,X),                 #Predictor Matrix
                         w=cbind(1,W),
                         y=as.numeric(y),  #Dependent Variable)
                         zip=zip_level,
                         cluster_for_zip=cluster_for_zip
) 


####MODEL####
#Estimate multinom model
res.multinom <- multinom(y~., data = X, Hess = TRUE)


# estimate Stan model
bgam_cons.out <- stan(file='./stan_stuff/multilog.stan',
              data=datlist,
              iter = 1000,
              chains = 2,
              seed = 12591)

# launch_shinystan(b.out)
res.stan <- summary(b.out, par="beta", probs=.5)$summary %>% as.data.frame

compare_multinom_stan <- function(y, df, res.stan, res.multinom){
  #Store output to compare later on:
  out.multinom <- tidyr::gather(as.data.frame(coef(res.multinom)), values) %>%
    mutate(multinom.std=gather(as.data.frame(summary(res.multinom)$standard.error))$value) %>%
    mutate(option=rep(row.names(coef(res.multinom)), ncol(df)+1)) %>%
    mutate(coef= paste0(option, ":", values))%>%
    dplyr::select(-option, -values)%>%
    rename(multinom = value)
  
  #store
  out.stan <- data.frame(beta=rep(c('(Intercept)', colnames(df)), length(levels(y))), 
                         value.stan = res.stan[,1]) %>%
    mutate(stan.std=res.stan[,2]) %>%
    mutate(option=rep(levels(y), each=ncol(df)+1))%>%
    mutate(coef= paste0(option, ":", beta))%>%
    dplyr::select(-option, -beta)
  
  #compare
  merge(out.multinom, out.stan, by="coef", all.y=T)
}

compare_multinom_stan(y, X, res.stan, res.multinom)

#### Run unpooled Multilog ####

#Estimate Unpooled Multinom
#However this is difficult because for some clusters not all levels are available...
multinom_unpooled <- function(y, df, cluster, current_cluster){
  res.multinom <- multinom(y[cluster==current_cluster]~., data = df[cluster==current_cluster,])
}
multinom.out <- multinom_unpooled(y,X,cluster,3)

#Estimate Bayesian Model
b8.out <- stan(file='./stan_stuff/multilog_based_on_mcstan.stan', data = datlist.unpooled, iter = 1000, chains = 1)
#B1 out was original, same parameters as multinom but lower se
#B2 out is addition of mu matrix of dimension KDN_C takes very long, somehow different parameters even tho mu = 0
#B3 is addition fo mu rowvector 
#B4 is mu rowvector that is not zero
#B5 mu and error cluster dependent, should be same as b1. 

# launch_shinystan(b.out)
resunpooled.stan <- summary(b.out, par="beta", probs=.5)$summary %>% as.data.frame

df_stan <-function(y,df,n_cluster,res.stan){
  out.stan <- data.frame(beta=rep(c('(Intercept)', colnames(df)), length(levels(y))), 
                         value.stan = res.stan[,1]) %>%
    mutate(stan.std=res.stan[,2]) %>%
    mutate(group=rep(1:n_cluster, each=(length(levels(y)))*(ncol(X)+1))) %>%
    mutate(option=rep(rep(levels(y), each=(ncol(df)+1)),n_cluster))%>%
    mutate(coef= paste0(group, ":",option, ":", beta))%>%
    dplyr::select(-option, -beta, -group)
  
}
outmultilog.stan <- df_stan(y,X,2,resunpooled.stan)


#### Run unpooled Multilog with hierarchical prior####

#Estimate Unpooled Multinom hierarchical
#However this is difficult because for some clusters not all levels are available...
# multinom_unpooled <- function(y, df, cluster, current_cluster){
#   res.multinom <- multinom(y[cluster==current_cluster]~., data = df[cluster==current_cluster,])
# }

#Estimate Bayesian Model
bhier.out <- stan(file='./stan_stuff/unpooledmultiloghierarchical.stan', 
              data = datlist.unpooled, iter = 1000, chains = 1, control=list(adapt_delta = 0.85))
# b.maxout <- stan(file='./stan_stuff/unpooledmultiloghierarchicalmax.stan', 
#                  data = datlist.unpooled, iter = 1000, chains = 1)
launch_shinystan(b.out)
res.stan <- summary(bhier.out, par="beta", probs=.5)$summary %>% as.data.frame

df_stan <-function(y,df,n_cluster,res.stan){
  out.stan <- data.frame(beta=rep(c('(Intercept)', colnames(df)), length(levels(y))), 
                         value.stan = res.stan[,1]) %>%
    mutate(stan.std=res.stan[,2]) %>%
    mutate(group=rep(1:n_cluster, each=(length(levels(y)))*(ncol(X)+1))) %>%
    mutate(option=rep(rep(levels(y), each=(ncol(df)+1)),n_cluster))%>%
    mutate(coef= paste0(group, ":",option, ":", beta))%>%
    dplyr::select(-option, -beta, -group)
  
}
out.stan <- df_stan(y,X,2,res.stan)

df.compare <-merge(outmultilog.stan, out.stan, by.x="coef")


#Sample hierarchical model from proposal
bproposal.out <-stan(file='./stan_stuff/hierarchical_proposal.stan', 
                     data = datlist.proposal, iter = 1000, chains = 1, control=list(adapt_delta = 0.85))
-