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
WX_interest <- c(W_interest, X_interest)

small_subset <- c(1:500)
##Obtain dataframes that will be used in analysis, assume that missing values have
##been handled in python
y <- factor(y_df$DV)
y <- y[small_subset]
X <- X_df[small_subset,X_interest]
WX <- WX_df[small_subset,WX_interest]
W <- W_df[small_subset,W_interest]
cluster <- WX_df$cluster[small_subset]

datlist <- list(N=nrow(X),           #Nr of obs
                K=length(unique(y)),  #Possible outcomes
                D=ncol(X)+1,            #NR of predictors
                x=cbind(1,X),                 #Predictor Matrix
                y=as.numeric(y))      #Dependent Variable
datlist.unpooled <- list(N=nrow(X),           #Nr of obs
                         K=length(unique(y)),  #Possible outcomes
                         D=ncol(X)+1,            #NR of predictors
                         x=cbind(1,X),                 #Predictor Matrix
                         y=as.numeric(y),  #Dependent Variable)
                         n_cluster = length(unique(cluster)),
                         cluster = cluster
                         )     


####MODEL####
#Estimate multinom model
res.multinom <- multinom(y~., data = X, Hess = TRUE)


# estimate Stan model
b.out <- stan(file='./stan_stuff/multilog.stan',
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

#Estimate Bayesian Model
b.out <- stan(file='./stan_stuff/unpooledmultilog.stan', data = datlist.unpooled, iter = 1000, chains = 1)

# launch_shinystan(b.out)
res.stan <- summary(b.out, par="beta", probs=.5)$summary %>% as.data.frame

df_stan <-function(y,df,n_cluster,res.stan){
  out.stan <- data.frame(beta=rep(c('(Intercept)', colnames(df)), length(levels(y))), 
                         value.stan = res.stan[,1]) %>%
    mutate(stan.std=res.stan[,2]) %>%
    mutate(group=rep(1:n_cluster, each=(length(levels(y)))*(ncol(X)+1))) %>%
    mutate(option=rep(levels(y), each=n_cluster*(ncol(df)+1)))%>%
    mutate(coef= paste0(group, ":",option, ":", beta))%>%
    dplyr::select(-option, -beta, -group)
}
out.stan <- df_stan(y,X,5,res.stan)


#### Run unpooled Multilog with hierarchical prior####

#Estimate Unpooled Multinom hierarchical
#However this is difficult because for some clusters not all levels are available...
# multinom_unpooled <- function(y, df, cluster, current_cluster){
#   res.multinom <- multinom(y[cluster==current_cluster]~., data = df[cluster==current_cluster,])
# }

#Estimate Bayesian Model
b.out <- stan(file='./stan_stuff/unpooledmultiloghierarchical.stan', 
              data = datlist.unpooled, iter = 1000, chains = 1)
b.maxout <- stan(file='./stan_stuff/unpooledmultiloghierarchicalmax.stan', 
                 data = datlist.unpooled, iter = 1000, chains = 1)
# launch_shinystan(b.out)
resmax.stan <- summary(b.maxout, par="beta", probs=.5)$summary %>% as.data.frame

df_stan <-function(y,df,n_cluster,res.stan){
  out.stan <- data.frame(beta=rep(c('(Intercept)', colnames(df)), length(levels(y))), 
                         value.stan = res.stan[,1]) %>%
    mutate(stan.std=res.stan[,2]) %>%
    mutate(group=rep(1:n_cluster, each=(length(levels(y)))*(ncol(X)+1))) %>%
    mutate(option=rep(levels(y), each=n_cluster*(ncol(df)+1)))%>%
    mutate(coef= paste0(group, ":",option, ":", beta))%>%
    dplyr::select(-option, -beta, -group)
}
outmax.stan <- df_stan(y,X,5,resmax.stan)

df.compare <-merge(outmax.stan, out.stan, by="coef", all.y=T)
