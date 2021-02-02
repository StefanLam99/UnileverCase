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

small_subset <- c(1:nrow(y_df))
##Obtain dataframes that will be used in analysis, assume that missing values have
##been handled in python
y <- factor(y_df$DV)
y <- y[small_subset]
X <- X_df[small_subset,X_interest]
WX <- WX_df[small_subset,WX_interest]
W <- W_df[small_subset,W_interest]

datlist <- list(N=nrow(X),           #Nr of obs
                K=length(unique(y)),  #Possible outcomes
                D=ncol(X)+1,            #NR of predictors
                x=cbind(1,X),                 #Predictor Matrix
                y=as.numeric(y))      #Dependent Variable

####MODEL####
#Estimate multinom model
res.multinom <- multinom(y~., data = X, Hess = TRUE)


#estimate Stan model
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
