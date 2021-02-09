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
# X_df <- read.csv("./Data/preprocessedData/X.csv")
WX_df <- read.csv("./Data/preprocessedData/WX.csv")
# W_df <- read.csv("./Data/preprocessedData/W.csv")
y_df <- read.csv("./Data/preprocessedData/y.csv", header = FALSE)

##Choose columns of interest (Can also be done in python)
colnames(y_df) <- c('name', 'DV')
W_interest <- c("INWONER", "higher_education")
X_interest <- c("globalChannel_fastfood", "globalChannel_other", "rating")
WX_interest <- c("INWONER", "GEM_HH_GR", "AV5_HORECA", "OAD", 
                 "median_inc", "globalChannel_fastfood", "globalChannel_other", 
                 "rating")

small_subset <- c(1:23042)
##Obtain dataframes that will be used in analysis, assume that missing values have
##been handled in python
y <- factor(y_df$DV)
y <- y[small_subset]
# X <- X_df[small_subset,X_interest]
WX <- WX_df[small_subset,WX_interest]
# W <- W_df[small_subset,W_interest]
cluster <- WX_df$cluster[small_subset]

datlist <- list(N=nrow(WX),           #Nr of obs
                K=length(unique(y)),  #Possible outcomes
                D=ncol(WX)+1,            #NR of predictors
                x=cbind(1,WX),                 #Predictor Matrix
                y=as.numeric(y))      #Dependent Variable
datlist.unpooled <- list(N=nrow(WX),           #Nr of obs
                         K=length(unique(y)),  #Possible outcomes
                         D=ncol(WX)+1,            #NR of predictors
                         x=cbind(1,WX),                 #Predictor Matrix
                         y=as.numeric(y),  #Dependent Variable)
                         n_cluster = length(unique(cluster)),
                         cluster = cluster
)  

#Started running at 23:06
# estimate Stan model
bml.out <- stan(file='./stan_stuff/multilog.stan',
              data=datlist,
              iter = 2000,
              chains = 3,
              seed = 12591)
saveRDS(bml.out, "./stan_stuff/stan_model_output/bml.rds")
bun.out <- stan(file='./stan_stuff/unpooledmultilog.stan', 
                data = datlist.unpooled, 
                iter = 2000, 
                chains = 3, 
                seed = 12591)
saveRDS(bun.out, "./stan_stuff/stan_model_output/bun.rds")

bh.out <- stan(file='./stan_stuff/multilog_based_on_mcstan.stan', 
               data = datlist.unpooled, 
               iter = 2000, chains = 3, seed = 12591)
saveRDS(bh.out, "./stan_stuff/stan_model_output/bh.rds")