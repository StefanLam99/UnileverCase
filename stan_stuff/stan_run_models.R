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

#SCALE
WX_df$INWONER_scaled <- as.array(scale(WX_df$INWONER), col.names = "INWONER_scaled")
WX_df$medianinc_scaled <- as.array(scale(WX_df$INWONER), col.names = "median_inc")

##Choose columns of interest (Can also be done in python)
colnames(y_df) <- c('name', 'DV')
W_interest <- c("INWONER", "higher_education")
X_interest <- c("globalChannel_fastfood", "globalChannel_other", "rating")
WX_interest <- c("AV5_HORECA", "globalChannel_fastfood", "globalChannel_other", 
                 "rating")
WX_interest <- c("INWONER", "GEM_HH_GR", "AV5_HORECA")
# WX_interest<- c("INWONER", "GEM_HH_GR", "AV5_HORECA", "OAD", "median_inc",
#                 "globalChannel_fastfood", "globalChannel_other",
#                 "rating")
WX_interest <- c(X_interest, "GEM_HH_GR", "INWONER_scaled", 'medianinc_scaled')
# WX_interest<-c(X_interest)

small_subset <- c(1:1000)

#Equal sample
ind_operational <- which(y_df$DV=="OPERATIONAL", arr.ind=TRUE)
ind_perm <- which(y_df$DV=="CLOSED_PERMANENTLY", arr.ind=TRUE)
ind_temp<- which(y_df$DV=="CLOSED_TEMPORARILY", arr.ind=TRUE)
small_subset <- c(ind_perm[1:500], ind_temp, sample(ind_operational, length(ind_perm))[1:500])

##Obtain dataframes that will be used in analysis, assume that missing values have
##been handled in python
y <- factor(y_df$DV)
y <- y[small_subset]
# X <- X_df[small_subset,X_interest]
WX <- WX_df[small_subset,WX_interest]
# W <- W_df[small_subset,W_interest]
cluster <- sample(1:3, length(small_subset), replace=TRUE)
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

# estimate Stan model
b_equal.out <- stan(file='./stan_stuff/multilog.stan',
              data=datlist,
              iter = 500,
              chains = 4,
              seed = 12591)
saveRDS(bml.out, "./stan_stuff/stan_model_output/bml_larger.rds")

bml_nc_fewpar.out <- stan(file='./stan_stuff/ml_constrained_non_centered.stan',
                   data=datlist,
                   iter = 500,
                   chains = 4,
                   seed = 12591)
saveRDS(bml_nc_fewpar.out, "./stan_stuff/stan_model_output/bml_nc_fewpar.rds")

bun.out <- stan(file='./stan_stuff/unpooledmultilog.stan', 
                data = datlist.unpooled, 
                iter = 500, 
                chains = 3, 
                seed = 12591)
saveRDS(bun.out, "./stan_stuff/stan_model_output/bun.rds")

bh_identified_more.out <- stan(file='./stan_stuff/multilog_based_on_mcstan.stan', 
               data = datlist.unpooled, 
               iter = 500, chains = 3, seed = 12591)
saveRDS(bh_identified.out, "./stan_stuff/stan_model_output/bh.rds")


#### Prediction ####
N <- max(small_subset)
N_train <- N0.8
N_test <- N0.2
train_ind <- sample(c(1:N), size = N_train, replace = FALSE)
x_train <- WX[train_ind,]
x_test <- WX[-train_ind,]
y_train <- y[train_ind]
y_test <- y[-train_ind]

datlist <- list(N=N_train,           #Nr of obs
                N_test = N_test,
                K=length(unique(y)),  #Possible outcomes
                D=ncol(WX)+1,            #NR of predictors +1 for intercept
                x=cbind(1,x_train), #Predictor Matrix for training
                x_test_stan = cbind(1,x_test), #Predictor Matrix for training
                y=y_train)      #Dependent Variable