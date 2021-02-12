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
WX_df_train <- read.csv("./Data/preprocessedData/WX_tr_re.csv")
y_df_train <- read.csv("./Data/preprocessedData/y_tr_re.csv", header = FALSE)
WX_df_test <-read.csv("./Data/preprocessedData/WX_test.csv")
y_df_test <- read.csv("./Data/preprocessedData/y_test.csv", header = FALSE)

#SCALE
WX_df_train$INWONER_scaled <- as.array(scale(WX_df_train$INWONER), col.names = "INWONER_scaled")
WX_df_train$medianinc_scaled <- as.array(scale(WX_df_train$INWONER), col.names = "median_inc")

WX_df_test$INWONER_scaled <- as.array(scale(WX_df_test$INWONER), col.names = "INWONER_scaled")
WX_df_test$medianinc_scaled <- as.array(scale(WX_df_test$INWONER), col.names = "median_inc")

##Choose columns of interest (Can also be done in python)
colnames(y_df_train) <- c('name', 'DV')
colnames(y_df_test)<-c('name', 'DV')
W_interest <- c("INWONER", "higher_education")
X_interest <- c("globalChannel_fastfood", "globalChannel_other", "rating")
WX_interest <- c("INWONER", "GEM_HH_GR", "AV5_HORECA", "OAD", 
                 "median_inc", "globalChannel_fastfood", "globalChannel_other", 
                 "rating")

WX_interest <- c("INWONER", "GEM_HH_GR", "AV5_HORECA")
# WX_interest<- c("INWONER", "GEM_HH_GR", "AV5_HORECA", "OAD", "median_inc",
#                 "globalChannel_fastfood", "globalChannel_other",
#                 "rating")
WX_interest <- c(X_interest, "GEM_HH_GR", "INWONER_scaled", 'medianinc_scaled')
# WX_interest<-c(X_interest)

small_subset <- c(1:1000)
sample_length = round(1000/3)

#Equal sample
ind_operational <- which(y_df_train$DV==1, arr.ind=TRUE)
ind_perm <- which(y_df_train$DV==2, arr.ind=TRUE)
ind_temp<- which(y_df_train$DV==3, arr.ind=TRUE)
small_subset <- c(sample(ind_perm, sample_length), 
                  sample(ind_temp, sample_length),
                  sample(ind_operational, sample_length))

#subset without temp
small_subset <- c(sample(ind_perm, sample_length), 
                  sample(ind_operational, sample_length))

##Obtain dataframes that will be used in analysis, assume that missing values have
##been handled in python
y <- factor(y_df_train$DV)
y <- y[small_subset]

# X <- X_df[small_subset,X_interest]
WX <- WX_df_train[small_subset,WX_interest]
# W <- W_df[small_subset,W_interest]
cluster <- sample(1:3, length(small_subset), replace=TRUE)
cluster <- WX_df_train$cluster[small_subset]

#Test dataframes
y_test <- factor(y_df_test$DV)
WX_test <- WX_df_test[, WX_interest]

datlist <- list(N=nrow(WX),           #Nr of obs
                K=length(unique(y)),  #Possible outcomes
                D=ncol(WX)+1,            #NR of predictors
                x=cbind(1,WX),                 #Predictor Matrix
                y=as.numeric(y),      #Dependent Variable
                x_test = cbind(1,WX_test),
                y_test = y_test,
                N_test = nrow(WX_test))
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

b_logit.out <- stan(file='./stan_stuff/logitpooled.stan',
                    data=datlist,
                    iter = 500,
                    chains = 1,
                    seed = 12591)

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