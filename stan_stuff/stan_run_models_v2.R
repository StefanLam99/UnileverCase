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
                 "P_NW_MIG_A","GEM_HH_GR", "UITKMINAOW",  "P_HINK_HH", "log_median_inc" )

table(y_df_train$DV) #1 = OPERATIONAl, 2= PERMANENTLY CLOSED, 3 = TEMPORARILY CLOSED
y_df_train$DV <- factor(y_df_train$DV)

small_subset <- c(1:1000)
sample_length = round(3000/3)

#Equal sample
set.seed(5)
ind_operational <- which(y_df_train$DV==1, arr.ind=TRUE)
ind_temp <- which(y_df_train$DV==3, arr.ind=TRUE)
ind_perm <- which(y_df_train$DV==2, arr.ind=TRUE)
small_subset_training <- c(sample(ind_perm, sample_length), 
                          ind_temp,
                          sample(ind_operational, sample_length))
small_subset_undersampling <- c(sample(ind_perm, length(ind_temp)), 
                                ind_temp,
                                sample(ind_operational, length(ind_temp)))
# small_subset_oversampling #TODO
full_dataset <- c(ind_operational, ind_temp, ind_perm)

#FUNCTION TO CREATE DATALIST
create_datlist <-  function(subset, with_zip = TRUE){
  y <-  factor(y_df_train$DV)
  y <- y[subset]
  
  if(with_zip){
    WX <- WX_df_train[subset,WX_interest]
    
    #Test dataframes
    y_test <- factor(y_df_test$DV)
    WX_test <- WX_df_test[, WX_interest]
  }else{
    WX <- WX_df_train[subset,X_interest]
    
    #Test dataframes
    y_test <- factor(y_df_test$DV)
    WX_test <- WX_df_test[, X_interest]
  }
  

  cluster <- WX_df_train$labels[subset]
  cluster_test <- WX_df_test$labels
  
  datlist <- list(N=nrow(WX),           #Nr of obs
                  K=length(unique(y)),  #Possible outcomes
                  D=ncol(WX)+1,            #NR of predictors
                  x=cbind(1,WX),                 #Predictor Matrix
                  y=as.numeric(y),  #Dependent Variable)
                  n_cluster = length(unique(cluster)), #Length of cluster
                  cluster = cluster, #Cluster
                  
                  boolean_test = 0, #Whether to test or not
                  
                  y_test = y_test, #dependent test variable (probably not needed for stan but easy for prediction)
                  X_test = WX_test, #Test set
                  N_test = nrow(WX_test), #observations in testset
                  cluster_test = cluster_test, #clusters in test set
                  n_cluster_test = length(unique(cluster)) #No.  cluster in testset.
                  )
  datlist
}

datlist_zip <- create_datlist(small_subset_training, with_zip = TRUE)
datlist_restaurant_only <- create_datlist(small_subset_training, with_zip = FALSE)

####ESTIMATE####
#FUNCTION TO ESTIMATE MODELS
#MODEL TYPE: 1 = multilog pooled, 2 = multilog unpooled, 3 = multilog with mu
function(datlist, model_type, test = TRUE, save = FALSE, iter = 1000, chains = 4){
  if(test){
    datlist$boolean_test <- 1
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
  }else{
    b.out <- stan(file='./stan_stuff/multilog_based_on_mcstan.stan',
                  data=datlist,
                  iter = iter,
                  chains = chains,
                  seed = 12591)
    if(save){
      saveRDS(b.out, "./stan_stuff/stan_model_output/unpooledhierarchical.rds")
    }
  }
}

# estimate Stan model
b_changed.out <- stan(file='./stan_stuff/multilog_insample.stan',
                      data=datlist.changed,
                      iter = 1000,
                      chains = 4,
                      seed = 12591)
saveRDS(b_changed.out, "./stan_stuff/stan_model_output/b_changed.rds")
b_unchanged.out <- stan(file='./stan_stuff/multilog_insample.stan',
                        data=datlist.unchanged,
                        iter = 1000,
                        chains = 4,
                        seed = 12591)
saveRDS(b_unchanged.out, "./stan_stuff/stan_model_output/b_normal.rds")

####PREDICTION EVALUATION####
contingency_table <- function(b.out, datlist){
  fit <- b.out
  fit.ext <- rstan::extract(fit)
  
  
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  prediction_median <- apply(fit.ext$y_pred_insample, 2, median)
  prediction_mean <- round(apply(fit.ext$y_pred_insample, 2, mean))
  prediction_mode <- apply(fit.ext$y_pred_insample, 2, getmode)
  
  print(mean(prediction_mode == datlist$y))
  cont_table <- table(prediction_mode, datlist$y)
  cont_table <- (rbind(cont_table, apply(cont_table, 2, sum)))
  cont_table <- cbind(cont_table, apply(cont_table,1, sum))
  print(cont_table)
  
  prediction <- as.factor(prediction_mode)
  # levels(prediction) <- c(1,2,3)
  datlist$y <- as.factor(datlist$y)
  print(levels(prediction))
  print(levels(datlist$y))
  print(confusionMatrix(prediction, reference=datlist$y))
}

print("CHANGED")
contingency_table(b_changed.out, datlist.changed)
print("UNCHANGED")
contingency_table(b_unchanged.out, datlist.unchanged)



