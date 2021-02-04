library('rstan')
library('shinystan')
library('nnet')
library('tidyr')
library("dplyr")
library('notifier')


setwd("C:/Users/bartd/Erasmus/Erasmus_/Jaar 4/Master Econometrie/Seminar/UnileverCase_Conda/")

dat <- list(N = 2, J = 3, Q = 4,
                             data_matrix = array(1:24, dim = c(4, 2, 3)),
                             data_matrix2 = array(1:24, dim = c(4, 2, 3)),
                             data_vector = array(1:8, dim = c(4, 2)));

fit <- stan("./stan_stuff/TestDimensions.stan", data = dat, iter = 1, chains = 1)
    
    