data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> Q;
  matrix[N, J] data_matrix[Q];
  int data_matrix2[Q, N, J]; // specify the same data with data_matrix
  vector[N] data_vector[Q];
  
}
transformed data{
  matrix[N, J] data_matrix_transf[Q];
  data_matrix_transf = data_matrix;
  for (n in 1:N)
    for (j in 1:J)
      data_matrix_transf[1,n,j] = 0;
}
parameters {
  real y;
}

transformed parameters{
  print("dm=", data_matrix_transf); 
  print("dm[3,2,1]=", data_matrix[3,2,1]);
  print("dm2[3,2,1]=", data_matrix2[3,2,1]);
  print("dm[2,1,2]=", data_matrix[2,1,2]);
  print("dm2[2,1,2]=", data_matrix[2,1,2]);
  print("dm2=", data_matrix2); 
  print("dv=", data_vector); 
  print("dm_sub[,,2]", data_matrix[,,2]);
  print("dm2_sub[,,2]", data_matrix2[,,2]);
  print("dm[,,1] = ", data_matrix[,,1];
  
}

model {
  y ~ normal(0, 1);
} 

// Chain 1: 
//dm=[ 1  9 17
//  5 13 21, 2 10 18
//  6 14 22, 3 11 19
//  7 15 23, 4 12 20
//  8 16 24]
// dm[3,2,1]=7
// dm2[3,2,1]=7
// dm[2,1,2]=10
// dm2[2,1,2]=10
// dm2=[[[1,9,17],[5,13,21]],[[2,10,18],[6,14,22]],[[3,11,19],[7,15,23]],[[4,12,20],[8,16,24]]]
// dv=[1
// 5,2
// 6,3
// 7,4
// 8]


