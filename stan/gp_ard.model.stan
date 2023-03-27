functions {
  matrix cov_exp_quad_ARD(array[] vector x1,
                          array[] vector x2,
                          real alpha,
                          vector rho) {
    int N1 = size(x1);
    int N2 = size(x2);
    matrix[N1, N2] K;
    real sq_alpha = square(alpha);
    for (i in 1:N1) {
      for (j in 1:N2) {
        K[i, j] = sq_alpha * exp(-0.5 * dot_self((x1[i]-x2[j]) ./ rho ));
      }
    }
    return K;
  }
  vector gp_pred_rng(array[] vector x2,
                     vector y1,
                     array[] vector x1,
                     real alpha,
                     vector rho,
                     real sigma,
                     real delta) {
    int N1 = size(x1);
    int N2 = size(x2);

    vector[N2] f2;
    {
      matrix[N1, N1] L_K;
      vector[N1] K_div_y1;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N2] v_pred;
      vector[N2] f2_mu;
      matrix[N2, N2] cov_f2;
      matrix[N1, N1] K;
      K = cov_exp_quad_ARD(x1, x1, alpha, rho);
      for (n in 1:N1)
        K[n, n] = K[n,n] + square(sigma);
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_x2 = cov_exp_quad_ARD(x1, x2, alpha, rho);
      f2_mu = (k_x1_x2' * K_div_y1);
      v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      cov_f2 = cov_exp_quad_ARD(x2, x2, alpha, rho) - v_pred' * v_pred;

      f2 = multi_normal_rng(f2_mu, add_diag(cov_f2, rep_vector(delta, N2)));
    }
    return f2;
  }  
}

data {
  int<lower=1> D;
  int<lower=1> N1;
  int<lower=1> N2;
  array[N1] vector[D] x1;
  array[N2] vector[D] x2;
  vector[N1] y1;
}
transformed data {
  real delta = 1e-9;
}
parameters {
  vector<lower=0>[D] rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
}

model {
  // covariances and Cholesky decompositions
  matrix[N1, N1] cov = cov_exp_quad_ARD(x1, x1, alpha, rho);
  matrix[N1, N1] L_cov = cholesky_decompose(add_diag(cov, sigma^2));

  // priors
  rho ~ inv_gamma(5, 5);
  alpha ~ inv_gamma(5, 5);
  sigma ~ std_normal();

  // model
  y1 ~ multi_normal_cholesky(rep_vector(0, N1), L_cov);
}
generated quantities {
  vector[N2] y2 = gp_pred_rng(x2, y1, x1, alpha, rho, sigma, delta);
}