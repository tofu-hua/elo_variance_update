def Create_cov_mat(cov_a, cov_b):
  ID_mat = jnp.zeros((cov_a.shape[0], cov_a.shape[1]))
  mat_a = onp.concatenate((cov_a, ID_mat), axis = 1)
  mat_b = onp.concatenate((ID_mat, cov_b), axis = 1)
  cov_full = onp.concatenate((mat_a, mat_b), axis = 0)

  return cov_full