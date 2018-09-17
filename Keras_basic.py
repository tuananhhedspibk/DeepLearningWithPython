import numpy as np

def naive_vector_dot(x, y):
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  assert x.shape[0] == y.shape[0]

  z = 0
  for i in range(x.shape[0]):
    z += x[i] * y[i]

  return z

def naive_matrix_dot(x, y):
  assert len(x.shape) == 2
  assert len(y.shape) == 2
  assert x.shape[1] == y.shape[0]

  z = np.zeros((x.shape[0], y.shape[1]))
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      row_x = x[i, :]
      col_y = y[:, j]

      z[i, j] = naive_vector_dot(row_x, col_y)
  
  return z

x = np.array([[1, 2], [3, 4]])

y = np.array([[1, 2, 3], [4, 5, 6]])

# print naive_matrix_dot(x, y)

print y.shape