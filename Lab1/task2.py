import tensorflow as tf

A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

print("Matrix A:\n", A.numpy())
print("Matrix B:\n", B.numpy())

# Addition & Subtraction
print("A + B:\n", tf.add(A, B).numpy())
print("A - B:\n", tf.subtract(A, B).numpy())

# Matrix Multiplication
print("A x B:\n", tf.matmul(A, B).numpy())

# Transpose
print("Transpose of A:\n", tf.transpose(A).numpy())

# Determinant
print("Determinant of A:", tf.linalg.det(A).numpy())

# Inverse
print("Inverse of A:\n", tf.linalg.inv(A).numpy())

# Eigenvalues
eigen_values, eigen_vectors = tf.linalg.eig(A)
print("Eigen Values:\n", eigen_values.numpy())
print("Eigen Vectors:\n", eigen_vectors.numpy())

# Rank
print("Rank of A:", tf.linalg.matrix_rank(A).numpy())

# Norm
print("Frobenius Norm:", tf.norm(A).numpy())
