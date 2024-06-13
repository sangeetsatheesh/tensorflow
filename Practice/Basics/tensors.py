import tensorflow as tf

scalar = tf.constant(5)
print(f"Scalar value:", scalar)
print(f"Scalar via numpy:", scalar.numpy())
print(f"Scalar dimension:", scalar.ndim)

vector = tf.constant([1, 2, 3])
print(f"Vector value:", vector)
print(f"Vector via numpy:", vector.numpy())
print(f"Vector dimension:", vector.ndim)
print(f"Vector dtype:", vector.dtype)
print(f"Vector shape:", vector.shape)

matrix = tf.constant([[1, 2], [3, 4]])
print(f"Matrix value:", matrix)
print(f"Matrix via numpy:", matrix.numpy())
print(f"Matrix dimension:", matrix.ndim)
print(f"Matrix dtype:", matrix.dtype)
print(f"Matrix shape:", matrix.shape)

tensor = tf.constant(
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]])
print(f"Tensor value:", tensor)
print(f"Tensor via numpy:", tensor.numpy())
print(f"Tensor dimension:", tensor.ndim)
print(f"Tensor dtype:", tensor.dtype)
print(f"Tensor shape:", tensor.shape)

changeable_tensor = tf.Variable([10., 7.])
print(f"Changeable tensor value:", changeable_tensor)
print(f"Changeable tensor via numpy:", changeable_tensor.numpy())
print(f"Changeable tensor dtype:", changeable_tensor.dtype)
print(f"Changeable tensor shape:", changeable_tensor.shape)
changeable_tensor.assign([1, 2])
print(f"Changeable tensor via numpy after assigning int values:", changeable_tensor.numpy())
print(f"Changeable tensor dtype after assigning int values:", changeable_tensor.dtype)

changeable_tensor2 = tf.Variable([1, 2])
print(f"Another Changeable tensor value:", changeable_tensor2)
print(f"Another Changeable tensor via numpy:", changeable_tensor2.numpy())
print(f"Another Changeable tensor dtype:", changeable_tensor2.dtype)
print(f"Another Changeable tensor shape:", changeable_tensor2.shape)
# changeable_tensor2.assign([3., 5.])
# print(f"Another Changeable tensor via numpy after assigning float values:", changeable_tensor2.numpy())
# print(f"Another Changeable tensor dtype after assigning float values:", changeable_tensor2.dtype)
# print(f"Tensor type does not change with assignment and hence cannot convert float to int and  throws error!")


random_tensor = tf.random.Generator.from_seed(42)
print(f"Random tensor generator with seed 42\n", random_tensor)
random1 = random_tensor.normal(shape=(3, 2))
print(f"Random tensor value from normal distribution:\n", random1)
print(f"Random tensor via numpy from normal distribution:\n", random1.numpy())
print(f"Random tensor dtype from normal distribution:", random1.dtype)
print(f"Random tensor shape from normal distribution:", random1.shape)

random2 = random_tensor.uniform(shape=(3, 2))
print(f"Random tensor value from uniform distribution:\n", random2)
print(f"Random tensor via numpy from uniform distribution:\n", random2.numpy())
print(f"Random tensor dtype from uniform distribution:", random2.dtype)
print(f"Random tensor shape from uniform distribution:", random2.shape)

# Shuffle

not_shuffled = tf.constant([[1, 2], [3, 4], [5, 6]])
print(f"Not shuffled tensor values:\n", not_shuffled.numpy())
tf.random.set_seed(42)
# print(f"After shuffling tensor :\n", tf.random.shuffle(not_shuffled))
not_shuffled = tf.random.shuffle(not_shuffled)
print(f"After shuffling tensor :\n", not_shuffled.numpy())

# Create a rank 4 tensors of all zeros and ones

r4_tensor_zeros = tf.zeros([2, 3, 4, 5])
print(f"Rank 4 tensor with zeros:\n", r4_tensor_zeros.numpy())
print(f"Shape of rank 4 tensor without zeros:\n", r4_tensor_zeros.shape)
print(f"Dataype of rank 4 tensor without zeros:\n", r4_tensor_zeros.dtype)
print(f"Shape of rank 4 tensor with zeros:\n", r4_tensor_zeros.shape)
print(f"Elements of axis 0 of tensor:\n", r4_tensor_zeros.shape[0])
print(f"Elements of axis 1 of tensor:\n", r4_tensor_zeros.shape[1])
print(f"Elements of last axis of tensor:\n", r4_tensor_zeros.shape[-1])
print(f"Total number of elements in rank 4 tensor:\n", tf.size(r4_tensor_zeros).numpy())
print(f"Expand dims last axis:\n", tf.expand_dims(r4_tensor_zeros, axis=-1))

r4_tensor_zeros += 10
print(f"Add 10 to all elements:\n", r4_tensor_zeros.numpy())

X = tf.constant([[1, 2], [3, 4], [5, 6]])
Y = tf.constant([[7, 8, 9], [10, 11, 12]])

print(f"X matrix values are:\n{X.numpy()}\nand Y matrix values are\n{Y.numpy()}")
print(f"Shape of X matrix {X.shape} and Y matrix {Y.shape}")
print(f"Matrix multiplying these two matrices")
Z = tf.matmul(X, Y)
print(f"Z matrix values:\n{Z.numpy()}")
print(f"Shape of Z matrix {Z.shape}")

print(f"Matrix X is:\n{X.numpy()}")
A = tf.reshape(X, shape=(2, 3))
print(f"Matrix A values from reshaping X:\n{A.numpy()}")
print(f"Matrix A shape:{A.shape}")

B = tf.transpose(A)
print(f"Values of Matrix B which is the transpose of A:\n{B.numpy()}")
print(f"Matrix B shape:{B.shape}")
print(f"Note: Reshape and transpose are different!")
# Performing Dot product operation
C = tf.tensordot(X, Y, axes=1)
print(f"Values of Matrix C which is the dot product of X and Y:\n{C.numpy()}")
print(f"Matrix C type:{C.dtype}")
print(f"Changing type to float")
C = tf.cast(C, tf.float32)
print(f"Values of Matrix C after cast():\n{C.numpy()}")
print(f"Matrix C type:{C.dtype}")

# Find the minimum, maximum, mean and sum of Matrix C
print(f"Minimum value of Matrix C:\n{tf.reduce_min(C)}")
print(f"Maximum value of Matrix C:\n{tf.reduce_max(C)}")
print(f"Mean value of Matrix C:\n{tf.reduce_mean(C)}")
print(f"Product of all values Matrix X (for simplicity -> (6!)) :\n{tf.reduce_prod(X)}")
print(f"Sum of Matrix C:\n{tf.reduce_sum(C)}")
