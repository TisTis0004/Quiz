#Inverse
import numpy as np
A = np.array([[-2, 7], [3, 6]])
inverse_A = np.linalg.inv(A)
print("Inverse:")
print(inverse_A)
print()
checking_result = A @ inverse_A
print("Checking:")
print(checking_result)
print("Is close to identity matrix:")
print(np.allclose(checking_result, np.eye(A.shape[0])))
##############################
#Left Inverse
import numpy as np
A = np.array([[1, -3, 7], [-1, 4, -6], [1, 4, 6], [1, -3, 0]])
left = np.linalg.inv(A.T @ A) @ A.T
checking_left = A @ left
print("Left Inverse:")
print(left)
print()
print("Checking:")
print(checking_left)
print("Is close to identity matrix:")
print(np.allclose(checking_left, np.eye(A.shape[0])))
##############################
#Right Inverse
import numpy as np
A = np.array([[1,-3,7],[-1,4,-6],[1,4,6],[1,-3,0]])
right = (A.T) @ np.linalg.inv(A @ A.T)
checking_right = A @ right
print("Right Inverse:")
print(right)
print()
print("Checking:")
print(checking_right)
print("Is close to identity matrix:")
print(np.allclose(checking_right, np.eye(A.shape[0])))
##############################
#QR factorization
import numpy as np
A = np.array([[1,-3,7],[-1,4,-6],[1,4,6],[1,-3,0]])
Q, R = np.linalg.qr(A)
Q = -Q
R= -R

#If he ask for upper triangle matrix :
print("Upper Triangle Matrix:")
print(R)

print()
#If he ask for orthogonal matrix :
print("Orthogonal Matrix:")

print(Q)
##############################
#if A was n*n matrix (square) and b was n-vector you can solve this
#equation Ax = b
import numpy as np
A = np.array([[1,-3,7,3],[-1,4,-6,6],[1,4,6,8],[2,6,3,9]])
b = np.array([2,5,1,6])
x = np.linalg.solve(A,b)
print(x)
##############################
#Pseudo Inverse
import numpy as np
A = np.array([[-3,-4],[4,6],[1,1]])
print(np.linalg.pinv(A))
##############################
#Upper Triangle
import numpy as np
A = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(np.triu(A))
##############################
#Power Matrix
import numpy as np
A = np.array([[1,-3,7],[-1,4,-6],[1,4,6]])#must be square
print(np.linalg.matrix_power(A,2))#ENTER THE POWER 2ND Parameter
#if he ask for the diagonal :
print("Diagonal")
print(np.diag(A))
##############################
import numpy as np

def calculate_differences(original_matrix):
    # Calculate the first difference matrix
    first_diff_matrix = np.diff(original_matrix, axis=0)
    
    # Calculate the second difference matrix
    second_diff_matrix = np.diff(first_diff_matrix, axis=0)
    
    return first_diff_matrix, second_diff_matrix

# Example matrix
original_matrix = np.array([#######Change this
    [1,7,7,4,5],
    [-3,4,10,2,0],
    [5,1,36,2,7]
])

first_difference, second_difference = calculate_differences(original_matrix)

print("Original Matrix:")
print(original_matrix)
print("\nFirst Difference Matrix:")
print(first_difference)
print("\nSecond Difference Matrix:")
print(second_difference)
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
'''
to find complexity of inner product (2 matrices)
2*m*n*p

multiplying two 1000 x 1000 matrices requires 2 billion flops
1000*1000 @ 1000*1000
m n n p
2*m*n*p = 2*1000*1000*1000 = 2 billion
'''
