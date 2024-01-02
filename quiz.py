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
#Find Solution of X :
import numpy as np
A = np.array([[1,-3,7],[-1,4,-6],[1,4,6],[1,-3,0]])
b = np.array([3,0,3,1])

def back_subst(R,b_tilde):
    n = R.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = b_tilde[i]
        for j in range(i+1,n):
            x[i] = x[i] - R[i,j]*x[j]
        x[i] = x[i]/R[i,i]
    return x

def solve_via_backsub(A,b):
    Q, R = np.linalg.qr(A)
    Q = -Q
    R= -R
    b_tilde = Q.T @ b
    x = back_subst(R,b_tilde)
    return x

print(solve_via_backsub(A,b))
print()
# print("Checking:(doesn't have to be perfect)")
# print(solve_via_backsub(A,b) @ A.T)
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
#First and Second Difference Matrix
import numpy as np

def first_difference_matrix(matrix):
    return np.diff(matrix, axis=0)

def second_difference_matrix(matrix):
    return np.diff(matrix, axis=0, n=2)
A = np.array([
    [1, 4, 6],
    [3, 7, 2],
    [9, 5, 8],
    [2, 6, 1]
])

first_diff = first_difference_matrix(A)
second_diff = second_difference_matrix(A)
print("The Original Matrix:")
print(A)

print()

print("First Difference Matrix:")
print(first_diff)

print("\nSecond Difference Matrix:")
print(second_diff)

##############################
'''
to find complexity of inner product (2 matrices)
2*m*n*p

multiplying two 1000 x 1000 matrices requires 2 billion flops
1000*1000 @ 1000*1000
m n n p
2*m*n*p = 2*1000*1000*1000 = 2 billion
'''
