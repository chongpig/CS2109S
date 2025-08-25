import copy

def mult_matrices(A, B):
    """
    Multiplies matrix A by matrix B, giving AB.
    Note
    ----
    Do not use numpy for this question.
    """
    if len(A[0]) != len(B):
        raise Exception('Incompatible dimensions for matrix multiplication of A and B')
    """ YOUR CODE HERE """
    rows = len(A)
    cols = len(B[0])
    C = []
    for i in range(rows):
        newRow = []
        for j in range(cols):
            newNum = 0
            for k in range(len(A[0])):
                newNum = newNum + A[i][k] * B[k][j]
            newRow.append(newNum)
        C.append(newRow)
    return C
            
    """ YOUR CODE END HERE """

# Test cases
A = [[5, 7, 9], [1, 4, 3]]
B = [[2, 5], [3, 6], [4, 7]]
A_copy = copy.deepcopy(A)
B_copy = copy.deepcopy(B)

actual = mult_matrices(A, B)
expected = [[67, 130], [26, 50]]
assert(A == A_copy and B == B_copy)
assert(actual == expected)

A2 = [[-13, -10], [-24, 14]]
B2 = [[1, 0], [0, 1]]
A2_copy = copy.deepcopy(A2)
B2_copy = copy.deepcopy(B2)

actual2 = mult_matrices(A2, B2)
expected2 = [[-13, -10], [-24, 14]]
assert(A2 == A2_copy and B2 == B2_copy)
assert(actual2 == expected2)
