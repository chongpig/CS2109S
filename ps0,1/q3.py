import copy
def transpose_matrix(A):
    """
    Returns a new matrix that is the transpose of matrix A.
    Note
    ----
    Do not use numpy for this question.
    """
    """ YOUR CODE HERE """
    B = []
    rows = len(A)
    cols = len(A[0])
    for i in range(cols):
        newRow = []
        for j in range(rows):
            newRow.append(A[j][i])
        B.append(newRow)
    return B
    
    """ YOUR CODE END HERE """

# Test cases
A = [[5, 7, 9], [1, 4, 3]]
A_copy = copy.deepcopy(A)

actual = transpose_matrix(A_copy)
expected = [[5, 1], [7, 4], [9, 3]]
assert(A == A_copy)
assert(actual == expected)
