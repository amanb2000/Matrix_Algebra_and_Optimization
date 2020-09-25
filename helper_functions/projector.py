import numpy as np

def project_full_soln(y_, S, b):
    """
    Program to carry out the full solution to an affine projection question.

    TAKES:
        - y_:       Input vector (1D numpy array length n).
        - S:        Set of vectors that define subspace (2D numpy m x n array where 
                    m is the number of vectors in the set).
        - b:        Affine offset vector for the vector space (1D numpy array length n). 

    RETURNS: Tuple composed of...
        - alphas:   Array of coefficients for the vectors in set S (1D numpy array length m)
        - y_proj:   The final projected vector ('y-star' in class).

    It also spits out a whole lot of text (see steps below) to help you check your work.

    1. Subtract b from y (get back to subspace).
    2. Create a matrix of inner products of the set.
    3. Create a vector of inner products of the vectors in the set and the input.
    4. Invert the matrix.
    5. Multiply the inverse by the vector from 3 to get alpha values.
    6. Reveal values of alpha values, b value, and the projected vector value as alphas.dot(v) + b.
    """
    __author__ = "Aman Bhargava"
    
    y = np.copy(y_)
    
    y_original = np.copy(y)
    y -= b
    
    print('Step 1: Adjusted y (y-b) is: {} - {} = {}'.format(y_original, b, y))
    
    M = np.zeros([len(S), len(S)])
    for i in range(len(S)):
        for j in range(len(S)):
            M[i,j] = S[i].dot(S[j])
    
    print('Step 2: Our matrix of the inner products in S is: {}'.format(M))
    
    
    v = S.dot(y)
    print('step 3: Our vector of inner products <v_i, x> for each v_i in S is: {}'.format(v))
    
    M_inv = np.linalg.inv(M)
    print('The inverted matrix M_inv is {} with M_inv*M = {}'.format(M_inv, M.dot(M_inv)))
    
    alphas = M_inv.dot(v)
    y_proj = alphas.dot(S) + b
    
    print('Alphas: {}'.format(alphas))
    print('y_proj = alphas * S + b = {}'.format(y_proj))
    
    error = y_proj - y_original
    
    print('Error between projection and original: {}'.format(error))
    
    print('\nAre the inner products with all the set vectors and the error term:')
    
    for i in S:
        print(i.dot(error))
        
    print('\nDone computation')

    return(alphas, y_proj)
    
    
