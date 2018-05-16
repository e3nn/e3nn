import numpy as np

from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
from se3_cnn.SO3 import rot

def build_symm_traceless():
    ''' build random symmetric traceless tensor '''
    a,b,c,d,e = np.random.normal(size=5)
    A = np.zeros((3,3))
    A[0,0] = a
    A[1,1] = d
    A[2,2] = -(a+d)
    A[0,1] = A[1,0] = b
    A[0,2] = A[2,0] = c
    A[1,2] = A[2,1] = e
    assert np.trace(A)==0, 'matrix not traceless'
    assert np.all(A==A.T), 'matrix non symmetric'
    return A

def retrive_dof(matrix):
    ''' extract 5 degrees of freedom (a,b,c,d,e) of symmetric traceless tensor '''
    a,b,c = matrix[0,:]
    d,e   = matrix[1,1:]
    return np.array([a,b,c,d,e])

def conjugate_rot(matrix, alpha, beta, gamma):
    ''' transform cartesian tensor by conjugating it with rotation matrices '''





    R = rot(alpha,beta,gamma)
    # R = rot(alpha,beta,gamma).T




    conj = R @ matrix @ (R.T)
    assert np.allclose(conj, conj.T), 'conjugation not symmetric'
    assert np.allclose(np.trace(conj), 0), 'conjugation not traceless'
    return conj

def solve_Q_C2S():
    def _kron():
        angles = np.pi*np.array([2,1,2])*np.random.rand(3)
        id_ = np.eye(5)




        D = wigner_D_matrix(2, *angles)
        # D = wigner_D_matrix(2, *angles).T





        A = build_symm_traceless()
        phiA = retrive_dof(A).reshape(-1,1)
        phiRAR = retrive_dof(conjugate_rot(A, *angles)).reshape(-1,1)
        return np.kron(phiA.T, D) - np.kron(phiRAR.T, id_)
    def _concat_kron(N=100):
        return np.concatenate([_kron() for _ in range(N)])
    def _compute_kernel(matrix, eps=1e-12):
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        kernel = v[s < eps]
        return kernel
    vecQ = _compute_kernel(_concat_kron())
    assert len(vecQ) == 1
    return vecQ.reshape(5,5).T

if __name__ == '__main__':
    Q_C2S = solve_Q_C2S()
    Q_S2C = np.linalg.inv(Q_C2S)

    angles = np.pi*np.random.rand(3)*np.array([2,1,2])


    print(np.round(Q_C2S, decimals=2))

    D = wigner_D_matrix(2, *angles)
    # D = wigner_D_matrix(2, *angles).T




    A = build_symm_traceless()
    phi = retrive_dof(A)
    phi_R = retrive_dof(conjugate_rot(A, *angles))

    print(np.allclose(Q_S2C @ D @ Q_C2S @ phi,  phi_R))
    print(np.allclose(D @ Q_C2S @ phi,  Q_C2S @ phi_R))
