import numpy as np
import scipy.linalg

def Eigenproblem(nDOF, M_mode, K_mode):
    K = K_mode
    M = M_mode

    vals, vecs = scipy.linalg.eig(M,K, left=False, right=True) # To match He (2022) paper - note must invert eigenvalues to get to natural frequencies
    
    # Normalize eigenvectors with M matrix
    norm_fac = np.zeros((1, nDOF))
    vecs_mortho = np.zeros((nDOF, nDOF))
    for i in range(nDOF):
        norm_fac[0,i] = np.sqrt(1./(vecs[:,i].T @ M @ vecs[:,i]))
        if not np.isnan(norm_fac[0,i]) : # To avoid ending up with an entire eigenvector of NaNs 
            vecs_mortho[:,i] = norm_fac[0,i] * vecs[:,i]
    
    # # Throw errors for unexpected eigenvalues
    if any(np.imag(vals) != 0.) :
        raise Exception('Imaginary eigenvalues')
    if any(np.real(vals) < -1.e-03) :
        raise Exception('Negative eigenvalues')

    # Check solution
    if not np.allclose((M @ vecs) - (K @ vecs @ np.diag(vals)), np.zeros((nDOF,nDOF)), atol=1.0) :
        raise Exception('Eigenvalue problem looks wrong')
    if not np.allclose((vecs_mortho.T @ M @ vecs_mortho) - np.eye(nDOF), np.zeros((nDOF,nDOF)), atol=1.0) :
        raise Exception('Eigenvectors not scaled properly')

    # Calculate modal expansion
    u_vec = np.ones(nDOF)
    q_vec = np.zeros(nDOF)
    for i in range(nDOF):
        q_vec[i] = (vecs[:,i].T @ M @ u_vec)/(vecs[:,i].T @ M @ vecs[:,i])

    eig_vectors = vecs_mortho
    eig_vals = np.diag(np.real(vals))   

    return eig_vectors, eig_vals

def EigSelect(idx, eig_vectors, eig_vals):
    # He (2022) Eigenproblem definition
    eig_vec = eig_vectors[:, idx]
    eig_freq = np.sqrt(1./np.diag(eig_vals)[idx]) / (2*np.pi) # Export in Hz

    return eig_vec, eig_freq
    