import numpy as np
import scipy.linalg

def Eigenproblem(nDOF, M_mode, K_mode):
    """
    returns:
       Q     : matrix of column eigenvectors
       Lambda: matrix where diagonal values are eigenvalues
              frequency = np.sqrt(np.diag(Lambda))/(2*np.pi)
         or
    frequencies (if freq_out is True)
    """
    M = M_mode
    K = K_mode
    
    D,Q = scipy.linalg.eig(K,M)
    for j in range(M.shape[1]):
        q_j = Q[:,j]
        modalmass_j = np.dot(q_j.T,M).dot(q_j)
        Q[:,j]= Q[:,j]/np.sqrt(modalmass_j)
    Lambda=np.dot(Q.T,K).dot(Q)
    lambdaDiag=np.diag(Lambda) # Note lambda might have off diagonal values due to numerics
    # Sorting
    I = np.argsort(lambdaDiag)
    Q = Q[:,I]
    lambdaDiag = lambdaDiag[I]
    # Export frequencies
    Lambda = np.sqrt(lambdaDiag)/(2*np.pi) # frequencies [Hz]
   
    # --- Renormalize modes 
    for j in range(Q.shape[1]):
        q_j = Q[:,j]
        iMax = np.argmax(np.abs(q_j))
        scale = q_j[iMax] # not using abs to normalize to "1" and not "+/-1"
        Q[:,j]= Q[:,j]/scale

    # --- Sanitization, ensure real values
    Q_im    = np.imag(Q)
    Q       = np.real(Q)
    imm     = np.mean(np.abs(Q_im),axis = 0)
    bb = imm>0
    if sum(bb)>0:
        W=list(np.where(bb)[0])
        print('[WARN] Found {:d} complex eigenvectors at positions {}/{}'.format(sum(bb),W,Q.shape[0]))
    Lambda = np.real(Lambda)

    return Q,Lambda

# --------------------------------------------------------------------------------}
# --- Mode tools 
# --------------------------------------------------------------------------------{
def modeNorms(q, iDOFstart=0, nDOF=6):
    """ 
    Return norms of components of a mode
    Norm is computed as sum(abs())
        q: mode vector
        iDOFStart: where to start in mode vector
        nDOF: number of DOF per node typically 6 for 3D and 2/3 for 2D
    """
    MaxMag=np.zeros(nDOF)
    for i in np.arange(nDOF): 
        MaxMag[i] = np.sum(np.abs(q[iDOFstart+i::nDOF]))
    return MaxMag

def normalize_to_last(Q, Imodes, iDOFStart=0, nDOF=6):
    for iimode, imode in enumerate(Imodes):
        mag = modeNorms(Q[:,imode], iDOFStart, nDOF)[:int(nDOF/2)]
        if np.max(mag) ==0:
            print('>>> mag', mag)
            raise Exception('Problem in mode magnitude, all norms are 0.')
        iMax= np.argmax(mag)
        v_= Q[iDOFStart+iMax::nDOF, imode]
        if np.abs(v_[-1])>1e-9:
            Q[:, imode]= Q[:, imode]/v_[-1]
        else:
            print('[WARN] fem_beam:normalize_to_last, mode {} has 0 amplitude at tip'.format(imode))
            Q[:, imode]= Q[:, imode]/v_[-1]
    return Q

def orthogonalizeModePair(Q1, Q2, iDOFStart=0, nDOF=6):
    # Find magnitudes to see in which direction the mode is the most
    mag1 = modeNorms(Q1, iDOFStart, nDOF)[:int(nDOF/2)]
    mag2 = modeNorms(Q2, iDOFStart, nDOF)[:int(nDOF/2)]
    idx1= np.argsort(mag1)[-1::-1]
    idx2= np.argsort(mag2)[-1::-1]
    if (idx1[0]>idx2[0]):
        idx=[idx2[0],idx1[0]]
    else:
        idx=[idx1[0],idx2[0]]
    k11 = sum(Q1[iDOFStart+idx[0]::nDOF])
    k12 = sum(Q1[iDOFStart+idx[1]::nDOF])
    k21 = sum(Q2[iDOFStart+idx[0]::nDOF])
    k22 = sum(Q2[iDOFStart+idx[1]::nDOF])
    Q1_ = k11*Q1 + k21*Q2
    Q2_ = k12*Q1 + k22*Q2
    return Q1_, Q2_

def identifyAndNormalizeModes(Q, nModes=None, element='frame3d', normalize=True):
    """ 
    Attempts to identify and normalized the first `nModes` modes
    Modes are normalized by last values unless this value is too small compared to the max
    in which case the max is used.
    Mode names are returned of the form [u,v][x,y,z][n]
      where "u": displacements, "v": slopes, and "n" is the mode number in that direction
    """
    if nModes is None: nModes=Q.shape[1]
    if element in ['frame3d','frame3dlin']:
        nDOF=6
        sDOF=['ux','uy','uz','vx','vy','vz']

    cDOF=np.zeros(nDOF,dtype=int) # Counter on Modes in each DOF
    modeNames=[]

    for i in np.arange(nModes):
        q=Q[:,i]
        mag = modeNorms(q, iDOFstart=0, nDOF=nDOF)
        idx= np.argsort(mag)[-1::-1]
        iMax = idx[0]
        U = Q[iMax::nDOF,i]
        # Detect rigid body mode (0 or NaN frequencies), component constant and non-zero
        rigid=False
        for idof in np.arange(nDOF):
            Ui = Q[idof::nDOF,i]
            Umax  = max(abs(Ui))
            if Umax>1e-6:
                if len(np.unique(np.around(Ui/Umax,3)))==1:
                    icst=idof
                    rigid=True
                    break
        # Mode name
        if rigid:
            mode_name =sDOF[iMax]+'_'+sDOF[icst]+'_rigid'
        else:
            cDOF[iMax]+=1
            mode_name = sDOF[iMax]+str(cDOF[iMax])
        modeNames.append(mode_name)

        #if sDOF[iMax] in ['vy','vz']:
        #    print('Mode {} has strong slope, double check identification'.format(i))
        #print('>>>Mode',i, 'name:',mode_name, mag)

        # Normalization by max or last
        Umax  = max(abs(U))
        Ulast = abs(U[-1])
        if Ulast*100< Umax: # some tuning factor if last is close to 0
            # Normalize by max
            fact = Umax*np.sign(U[-1])
        else:
            # Normalize by last
            fact = Ulast*np.sign(U[-1])
        if normalize:
            Q[:,i]= Q[:,i]/fact
    return Q, modeNames