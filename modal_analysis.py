import numpy as np
import myconstants as myconst

def ModalMass(M_beam, z_nodes):
    nElem = len(M_beam)
    nModes = z_nodes.shape[1]

    M_modal = np.zeros((nModes,nModes))

    for i in range(nElem):
        # modal_contrib = np.matmul(z_nodes[i,:].reshape(nModes,1),z_nodes[i,:].reshape(1,nModes))
        # M_modal += M_beam[i] * modal_contrib
        for j in range(nModes):
            for k in range(nModes):
                M_modal[j,k] += M_beam[i] * z_nodes[i,j] * z_nodes[i,k]

    return M_modal

def ModalStiff(D_beam, wt_beam, L_beam, P_beam, z_d_nodes, z_dd_nodes):
    nElem = len(D_beam)
    nModes = z_d_nodes.shape[1]

    K_modal = np.zeros((nModes,nModes))

    for i in range(nElem):
        EI = (np.pi / 64.) * (D_beam[i]**4. - (D_beam[i] - 2. * wt_beam[i])**4.) * myconst.E_STL
        ax_force = P_beam[i]

        modal_contrib_d = np.matmul(z_d_nodes[i,:].reshape(nModes,1),z_d_nodes[i,:].reshape(1,nModes))
        modal_contrib_dd = np.matmul(z_dd_nodes[i,:].reshape(nModes,1),z_dd_nodes[i,:].reshape(1,nModes))

        K_modal += EI * L_beam[i] * modal_contrib_dd
        K_modal += ax_force * L_beam[i] * modal_contrib_d

    return K_modal