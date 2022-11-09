#%% Run as Notebook
import numpy as np
import scipy.linalg
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND
import myconstants as myconst
import os

import matplotlib.pyplot as plt

from utils import *
from cantilever_group import Cantilever

## --- Processing nodes (can be done outside of optimization!)
nElem = 20
nNode = nElem + 1
nDOFperNode =  6
nNodeperElem =  2
Elem2Nodes, Nodes2DOF, Elem2DOF = LinearDOFMapping(nElem, nNodeperElem, nDOFperNode)
nDOF_tot = nDOFperNode * nNode
IDOF_All = np.arange(0,nDOF_tot)
# Tip and root degrees of freedom
IDOF_root = Nodes2DOF[Elem2Nodes[0,:][0] ,:]
IDOF_tip  = Nodes2DOF[Elem2Nodes[-1,:][1],:]
# Handle BC and root/tip conditions
BC_root = [0,0,0,0,0,0]
BC_tip  = [1,1,1,1,1,1]
# Boundary condition transformation matrix (removes row/columns)
Tr=np.eye(nDOF_tot)
# Root and Tip BC
IDOF_removed = [i for i,iBC in zip(IDOF_root, BC_root) if iBC==0]
IDOF_removed += [i for i,iBC in zip(IDOF_tip, BC_tip) if iBC==0]
Tr = np.delete(Tr, IDOF_removed, axis=1) # removing columns
# --- Create mapping from M to Mr
nDOF_r = Tr.shape[1]
IDOF_BC = list(np.setdiff1d(IDOF_All, IDOF_removed))
IFull2BC = np.zeros(nDOF_tot,dtype=int)
IBC2Full = np.zeros(nDOF_r,dtype=int)
k=0
for i in IDOF_All:
    if i in IDOF_removed:
        IFull2BC[i]=-1
    else:
        IFull2BC[i]=k
        IBC2Full[k]=i
        k+=1

nMode = 5
nodal_data = {
    'nElem': nElem,
    'nNode': nNode,
    'nDOFperNode': nDOFperNode,
    'nNodeperElem': nNodeperElem,
    'Elem2Nodes': Elem2Nodes, 
    'Nodes2DOF': Nodes2DOF, 
    'Elem2DOF': Elem2DOF,
    'nDOF_tot': nDOF_tot,
    'IDOF_root': IDOF_root, 
    'IDOF_tip': IDOF_tip, 
    'BC_root': BC_root, 
    'BC_tip': BC_tip,
    'IDOF_removed': IDOF_removed,
    'Tr': Tr, 
    'nDOF_r': nDOF_r,
    'nMode': nMode,
}

# Bring in problem with defined defaults
prob = om.Problem()
cantilever_group = Cantilever(nodal_data=nodal_data) # Increased nodes
# cantilever_group.linear_solver = om.DirectSolver(assemble_jac=True)
# cantilever_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6, use_aitken=True)

# Set inputs
prob.model.set_input_defaults('L_beam_tot', val=5., units='m')
prob.model.set_input_defaults('D_beam', val=0.25*np.ones(nElem), units='m')
prob.model.set_input_defaults('wt_beam', val=0.01*np.ones(nElem), units='m')

prob.model.add_subsystem('cantilever', 
    cantilever_group, 
    promotes_inputs=['D_beam', 'wt_beam', 'L_beam_tot'],
    promotes_outputs=['L_beam', 'A_beam', 'Ix_beam', 'Iy_beam', 'M_beam', 'x_beamnode', 'y_beamnode', 'z_beamnode', 'dir_cosines', 'Q', 'eig_freqs', 'x_nodes', 'y_nodes', 'z_nodes', 'y_d_nodes', 'z_d_nodes', 'y_dd_nodes', 'z_dd_nodes', 'M_modal', 'K_modal'])

# Setup and run problem
prob.setup(mode='rev', derivatives=True)
prob.set_solver_print(level=1)
prob.run_model()

print('----- FROM FINITE ELEMENT MODEL -----')
for n in range(nMode):
    m = (n+1)
    print('Mode %1d Nat. Period: %3.3f Hz (%3.3f s)' %(m, (prob['eig_freqs'][n]), (1./prob['eig_freqs'][n])))

## --- Check Eigenvals
M_modal = prob['M_modal'] 
K_modal = prob['K_modal']

eig_vals, eig_vecs = scipy.linalg.eig(K_modal, M_modal)
modal_freqs = np.sort(np.sqrt(np.real(eig_vals)) /(2*np.pi))

print('----- FROM MODAL MATRICES -----')
for n in range(nMode):
    m = (n+1)
    print('Mode %1d Nat. Period: %3.3f Hz (%3.3f s)' %(m, (modal_freqs[n]), (1./modal_freqs[n])))

## --- Pull out Modeshape
x_nodes = prob['x_nodes']
y_nodes = prob['y_nodes']
z_nodes = prob['z_nodes']
# x_d_nodes = prob['x_d_nodes']
y_d_nodes = prob['y_d_nodes']
z_d_nodes = prob['z_d_nodes']
# x_dd_nodes = prob['x_dd_nodes']
y_dd_nodes = prob['y_dd_nodes']
z_dd_nodes = prob['z_dd_nodes']

## --- Shapes Plot from FEA
font = {'size': 16}
plt.rc('font', **font)
fig1, axs = plt.subplot_mosaic([['ul', '.'], ['ll', 'lr']], figsize=(12, 10), layout="constrained", sharey=True)

for i in range(nMode):
    axs['ul'].plot(x_nodes[:,i], y_nodes[:,i], label='Mode %2d: %2.3f Hz' %((i+1, prob['eig_freqs'][i])), ls='None', marker='o', ms=5)
    axs['ll'].plot(x_nodes[:,i], z_nodes[:,i], label='Mode %2d: %2.3f Hz' %((i+1, prob['eig_freqs'][i])), ls='None', marker='o', ms=5)
    axs['lr'].plot(y_nodes[:,i], z_nodes[:,i], label='Mode %2d: %2.3f Hz' %((i+1, prob['eig_freqs'][i])), ls='None', marker='o', ms=5)

# Set labels and legend
axs['ul'].grid()
axs['ul'].set_xlim(-0.1,5.1)
axs['ul'].set_ylabel('Y-displacement')
axs['ll'].grid()
axs['ll'].set_xlim(-0.1,5.1)
axs['ll'].set_xlabel('X-displacement')
axs['ll'].set_ylabel('Z-displacement')
# axs['ll'].set_ylim(-1,1.1)
axs['lr'].grid()
axs['lr'].set_xlim(-1.1,1.1)
axs['lr'].set_ylim(-1,1.1)
axs['lr'].set_xlabel('Y-displacement')

handles, labels = axs['lr'].get_legend_handles_labels()
fig1.legend(handles, labels, loc='upper right')
fig1.suptitle('Modeshapes from FEA')

# Show sketch
plt.show()
plt.tight_layout()
my_path = os.path.dirname(__file__)
fname = 'modeshapes'
plt.savefig(os.path.join(my_path,(fname+'.png')), dpi=400, format='png')
