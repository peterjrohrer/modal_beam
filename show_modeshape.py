#%% Run as Notebook
import numpy as np
import openmdao.api as om
import myconstants as myconst
import os

from matplotlib import rc
import matplotlib.pyplot as plt

from utils import *
from cantilever_group import Cantilever

# Bring in problem with defined defaults
prob = om.Problem()

nElem = 11
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

cantilever_group = Cantilever(nodal_data=nodal_data) # Increased nodes
# cantilever_group.linear_solver = om.DirectSolver(assemble_jac=True)
# cantilever_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6, use_aitken=True)

prob.model.add_subsystem('cantilever', 
    cantilever_group, 
    promotes_inputs=['D_beam', 'wt_beam', 'L_beam_tot'],
    promotes_outputs=['M_global', 'K_global', 'Z_beam', 'L_beam', 'M_beam', 'tot_M_beam',
        'eig_vector_*', 'eig_freq_*', 'z_beamnode', 'z_beamelem',
        'x_beamnode_*', 'x_d_beamnode_*', 'x_beamelem_*', 'x_d_beamelem_*', 'x_dd_beamelem_*',])

# Set inputs
prob.model.set_input_defaults('D_beam', val=0.75*np.ones(nElem), units='m')
prob.model.set_input_defaults('wt_beam', val=0.15*np.ones(nElem), units='m')
prob.model.set_input_defaults('L_beam_tot', val=10., units='m')

# Setup and run problem
prob.setup(mode='rev', derivatives=True)
prob.set_solver_print(level=1)
prob.run_model()

print('Mode 1 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_1']), (2*np.pi*float(prob['eig_freq_1'])), (float(prob['eig_freq_1']))))
print('Mode 2 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_2']), (2*np.pi*float(prob['eig_freq_2'])), (float(prob['eig_freq_2']))))
print('Mode 3 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_3']), (2*np.pi*float(prob['eig_freq_3'])), (float(prob['eig_freq_3']))))

## --- Check Eigenvals
M_glob = prob['M_global'] 
K_glob = prob['K_global']

# M_glob_inv = np.linalg.inv(M_glob)
# eig_mat = np.matmul(M_glob_inv, K_glob)
# eig_vals_raw, eig_vecs = np.linalg.eig(eig_mat)
# eig_vals = np.sqrt(np.real(np.sort(eig_vals_raw))) 
# eig_vecs_xloc = np.linspace(0,1,3)

# print('Eigenfrequencies: %3.3f rad/s, %3.3f rad/s, %3.3f rad/s' % (eig_vals[0], eig_vals[1], eig_vals[2]))

## --- Pull out Modeshape
x_beamnode_1 = prob['x_beamnode_1']
x_beamnode_2 = prob['x_beamnode_2']
x_beamnode_3 = prob['x_beamnode_3']
x_beamelem_1 = prob['x_beamelem_1']
x_beamelem_2 = prob['x_beamelem_2']
x_beamelem_3 = prob['x_beamelem_3']
x_d_beamnode_1 = prob['x_d_beamnode_1']
x_d_beamnode_2 = prob['x_d_beamnode_2']
x_d_beamnode_3 = prob['x_d_beamnode_3']
x_d_beamelem_1 = prob['x_d_beamelem_1']
x_d_beamelem_2 = prob['x_d_beamelem_2']
x_d_beamelem_3 = prob['x_d_beamelem_3']
x_dd_beamelem_1 = prob['x_dd_beamelem_1']
x_dd_beamelem_2 = prob['x_dd_beamelem_2']
x_dd_beamelem_3 = prob['x_dd_beamelem_3']
z_beamnode = prob['z_beamnode']
z_beamelem = prob['z_beamelem']

mode1_freq = (2*np.pi*float(prob['eig_freq_1']))
mode2_freq = (2*np.pi*float(prob['eig_freq_2']))
mode3_freq = (2*np.pi*float(prob['eig_freq_3']))

## --- Shapes PLOT from FEA
font = {'size': 16}
plt.rc('font', **font)
fig1, ax1 = plt.subplots(figsize=(12,8))

# Plot shapes
shape1 = ax1.plot(z_beamnode, x_beamnode_1, label='1st Mode: %2.2f rad/s' %mode1_freq, c='r', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)
shape2 = ax1.plot(z_beamnode, x_beamnode_2, label='2nd Mode: %2.2f rad/s' %mode2_freq, c='g', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
shape3 = ax1.plot(z_beamnode, x_beamnode_3, label='3rd Mode: %2.2f rad/s' %mode3_freq, c='b', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)

# Set labels and legend
ax1.legend()
ax1.set_title('Modeshapes from FEA')
ax1.set_xlabel('Length (z)')
ax1.set_ylabel('Deformation (x)')
ax1.grid()


# Show sketch
plt.show()
plt.tight_layout()
my_path = os.path.dirname(__file__)
fname = 'modeshapes'
plt.savefig(os.path.join(my_path,(fname+'.png')), dpi=400, format='png')
