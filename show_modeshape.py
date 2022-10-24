#%% Run as Notebook
import numpy as np
import openmdao.api as om
import myconstants as myconst
import os

from matplotlib import rc
import matplotlib.pyplot as plt

from cantilever_group import Cantilever


# Bring in problem with defined defaults
prob = om.Problem()

elements = 11
cantilever_group = Cantilever(nNode=(elements+1), nElem=elements, nDOF=(2*elements)) # Increased nodes
# cantilever_group.linear_solver = om.DirectSolver(assemble_jac=True)
# cantilever_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6, use_aitken=True)

prob.model.add_subsystem('cantilever', 
    cantilever_group, 
    promotes_inputs=['D_beam', 'wt_beam'],
    promotes_outputs=['M_global', 'K_global', 'Z_beam', 'L_beam', 'M_beam', 'tot_M_beam',
        'eig_vector_*', 'eig_freq_*', 'z_beamnode', 'z_beamelem',
        'x_beamnode_*', 'x_d_beamnode_*', 'x_beamelem_*', 'x_d_beamelem_*', 'x_dd_beamelem_*',])

# Set inputs
prob.model.set_input_defaults('D_beam', val=0.75*np.ones(elements), units='m')
prob.model.set_input_defaults('wt_beam', val=0.15*np.ones(elements), units='m')
# prob.model.set_input_defaults('D_beam', val=0.00635*np.ones(elements), units='m')
# prob.model.set_input_defaults('wt_beam', val=0.003174*np.ones(elements), units='m')

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

# ## --- Shapes PLOT after Modal
# fig2, ax2 = plt.subplots(figsize=(10,6))

# # Plot shapes
# shape1 = ax2.plot(eig_vecs_xloc, eig_vecs[:,0], label='1st Mode', c='r', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)
# shape2 = ax2.plot(eig_vecs_xloc, eig_vecs[:,1], label='2nd Mode', c='g', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
# shape3 = ax2.plot(eig_vecs_xloc, eig_vecs[:,2], label='3rd Mode', c='b', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)

# # Set axis limits
# # ax.set_xlim([-2, 4])
# # ax.set_ylim([-1, 1.5])
# # ax.set_aspect('equal')

# # Set labels and legend
# ax2.legend()
# ax2.set_title('Modeshapes from Modal')
# ax2.set_xlabel('Deformation (x)')
# ax2.set_ylabel('Length (z)')
# ax2.grid()

# # Show sketch
# plt.show()

## --- Derivatives PLOT
fig3, ax3 = plt.subplots(figsize=(10,6))

# Plot tower
tower_node = ax3.plot(z_towernode, x_towernode_3, label='Nodes', c='r', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)
all_cspl = ax3.plot(z_nodes, all_cubicspline(z_nodes), label='Spl.', c='k', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)

tower_1der_elem = ax3.plot(z_towerelem, x_d_towerelem_3, label='1 der', c='g', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
all_1der_cspl = ax3.plot(z_nodes, all_cubicspline(z_nodes, 1), label='Spl. 1 der', c='k', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
# fd_1der = ax.plot(x_d_lin, z_lin, label='FD', c='k', ls='-', marker='.', ms=7, mfc='g', alpha=0.7)

tower_2der_elem = ax3.plot(z_towerelem, x_dd_towerelem_3, label='2 der', c='b', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)
all_2der_cspl = ax3.plot(z_nodes, all_cubicspline(z_nodes, 2), label='Spl. 2 der', c='k', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)
# fd_2der = ax.plot(x_dd_lin, z_lin, label='FD2', c='k', ls='-', marker='.', ms=7, mfc='c', alpha=0.7)

# Set axis limits
# ax.set_xlim([-2, 4])
# ax.set_ylim([-1, 1.5])
# ax.set_aspect('equal')

# Set labels and legend
ax3.legend()
ax3.set_title('3rd Modeshape Derivatives')
ax3.set_xlabel('Length (z)')
ax3.set_ylabel('Deformation (x)')
ax3.grid()

# Show sketch
plt.show()
plt.tight_layout()
my_path = os.path.dirname(__file__)
fname = 'modeshapes'
plt.savefig(os.path.join(my_path,(fname+'.png')), dpi=400, format='png')
