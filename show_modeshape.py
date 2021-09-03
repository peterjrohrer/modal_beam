#%% Run as Notebook
import numpy as np
import openmdao.api as om
import myconstants as myconst

from matplotlib import rc
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

from cantilever_group import Cantilever


# Bring in problem with defined defaults
prob = om.Problem()

cantilever_group = Cantilever(nNode=11, nElem=10, nDOF=20) # 20 DOF because of cantilever BC
# cantilever_group = Cantilever(nNode=21, nElem=20, nDOF=40) # Increased nodes
# cantilever_group.linear_solver = om.DirectSolver(assemble_jac=True)
# cantilever_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6, use_aitken=True)

prob.model.add_subsystem('cantilever', 
    cantilever_group, 
    promotes_inputs=[],
    promotes_outputs=['M_global', 'K_global',
        'eig_vector_*', 'eig_freq_*', 'z_towernode', 'z_towerelem',
        'x_towernode_*', 'x_d_towernode_*', 'x_towerelem_*', 'x_d_towerelem_*', 'x_dd_towerelem_*',])

#  Set inputs
# prob.model.set_input_defaults('water_depth', val=10., units='m')

# Setup and run problem
prob.setup()
prob.set_solver_print(1)
prob.run_model()

print('Mode 1 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_1']), (2*np.pi*float(prob['eig_freq_1'])), (float(prob['eig_freq_1']))))
print('Mode 2 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_2']), (2*np.pi*float(prob['eig_freq_2'])), (float(prob['eig_freq_2']))))
print('Mode 3 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_3']), (2*np.pi*float(prob['eig_freq_3'])), (float(prob['eig_freq_3']))))

## --- Check Eigenvals
M_glob = prob['M_global'] 
K_glob = prob['K_global']

M_glob_inv = np.linalg.inv(M_glob)
# eig_mat = np.matmul(M_glob_inv, K_glob)/10000.
eig_mat = np.matmul(M_glob_inv, K_glob)
eig_vals_raw, eig_vecs = np.linalg.eig(eig_mat)
eig_vals = np.sqrt(np.real(np.sort(eig_vals_raw))) 
eig_vecs_xloc = np.linspace(0,1,3)

print('Eigenfrequencies: %3.3f rad/s, %3.3f rad/s, %3.3f rad/s' % (eig_vals[0], eig_vals[1], eig_vals[2]))

## --- Pull out Modeshape
x_towernode_1 = prob['x_towernode_1']
x_towernode_2 = prob['x_towernode_2']
x_towernode_3 = prob['x_towernode_3']
x_towerelem_1 = prob['x_towerelem_1']
x_towerelem_2 = prob['x_towerelem_2']
x_towerelem_3 = prob['x_towerelem_3']
x_d_towernode_1 = prob['x_d_towernode_1']
x_d_towernode_2 = prob['x_d_towernode_2']
x_d_towernode_3 = prob['x_d_towernode_3']
x_d_towerelem_1 = prob['x_d_towerelem_1']
x_d_towerelem_2 = prob['x_d_towerelem_2']
x_d_towerelem_3 = prob['x_d_towerelem_3']
x_dd_towerelem_1 = prob['x_dd_towerelem_1']
x_dd_towerelem_2 = prob['x_dd_towerelem_2']
x_dd_towerelem_3 = prob['x_dd_towerelem_3']
z_towernode = prob['z_towernode']
z_towerelem = prob['z_towerelem']

mode1_freq = (2*np.pi*float(prob['eig_freq_1']))
mode2_freq = (2*np.pi*float(prob['eig_freq_2']))
mode3_freq = (2*np.pi*float(prob['eig_freq_3']))

## -- SciPy Spline Calculation

z_nodes = z_towernode
x_nodes = x_towernode_3
# all_cubicspline = CubicSpline(z_nodes, x_nodes, bc_type='not-a-knot')
# all_cubicspline = CubicSpline(z_nodes, x_nodes, bc_type='natural')
all_cubicspline = CubicSpline(z_nodes, x_nodes, bc_type=((1,0.),(1,0.)))
# all_cubicspline = CubicSpline(z_nodes, x_nodes, bc_type=((1,spar_base_1der),(1,tower_top_1der)))

## --- Finite Differencing

# # First based on data points from spline
# z_range = z_nodes[-1] - z_nodes[0]
# h = z_range/100.
# z_lin = np.linspace(z_nodes[0],z_nodes[-1], 101)
# x_lin = all_cubicspline(z_lin)
# x_d_lin = np.zeros_like(x_lin)
# x_dd_lin = np.zeros_like(x_lin)

# for i in range(1,99):
#     x_d_lin[i] = (x_lin[i+1] - x_lin[i-1])/(2.*h)

# for i in range(1,99):
#     x_dd_lin[i] = (x_d_lin[i+1] - x_d_lin[i-1])/(2.*h)

## --- Shapes PLOT from FEA
font = {'size': 16}
plt.rc('font', **font)
fig1, ax1 = plt.subplots(figsize=(9,6))

# Plot shapes
shape1 = ax1.plot(z_towernode, x_towernode_1, label='1st Mode', c='r', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)
shape2 = ax1.plot(z_towernode, x_towernode_2, label='2nd Mode', c='g', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
shape3 = ax1.plot(z_towernode, x_towernode_3, label='3rd Mode', c='b', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)

# Set labels and legend
ax1.legend()
ax1.set_title('Modeshapes from FEA')
ax1.set_xlabel('Deformation (x)')
ax1.set_ylabel('Length (z)')
ax1.grid()

## --- Shapes PLOT after Modal
fig2, ax2 = plt.subplots(figsize=(10,6))

# Plot shapes
shape1 = ax2.plot(eig_vecs_xloc, eig_vecs[:,0], label='1st Mode', c='r', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)
shape2 = ax2.plot(eig_vecs_xloc, eig_vecs[:,1], label='2nd Mode', c='g', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
shape3 = ax2.plot(eig_vecs_xloc, eig_vecs[:,2], label='3rd Mode', c='b', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)

# Set axis limits
# ax.set_xlim([-2, 4])
# ax.set_ylim([-1, 1.5])
# ax.set_aspect('equal')

# Set labels and legend
ax2.legend()
ax2.set_title('3rd Modeshape Derivatives')
ax2.set_xlabel('Deformation (x)')
ax2.set_ylabel('Length (z)')
ax2.grid()

# Show sketch
plt.show()

# ## --- Derivatives PLOT
# fig3, ax3 = plt.subplots(figsize=(10,6))

# # Plot tower
# tower_node = ax3.plot(z_towernode, x_towernode_3, label='Nodes', c='r', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)
# all_cspl = ax3.plot(z_nodes, all_cubicspline(z_nodes), label='Spl.', c='k', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)

# tower_1der_elem = ax3.plot(z_towerelem, x_d_towerelem_3, label='1 der', c='g', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
# all_1der_cspl = ax3.plot(z_nodes, all_cubicspline(z_nodes, 1), label='Spl. 1 der', c='k', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
# # fd_1der = ax.plot(x_d_lin, z_lin, label='FD', c='k', ls='-', marker='.', ms=7, mfc='g', alpha=0.7)

# tower_2der_elem = ax3.plot(z_towerelem, x_dd_towerelem_3, label='2 der', c='b', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)
# all_2der_cspl = ax3.plot(z_nodes, all_cubicspline(z_nodes, 2), label='Spl. 2 der', c='k', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)
# # fd_2der = ax.plot(x_dd_lin, z_lin, label='FD2', c='k', ls='-', marker='.', ms=7, mfc='c', alpha=0.7)

# # Set axis limits
# # ax.set_xlim([-2, 4])
# # ax.set_ylim([-1, 1.5])
# # ax.set_aspect('equal')

# # Set labels and legend
# ax3.legend()
# ax3.set_title('3rd Modeshape Derivatives')
# ax3.set_xlabel('Deformation (x)')
# ax3.set_ylabel('Length (z)')
# ax3.grid()

# # Show sketch
# plt.show()

# SHOW SHAPES
# font = {'size': 16}
# plt.rc('font', **font)
# fig, ax = plt.subplots(figsize=(8,10))

# # Plot spar
# spar_1 = ax.plot(x_sparnode_1, z_sparnode, label='Spar Mode 1, %2.2f rad/s' % mode1_freq, c='r', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)
# spar_1_lin = ax.plot([x_sparnode_1[0],x_sparnode_1[-1]], [z_sparnode[0],z_sparnode[-1]], c='r', ls='--', lw=0.75)

# spar_2 = ax.plot(x_sparnode_2, z_sparnode, label='Spar Mode 2, %2.2f rad/s' % mode2_freq, c='g', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
# spar_2_lin = ax.plot([x_sparnode_2[0],x_sparnode_2[-1]], [z_sparnode[0],z_sparnode[-1]], c='g', ls='--', lw=0.75)

# spar_3 = ax.plot(x_sparnode_3, z_sparnode, label='Spar Mode 3, %2.2f rad/s' % mode3_freq, c='b', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)
# spar_3_lin = ax.plot([x_sparnode_3[0],x_sparnode_3[-1]], [z_sparnode[0],z_sparnode[-1]], c='b', ls='--', lw=0.75)

# # Plot tower
# tower_1 = ax.plot(x_towernode_1, z_towernode, label='Tower Mode 1', c='m', ls='-', marker='.', ms=10, mfc='m', alpha=0.7)
# tower_1_lin = ax.plot([x_towernode_1[0],x_towernode_1[-1]], [z_towernode[0],z_towernode[-1]], c='m', ls='--', lw=0.75)

# tower_2 = ax.plot(x_towernode_2, z_towernode, label='Tower Mode 2', c='y', ls='-', marker='.', ms=10, mfc='y', alpha=0.7)
# tower_2_lin = ax.plot([x_towernode_2[0],x_towernode_2[-1]], [z_towernode[0],z_towernode[-1]], c='y', ls='--', lw=0.75)

# tower_3 = ax.plot(x_towernode_3, z_towernode, label='Tower Mode 3', c='k', ls='-', marker='.', ms=10, mfc='k', alpha=0.7)
# tower_3_lin = ax.plot([x_towernode_3[0],x_towernode_3[-1]], [z_towernode[0],z_towernode[-1]], c='k', ls='--', lw=0.75)

# # Add waterline
# ax.axhline(y=0., zorder=100., color='b', label='SWL')

# # Set axis limits
# ax.set_xlim([-2, 4])
# ax.set_ylim([-1, 1.5])
# # ax.set_aspect('equal')

# # Set labels and leged
# ax.legend()
# ax.set_xlabel('Radial Distance from Column Center [m]')
# ax.set_ylabel('Vertical Distance from SWL [m]')
# ax.grid()

# # Save sketch
# plt.show()
# plt.tight_layout()
# my_path = os.path.dirname(__file__)
# fname = 'modeshapes'
# # plt.savefig(os.path.join(my_path,(fname+'.png')), dpi=400, format='png')
# # plt.savefig(os.path.join(my_path,(fname+'.eps')), format='eps')
# %%
