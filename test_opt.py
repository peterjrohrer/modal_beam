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

elements = 10
cantilever_group = Cantilever(nNode=(elements+1), nElem=elements, nDOF=(2*elements)) # Increased nodes
# cantilever_group.linear_solver = om.DirectSolver(assemble_jac=True)
# cantilever_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6, use_aitken=True)

prob.model.add_subsystem('cantilever', 
    cantilever_group, 
    promotes_inputs=['diameter', 'thickness'],
    promotes_outputs=['M_global', 'K_global', 'Z_beam', 'D_beam', 'L_beam', 'M_beam', 'tot_M_beam', 'wt_beam',
        'eig_vector_*', 'eig_freq_*', 'z_beamnode', 'z_beamelem',
        'x_beamnode_*', 'x_d_beamnode_*', 'x_beamelem_*', 'x_d_beamelem_*', 'x_dd_beamelem_*',])

# Set driver
prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['debug_print'] = ['desvars','objs']

# Set variables/defaults/objective/constraints
prob.model.set_input_defaults('diameter', val=0.15, units='m')
prob.model.set_input_defaults('thickness', val=0.05, units='m')

prob.model.add_design_var('diameter', lower=0.1, upper=1.0, ref0=0.09)
prob.model.add_design_var('thickness', lower=0.01, upper=0.1, ref0=0.009, ref=0.11)
prob.model.add_constraint('eig_freq_1', lower=0.035, ref0=0.025, ref=0.045)
prob.model.add_objective('tot_M_beam', ref0=5000., ref=10000.)

# Setup and run problem
prob.setup(mode='rev', force_alloc_complex=True)
prob.run_driver()

# # Setup and check partials
# prob.setup(mode='rev', force_alloc_complex=True)
# prob.run_model()
# prob.check_totals(of=['tot_M_beam', 'eig_freq_1'], wrt=['diameter', 'thickness'])

print('Mode 1 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_1']), (2*np.pi*float(prob['eig_freq_1'])), (float(prob['eig_freq_1']))))
print('Mode 2 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_2']), (2*np.pi*float(prob['eig_freq_2'])), (float(prob['eig_freq_2']))))
print('Mode 3 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_3']), (2*np.pi*float(prob['eig_freq_3'])), (float(prob['eig_freq_3']))))

## --- Check Eigenvals
M_glob = prob['M_global'] 
K_glob = prob['K_global']

M_glob_inv = np.linalg.inv(M_glob)
eig_mat = np.matmul(M_glob_inv, K_glob)
eig_vals_raw, eig_vecs = np.linalg.eig(eig_mat)
eig_vals = np.sqrt(np.real(np.sort(eig_vals_raw))) 
eig_vecs_xloc = np.linspace(0,1,3)

print('Check Eigenfrequencies: %3.3f rad/s, %3.3f rad/s, %3.3f rad/s' % (eig_vals[0], eig_vals[1], eig_vals[2]))

# ## --- Pull out Modeshape
# x_beamnode_1 = prob['x_beamnode_1']
# x_beamnode_2 = prob['x_beamnode_2']
# x_beamnode_3 = prob['x_beamnode_3']
# x_beamelem_1 = prob['x_beamelem_1']
# x_beamelem_2 = prob['x_beamelem_2']
# x_beamelem_3 = prob['x_beamelem_3']
# x_d_beamnode_1 = prob['x_d_beamnode_1']
# x_d_beamnode_2 = prob['x_d_beamnode_2']
# x_d_beamnode_3 = prob['x_d_beamnode_3']
# x_d_beamelem_1 = prob['x_d_beamelem_1']
# x_d_beamelem_2 = prob['x_d_beamelem_2']
# x_d_beamelem_3 = prob['x_d_beamelem_3']
# x_dd_beamelem_1 = prob['x_dd_beamelem_1']
# x_dd_beamelem_2 = prob['x_dd_beamelem_2']
# x_dd_beamelem_3 = prob['x_dd_beamelem_3']
# z_beamnode = prob['z_beamnode']
# z_beamelem = prob['z_beamelem']

# mode1_freq = (2*np.pi*float(prob['eig_freq_1']))
# mode2_freq = (2*np.pi*float(prob['eig_freq_2']))
# mode3_freq = (2*np.pi*float(prob['eig_freq_3']))

# ## --- Shapes PLOT from FEA
# font = {'size': 16}
# plt.rc('font', **font)
# fig1, ax1 = plt.subplots(figsize=(12,8))

# # Plot shapes
# shape1 = ax1.plot(z_beamnode, x_beamnode_1, label='1st Mode: %2.2f rad/s' %mode1_freq, c='r', ls='-', marker='.', ms=10, mfc='r', alpha=0.7)
# shape2 = ax1.plot(z_beamnode, x_beamnode_2, label='2nd Mode: %2.2f rad/s' %mode2_freq, c='g', ls='-', marker='.', ms=10, mfc='g', alpha=0.7)
# shape3 = ax1.plot(z_beamnode, x_beamnode_3, label='3rd Mode: %2.2f rad/s' %mode3_freq, c='b', ls='-', marker='.', ms=10, mfc='b', alpha=0.7)

# # Set labels and legend
# ax1.legend()
# ax1.set_title('Modeshapes from FEA')
# ax1.set_xlabel('Length (z)')
# ax1.set_ylabel('Deformation (x)')
# ax1.grid()

# plt.show()