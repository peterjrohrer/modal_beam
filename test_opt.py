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

elements = 3
cantilever_group = Cantilever(nNode=(elements+1), nElem=elements, nDOF=(2*elements)) # Increased nodes
# cantilever_group.linear_solver = om.DirectSolver(assemble_jac=True)
# cantilever_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6, use_aitken=True)

prob.model.add_subsystem('cantilever', 
    cantilever_group, 
    promotes_inputs=['D_beam', 'wt_beam'],
    promotes_outputs=['M_global', 'K_global', 'Z_beam', 'L_beam', 'M_beam', 'tot_M_beam',
        'eig_vector_*', 'eig_freq_*', 'z_beamnode', 'z_beamelem',
        'x_beamnode_*', 'x_d_beamnode_*', 'x_beamelem_*', 'x_d_beamelem_*', 'x_dd_beamelem_*',])

# Set driver/recorder
prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
prob.driver.options['debug_print'] = ['desvars','objs','nl_cons']

# Create a recorder variable
recorder = om.SqliteRecorder('cases.sql')
# Attach a recorder to the problem
prob.add_recorder(recorder)

# Set variables/defaults/objective/constraints
prob.model.set_input_defaults('D_beam', val=0.75*np.ones(elements), units='m')
prob.model.set_input_defaults('wt_beam', val=0.15*np.ones(elements), units='m')

prob.model.add_design_var('D_beam', lower=0.4*np.ones(elements), upper=3.0*np.ones(elements))
prob.model.add_design_var('wt_beam', lower=0.1*np.ones(elements), upper=0.25*np.ones(elements), ref0=0.09, ref=0.26)
prob.model.add_constraint('eig_freq_1', lower=0.1, ref0=0.09, ref=0.5)
prob.model.add_objective('tot_M_beam', ref0=50000., ref=100000.)

# Setup and run problem
# prob.setup(force_alloc_complex=True)
# prob.set_solver_print(1)
# prob.run_driver()
# prob.record('after_run_driver')

# Setup and check partials
prob.setup(mode='rev', force_alloc_complex=True)
prob.run_model()
prob.check_totals(of=['tot_M_beam', 'eig_freq_1'], wrt=['D_beam', 'wt_beam'])

# ## --- Debugging Prints
# print('-----------------------------------------------------------------')
# # Instantiate CaseReader
# cr = om.CaseReader('cases.sql')
# driver_cases = cr.list_cases('problem', out_stream=None)
# last_case = cr.get_case('after_run_driver')

# # print('Minimum: D = %2.2f m, t = %2.3f m' %(last_case.get_val('diameter'), last_case.get_val('thickness')))
# print('diameters:')
# print(last_case.get_val('D_beam'))
# print('thicknesses:')
# print(last_case.get_val('wt_beam'))
# print('m = %2.2f kg, Lowest Freq %2.2f Hz' %(last_case.get_val('tot_M_beam'), last_case.get_val('eig_freq_1')))
# if last_case.get_val('eig_freq_1') < 0.1:
#     print('Constraint violated! f < 0.1 Hz')
# elif last_case.get_val('eig_freq_1') >= 0.1:
#     print('Constraint: f >= 0.1 Hz')


# print('-----------------------------------------------------------------')
# print('Mode 1 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_1']), (2*np.pi*float(prob['eig_freq_1'])), (float(prob['eig_freq_1']))))
# print('Mode 2 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_2']), (2*np.pi*float(prob['eig_freq_2'])), (float(prob['eig_freq_2']))))
# print('Mode 3 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_3']), (2*np.pi*float(prob['eig_freq_3'])), (float(prob['eig_freq_3']))))

# ## --- Check Eigenvals
# M_glob = prob['M_global'] 
# K_glob = prob['K_global']

# M_glob_inv = np.linalg.inv(M_glob)
# eig_mat = np.matmul(M_glob_inv, K_glob)
# eig_vals_raw, eig_vecs = np.linalg.eig(eig_mat)
# eig_vals = np.sqrt(np.real(np.sort(eig_vals_raw))) 

# print('Check Eigenfrequencies: %3.3f rad/s, %3.3f rad/s, %3.3f rad/s' % (eig_vals[0], eig_vals[1], eig_vals[2]))
# print('-----------------------------------------------------------------')
