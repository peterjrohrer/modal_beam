#%% Run as Notebook
import numpy as np
import openmdao.api as om
import myconstants as myconst
import os

from matplotlib import rc
import matplotlib.pyplot as plt

from utils import *
from cantilever_group import Cantilever

## --- Processing nodes (can be done outside of optimization!)
nElem = 5
nNode = nElem + 1
nDOFperNode =  6
nNodeperElem =  2
Elem2Nodes, Nodes2DOF, Elem2DOF = LinearDOFMapping(nElem, nNodeperElem, nDOFperNode)
nDOF_tot = nDOFperNode * nNode
IDOF_All = np.arange(0,nDOF_tot)
# Tip and root degrees of freedom
IDOF_root = Nodes2DOF[Elem2Nodes[0,:][0] ,:]
IDOF_tip  = Nodes2DOF[Elem2Nodes[-1,:][1],:]
IDOF_tip2  = Nodes2DOF[Elem2Nodes[-2,:][1],:]
# Handle BC and root/tip conditions
BC_root = [0,0,0,0,0,0]
BC_tip  = [1,1,1,1,1,1]
# Boundary condition transformation matrix (removes row/columns)
Tr=np.eye(nDOF_tot)
# Root and Tip BC
IDOF_removed = [i for i,iBC in zip(IDOF_root, BC_root) if iBC==0]
IDOF_removed += [i for i,iBC in zip(IDOF_tip, BC_tip) if iBC==0]
Tr = np.delete(Tr, IDOF_removed, axis=1) # removing columns
nDOF_r = Tr.shape[1]
# Rigid linking
Tr[IDOF_tip,IDOF_tip2] += 1.
Tr[14,5] += 0.75
# Construct mask and find non-zeros for taking partials later
Tr_mask = np.nonzero(Tr-np.eye(nDOF_tot,nDOF_r)) # a tuple of nonzero values
Tr_part_mask = np.nonzero((Tr-np.eye(nDOF_tot,nDOF_r)).flatten())[0]

# --- Create mapping from M to Mr
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

nMode = 3

## Rotation Matrices
DCM_beam = elementDCMforHorizontal(nElem=nElem)
DCM_col = elementDCMforVertical(20)
DCM_pont = elementDCMforPontoons(10,3)
RR = transformMatrixfromDCM(DCM=DCM_beam)

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
    'Tr_mask': Tr_mask, 
    'Tr_part_mask': Tr_part_mask, 
    'nDOF_r': nDOF_r,
    'nMode': nMode,
    'DCM': DCM_beam,
    'RR': RR,
}

# Bring in problem with defined defaults
prob = om.Problem()
cantilever_group = Cantilever(nodal_data=nodal_data) # Increased nodes
# cantilever_group.linear_solver = om.DirectSolver(assemble_jac=True)
# cantilever_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6, use_aitken=True)

prob.model.add_subsystem('cantilever', 
    cantilever_group, 
    promotes_inputs=['D_beam', 'wt_beam', 'L_beam_tot', 'tip_mass', 'ref_to_cog', 'tip_inertia'],
    promotes_outputs=['L_beam', 'A_beam', 'Ix_beam', 'Iy_beam', 'M_beam', 'x_beamnode', 'y_beamnode', 'z_beamnode', 'tot_M', 'Q', 'eigfreqs', 'x_nodes', 'y_nodes', 'z_nodes', 'M_modal', 'K_modal'])

# Set driver/recorder
prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
prob.driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs', 'totals']
prob.driver.opt_settings['Major feasibility tolerance'] = 1.0e-3
prob.driver.opt_settings['Major optimality tolerance'] = 1.0e-3
prob.driver.opt_settings['Major iterations limit'] = 25
prob.driver.opt_settings['Major print level'] = 1
prob.driver.opt_settings['Print frequency'] = 10

# Create a recorder variable
recorder = om.SqliteRecorder('cases.sql')
# Attach a recorder to the problem
prob.add_recorder(recorder)

# Set variables/defaults/objective/constraints
prob.model.set_input_defaults('D_beam', val=0.25*np.ones(nElem), units='m')
prob.model.set_input_defaults('wt_beam', val=0.01*np.ones(nElem), units='m')
prob.model.set_input_defaults('L_beam_tot', val=5., units='m')
prob.model.set_input_defaults('tip_mass', val=100., units='kg')
ref2cog = np.zeros(3)
ref2cog[0] += 0.05
ref2cog[1] += 0.25
ref2cog[2] += 0.15
prob.model.set_input_defaults('ref_to_cog', val=ref2cog, units='m')
tip_inertia = np.zeros((3,3))
tip_inertia[0,0] += 1000.
tip_inertia[1,1] += 1200.
tip_inertia[0,1] += 500.
tip_inertia[0,2] += 300.
tip_inertia[1,2] += 800.
prob.model.set_input_defaults('tip_inertia', val=tip_inertia, units='kg*m*m')

prob.model.add_design_var('D_beam', lower=0.4*np.ones(nElem), upper=3.0*np.ones(nElem))
prob.model.add_design_var('wt_beam', lower=0.005*np.ones(nElem), upper=0.25*np.ones(nElem), ref0=0.09, ref=0.26)
prob.model.add_constraint('eigfreqs', indices=[0], lower=1.)
prob.model.add_objective('tot_M', ref0=50000., ref=100000.)

# Setup and run problem
# prob.setup(force_alloc_complex=True)
# prob.set_solver_print(1)
# prob.run_driver()
# prob.record('after_run_driver')

# Setup and check partials
prob.setup(force_alloc_complex=True)
prob.run_model()
prob.check_totals(of=['tot_M','eigfreqs'], wrt=['D_beam', 'wt_beam'])

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
