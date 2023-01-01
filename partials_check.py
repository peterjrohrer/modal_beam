import numpy as np
import myconstants as myconst

import openmdao.api as om

from utils import *
from cantilever_group import Cantilever

## --- Processing nodes (can be done outside of optimization!)
nElem = 6
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

# Set inputs
# prob.model.set_input_defaults('D_beam', val=0.25*np.ones(nElem), units='m')
prob.model.set_input_defaults('D_beam', val=np.linspace(0.5,0.25,nElem), units='m')
prob.model.set_input_defaults('wt_beam', val=0.01*np.ones(nElem), units='m')
prob.model.set_input_defaults('L_beam_tot', val=3., units='m')
prob.model.set_input_defaults('tip_mass', val=25., units='kg')
ref2cog = np.zeros(3)
ref2cog[0] += 0.05
ref2cog[1] += 0.25
ref2cog[2] += 0.15
prob.model.set_input_defaults('ref_to_cog', val=ref2cog, units='m')
tip_inertia = np.zeros((3,3))
tip_inertia[0,0] += 100.
tip_inertia[1,1] += 120.
tip_inertia[0,1] += 50.
tip_inertia[0,2] += 30.
tip_inertia[1,2] += 80.
prob.model.set_input_defaults('tip_inertia', val=tip_inertia, units='kg*m*m')

prob.model.add_subsystem('cantilever', 
    cantilever_group, 
    promotes_inputs=['D_beam', 'wt_beam', 'L_beam_tot', 'tip_mass', 'ref_to_cog', 'tip_inertia'],
    promotes_outputs=['L_beam', 'A_beam', 'Ix_beam', 'Iy_beam', 'M_beam', 'x_beamnode', 'y_beamnode', 'z_beamnode', 'tot_M', 'Q', 'eigfreqs', 'x_nodes', 'y_nodes', 'z_nodes', 'M_modal', 'K_modal'])

# Setup and run problem
prob.setup(derivatives=True, force_alloc_complex=True)
prob.set_solver_print(level=1)
prob.run_model()

comp_to_check = 'cantilever.fem_group.eigenvectors'
apart_tol = 1.e-5
rpart_tol = 1.e-6

# check_partials_data = prob.check_partials(method='fd', form='central', abs_err_tol=apart_tol, rel_err_tol=rpart_tol, step_calc='rel_avg', step=1e-8, show_only_incorrect=True, compact_print=True)
check_partials_data = prob.check_partials(method='fd', form='forward', includes=comp_to_check, step_calc='rel_element', step=1e-4, show_only_incorrect=False, compact_print=True)
# check_partials_data = prob.check_partials(method='cs', includes=comp_to_check, show_only_incorrect=False, compact_print=True)

om.partial_deriv_plot('Q_raw', 'Ar_eig', check_partials_data, binary=True)

# prob.check_totals(of=['tot_M','eigfreqs'], wrt=['D_beam', 'wt_beam'])