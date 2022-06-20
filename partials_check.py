# %% 
import numpy as np
import myconstants as myconst

import openmdao.api as om
from cantilever_group import Cantilever

# Bring in problem with defined defaults
prob = om.Problem()

elements = 3
# cantilever_group = Cantilever(nNode=11, nElem=10, nDOF=20) # 20 DOF because of cantilever BC
cantilever_group = Cantilever(nNode=(elements+1), nElem=elements, nDOF=(2*elements)) # Increased nodes
# cantilever_group.linear_solver = om.DirectSolver(assemble_jac=True)
# cantilever_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6, use_aitken=True)

prob.model.add_subsystem('cantilever', 
    cantilever_group, 
    promotes_inputs=[],
    promotes_outputs=['M_global', 'K_global', 'Z_beam', 'D_beam', 'L_beam', 'M_beam', 'tot_M_beam', 'wt_beam',
        'eig_vector_*', 'eig_freq_*', 'z_beamnode', 'z_beamelem',
        'x_beamnode_*', 'x_d_beamnode_*', 'x_beamelem_*', 'x_d_beamelem_*', 'x_dd_beamelem_*',])

#  Set inputs
# prob.model.set_input_defaults('water_depth', val=10., units='m')

# Setup and run problem
prob.setup(force_alloc_complex=True)
prob.set_solver_print(1)
prob.run_model()

comp_to_check = 'cantilever.modeshape_group.modeshape_eig_lhs_2'
apart_tol = 1.e-6
rpart_tol = 1.e-6

# check_partials_data = prob.check_partials(method='fd', form='central', abs_err_tol=apart_tol, rel_err_tol=rpart_tol, step_calc='rel_avg', step=1e-8, show_only_incorrect=True, compact_print=True)

check_partials_data = prob.check_partials(method='fd',form='central', includes=comp_to_check, step=1e-8, show_only_incorrect=False, compact_print=False)
# check_partials_data = prob.check_partials(method='cs', includes=comp_to_check, show_only_incorrect=True, compact_print=True)

om.partial_deriv_plot('lhs','M_mode', check_partials_data)
