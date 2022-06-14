from re import I
import numpy as np
import myconstants as myconst

import openmdao.api as om
from cantilever_group import Cantilever

# Bring in problem with defined defaults
prob = om.Problem()

# cantilever_group = Cantilever(nNode=11, nElem=10, nDOF=20) # 20 DOF because of cantilever BC
cantilever_group = Cantilever(nNode=11, nElem=10, nDOF=20) # Increased nodes
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
prob.setup()
prob.set_solver_print(1)
prob.run_model()

comp_to_check = 'cantilever_group.modeshape_group.modeshape_eigvector'

check_partials_data = prob.check_partials(method='fd',form='central',step=1e-6, show_only_incorrect=True, compact_print=True)
# check_partials_data = prob.check_partials(method='cs', includes=comp_to_check, show_only_incorrect=True, compact_print=True)

# om.partial_deriv_plot('normforce_mode_elem', 'Z_beam', check_partials_data, binary=False)
