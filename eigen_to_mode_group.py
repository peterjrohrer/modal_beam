import numpy as np
import openmdao.api as om

from choose_eig_vec import ChooseEigVec

from modeshape_disp import ModeshapeDisp
from beam_node_1_lhs import BeamNode1LHS
from beam_node_1_rhs import BeamNode1RHS
from beam_node_1_deriv import BeamNode1Deriv

from beam_elem_disp import BeamElemDisp
from beam_elem_1_deriv import BeamElem1Deriv
from beam_elem_2_deriv import BeamElem2Deriv

from modeshape_num import ModeshapeNum

class Eig2Mode(om.Group):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        nodal_data = self.options['nodal_data']     

        self.add_subsystem('modeshape_disp', 
            ModeshapeDisp(nodal_data=nodal_data), 
            promotes_inputs=['Q', 'x_beamnode', 'y_beamnode', 'z_beamnode'], 
            promotes_outputs=['x_nodes', 'y_nodes', 'z_nodes'])

        # self.add_subsystem('beam_node_1_lhs', 
        #     BeamNode1LHS(nNode=nNode,nElem=nElem,nDOF=nDOF), 
        #     promotes_inputs=['z_beamnode'], 
        #     promotes_outputs=['beam_spline_lhs'])

        # self.add_subsystem('beam_node_1_rhs', 
        #     BeamNode1RHS(nNode=nNode,nElem=nElem,nDOF=nDOF), 
        #     promotes_inputs=['z_beamnode', 'x_beamnode'], 
        #     promotes_outputs=['beam_spline_rhs'])

        # beam_node_1_deriv = BeamNode1Deriv(nNode=nNode,nElem=nElem,nDOF=nDOF)
        # beam_node_1_deriv.linear_solver = om.ScipyKrylov()
        # beam_node_1_deriv.linear_solver.precon = om.DirectSolver(assemble_jac=True)
        # # beam_node_1_deriv.linear_solver = om.DirectSolver(assemble_jac=True)
        # # beam_node_1_deriv.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        # self.add_subsystem('beam_node_1_deriv', 
        #     beam_node_1_deriv, 
        #     promotes_inputs=['beam_spline_lhs', 'beam_spline_rhs'], 
        #     promotes_outputs=['x_d_beamnode'])

        # self.add_subsystem('beam_elem_disp', 
        #     BeamElemDisp(nNode=nNode,nElem=nElem,nDOF=nDOF), 
        #     promotes_inputs=['z_beamnode', 'z_beamelem', 'x_beamnode', 'x_d_beamnode'], 
        #     promotes_outputs=['x_beamelem'])

        # self.add_subsystem('beam_elem_1_deriv', 
        #     BeamElem1Deriv(nNode=nNode,nElem=nElem,nDOF=nDOF), 
        #     promotes_inputs=['z_beamnode', 'x_beamnode', 'x_d_beamnode'], 
        #     promotes_outputs=['x_d_beamelem'])

        # self.add_subsystem('beam_elem_2_deriv', 
        #     BeamElem2Deriv(nNode=nNode,nElem=nElem,nDOF=nDOF), 
        #     promotes_inputs=['z_beamnode', 'x_d_beamnode'], 
        #     promotes_outputs=['x_dd_beamelem'])
