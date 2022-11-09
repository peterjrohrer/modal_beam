import numpy as np
import openmdao.api as om

from choose_eig_vec import ChooseEigVec

from modeshape_disp import ModeshapeDisp
from beam_node_lhs import BeamNodeLHS
from beam_node_rhs import BeamNodeRHS
from beam_node_deriv import BeamNodeDeriv

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

        self.add_subsystem('beam_node_lhs', 
            BeamNodeLHS(nodal_data=nodal_data, absca='x'), 
            promotes_inputs=['%s_nodes' %'x'], 
            promotes_outputs=['beam_spline_%s_lhs' %'x'])

        self.add_subsystem('beam_y_node_1_rhs', 
            BeamNodeRHS(nodal_data=nodal_data, absca='x', ordin='y', level=1), 
            promotes_inputs=['%s_nodes' %'x', '%s_nodes' %'y'], 
            promotes_outputs=['beam_spline_%s_1_rhs' %'y'])

        beam_y_node_1_deriv = BeamNodeDeriv(nodal_data=nodal_data, absca='x', ordin='y', level=1)
        beam_y_node_1_deriv.linear_solver = om.ScipyKrylov()
        beam_y_node_1_deriv.linear_solver.precon = om.DirectSolver(assemble_jac=True)
        beam_y_node_1_deriv.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        self.add_subsystem('beam_y_node_1_deriv', 
            beam_y_node_1_deriv, 
            promotes_inputs=['beam_spline_%s_lhs' %'x', 'beam_spline_%s_1_rhs' %'y'], 
            promotes_outputs=['%s_d_nodes' %'y'])

        self.add_subsystem('beam_z_node_1_rhs', 
            BeamNodeRHS(nodal_data=nodal_data, absca='x', ordin='z', level=1), 
            promotes_inputs=['%s_nodes' %'x', '%s_nodes' %'z'], 
            promotes_outputs=['beam_spline_%s_1_rhs' %'z'])

        beam_z_node_1_deriv = BeamNodeDeriv(nodal_data=nodal_data, absca='x', ordin='z', level=1)
        beam_z_node_1_deriv.linear_solver = om.ScipyKrylov()
        beam_z_node_1_deriv.linear_solver.precon = om.DirectSolver(assemble_jac=True)
        beam_z_node_1_deriv.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        self.add_subsystem('beam_z_node_1_deriv', 
            beam_z_node_1_deriv, 
            promotes_inputs=['beam_spline_%s_lhs' %'x', 'beam_spline_%s_1_rhs' %'z'], 
            promotes_outputs=['%s_d_nodes' %'z'])

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
