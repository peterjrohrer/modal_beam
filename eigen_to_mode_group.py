import numpy as np
import openmdao.api as om

from modeshape_disp import ModeshapeDisp
from beam_node_lhs import BeamNodeLHS
from beam_node_rhs import BeamNodeRHS
from beam_node_deriv import BeamNodeDeriv

from beam_elem_disp import BeamElemDisp
from beam_elem_1_deriv import BeamElem1Deriv
from beam_elem_2_deriv import BeamElem2Deriv

class Eig2Mode(om.Group):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        nodal_data = self.options['nodal_data']     

        self.add_subsystem('modeshape_disp', 
            ModeshapeDisp(nodal_data=nodal_data), 
            promotes_inputs=['Q', 'x_beamnode', 'y_beamnode', 'z_beamnode'], 
            promotes_outputs=['x_nodes', 'y_nodes', 'z_nodes'])

        for i in ['x', 'y', 'z']:
            self.add_subsystem('beam_node_%s_lhs' %i, 
                BeamNodeLHS(nodal_data=nodal_data, absca=i), 
                promotes_inputs=['%s_nodes' %i], 
                promotes_outputs=['beam_spline_%s_lhs' %i])

        for i in ['y', 'z']:
            absca = 'x' 
            # -- First derivative
            self.add_subsystem('beam_%s_node_1_rhs' %(i), 
                BeamNodeRHS(nodal_data=nodal_data, absca=absca, ordin=i, level=1), 
                promotes_inputs=['%s_nodes' %absca, '%s_nodes' %i], 
                promotes_outputs=['beam_spline_%s_1_rhs' %(i)])

            beam_y_node_1_deriv = BeamNodeDeriv(nodal_data=nodal_data, absca=absca, ordin=i, level=1)
            beam_y_node_1_deriv.linear_solver = om.ScipyKrylov()
            beam_y_node_1_deriv.linear_solver.precon = om.DirectSolver(assemble_jac=True)
            beam_y_node_1_deriv.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

            self.add_subsystem('beam_%s_node_1_deriv' %(i),
                beam_y_node_1_deriv, 
                promotes_inputs=['beam_spline_%s_lhs' %absca, 'beam_spline_%s_1_rhs' %(i)], 
                promotes_outputs=['%s_d_nodes' %(i)])

            # -- Second derivative
            self.add_subsystem('beam_%s_node_2_rhs' %(i), 
                BeamNodeRHS(nodal_data=nodal_data, absca=absca, ordin=i, level=2), 
                promotes_inputs=['%s_nodes' %absca, '%s_nodes' %i], 
                promotes_outputs=['beam_spline_%s_2_rhs' %(i)])

            beam_y_node_2_deriv = BeamNodeDeriv(nodal_data=nodal_data, absca=absca, ordin=i, level=2)
            beam_y_node_2_deriv.linear_solver = om.ScipyKrylov()
            beam_y_node_2_deriv.linear_solver.precon = om.DirectSolver(assemble_jac=True)
            beam_y_node_2_deriv.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

            self.add_subsystem('beam_%s_node_2_deriv' %(i),
                beam_y_node_2_deriv, 
                promotes_inputs=['beam_spline_%s_lhs' %absca, 'beam_spline_%s_2_rhs' %(i)], 
                promotes_outputs=['%s_dd_nodes' %(i)])

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
