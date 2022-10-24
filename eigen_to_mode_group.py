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
        self.options.declare('mode', types=int)
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        mode_num = self.options['mode']
        nNode = self.options['nNode']        
        nElem = self.options['nElem']        
        nDOF = self.options['nDOF']        
    
        self.add_subsystem('choose_eig_vec',
            ChooseEigVec(mode=mode_num,nDOF=nDOF),
            promotes_inputs=['eig_vector_*'],
            promotes_outputs=['eig_vector'])

        self.add_subsystem('modeshape_disp', 
            ModeshapeDisp(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['eig_vector'], 
            promotes_outputs=['x_beamnode'])

        self.add_subsystem('beam_node_1_lhs', 
            BeamNode1LHS(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_beamnode'], 
            promotes_outputs=['beam_spline_lhs'])

        self.add_subsystem('beam_node_1_rhs', 
            BeamNode1RHS(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_beamnode', 'x_beamnode'], 
            promotes_outputs=['beam_spline_rhs'])

        beam_node_1_deriv = BeamNode1Deriv(nNode=nNode,nElem=nElem,nDOF=nDOF)
        beam_node_1_deriv.linear_solver = om.ScipyKrylov()
        beam_node_1_deriv.linear_solver.precon = om.DirectSolver(assemble_jac=True)
        # beam_node_1_deriv.linear_solver = om.DirectSolver(assemble_jac=True)
        # beam_node_1_deriv.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        self.add_subsystem('beam_node_1_deriv', 
            beam_node_1_deriv, 
            promotes_inputs=['beam_spline_lhs', 'beam_spline_rhs'], 
            promotes_outputs=['x_d_beamnode'])

        self.add_subsystem('beam_elem_disp', 
            BeamElemDisp(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_beamnode', 'z_beamelem', 'x_beamnode', 'x_d_beamnode'], 
            promotes_outputs=['x_beamelem'])

        self.add_subsystem('beam_elem_1_deriv', 
            BeamElem1Deriv(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_beamnode', 'x_beamnode', 'x_d_beamnode'], 
            promotes_outputs=['x_d_beamelem'])

        self.add_subsystem('beam_elem_2_deriv', 
            BeamElem2Deriv(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_beamnode', 'x_d_beamnode'], 
            promotes_outputs=['x_dd_beamelem'])

        self.add_subsystem('modeshape_num',
            ModeshapeNum(mode=mode_num,nNode=nNode,nElem=nElem,nDOF=nDOF),
            promotes_inputs=['x_beamnode', 'x_d_beamnode', 'x_beamelem', 'x_d_beamelem', 'x_dd_beamelem'],
            promotes_outputs=['x_beamnode_%d' % mode_num, 'x_d_beamnode_%d' % mode_num, 'x_beamelem_%d' % mode_num, 'x_d_beamelem_%d' % mode_num, 'x_dd_beamelem_%d' % mode_num])