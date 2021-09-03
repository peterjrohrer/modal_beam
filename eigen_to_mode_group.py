import numpy as np
import openmdao.api as om

from choose_eig_vec import ChooseEigVec

from modeshape_disp import ModeshapeDisp
from tower_node_1_lhs import TowerNode1LHS
from tower_node_1_rhs import TowerNode1RHS
from tower_node_1_deriv import TowerNode1Deriv

from tower_elem_disp import TowerElemDisp
from tower_elem_1_deriv import TowerElem1Deriv
from tower_elem_2_deriv import TowerElem2Deriv

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
            promotes_outputs=['x_towernode'])

        self.add_subsystem('tower_node_1_lhs', 
            TowerNode1LHS(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_towernode'], 
            promotes_outputs=['tower_spline_lhs'])

        self.add_subsystem('tower_node_1_rhs', 
            TowerNode1RHS(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_towernode', 'x_towernode'], 
            promotes_outputs=['tower_spline_rhs'])

        tower_node_1_deriv = TowerNode1Deriv(nNode=nNode,nElem=nElem,nDOF=nDOF)
        tower_node_1_deriv.linear_solver = om.ScipyKrylov()
        tower_node_1_deriv.linear_solver.precon = om.DirectSolver(assemble_jac=True)
        #tower_node_1_deriv.linear_solver = om.DirectSolver(assemble_jac=True)

        self.add_subsystem('tower_node_1_deriv', 
            tower_node_1_deriv, 
            promotes_inputs=['tower_spline_lhs', 'tower_spline_rhs'], 
            promotes_outputs=['x_d_towernode'])

        self.add_subsystem('tower_elem_disp', 
            TowerElemDisp(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_towernode', 'z_towerelem', 'Z_tower', 'x_towernode', 'x_d_towernode'], 
            promotes_outputs=['x_towerelem'])

        self.add_subsystem('tower_elem_1_deriv', 
            TowerElem1Deriv(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_towernode', 'Z_tower', 'x_towernode', 'x_d_towernode'], 
            promotes_outputs=['x_d_towerelem'])

        self.add_subsystem('tower_elem_2_deriv', 
            TowerElem2Deriv(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['z_towernode', 'Z_tower', 'x_d_towernode'], 
            promotes_outputs=['x_dd_towerelem'])

        self.add_subsystem('modeshape_num',
            ModeshapeNum(mode=mode_num,nNode=nNode,nElem=nElem,nDOF=nDOF),
            promotes_inputs=['x_towernode', 'x_d_towernode', 'x_towerelem', 'x_d_towerelem', 'x_dd_towerelem'],
            promotes_outputs=['x_towernode_%d' % mode_num, 'x_d_towernode_%d' % mode_num, 'x_towerelem_%d' % mode_num, 'x_d_towerelem_%d' % mode_num, 'x_dd_towerelem_%d' % mode_num])