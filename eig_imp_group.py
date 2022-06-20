import numpy as np
import openmdao.api as om

from modeshape_eig_imp import ModeshapeEigenImp
from modeshape_eig_select import ModeshapeEigSelect

class EigenImp(om.Group):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']        
        nDOF = self.options['nDOF']        
        
        eigenproblem = ModeshapeEigenImp(nNode=nNode,nElem=nElem,nDOF=nDOF)

        # eigenproblem.linear_solver = om.LinearBlockGS()
        # eigenproblem.linear_solver.precon = om.LinearBlockGS()
        # nlbgs = eigenproblem.nonlinear_solver = om.NonlinearBlockGS()
        # nlbgs.options['maxiter'] = 100
        # nlbgs.options['iprint'] = 0
        eigenproblem.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)
        eigenproblem.linear_solver = om.DirectSolver()

        self.add_subsystem('modeshape_eig_imp',
            eigenproblem,
            promotes_inputs=['M_mode', 'K_mode'],
            promotes_outputs=['eig_vectors', 'eig_vals'])
        
        self.add_subsystem('modeshape_eig_select', 
            ModeshapeEigSelect(nNode=nNode,nElem=nElem,nDOF=nDOF),
            promotes_inputs=['eig_vectors', 'eig_vals'], 
            promotes_outputs=['eig_vector_1', 'eig_freq_1', 'eig_vector_2', 'eig_freq_2', 'eig_vector_3', 'eig_freq_3'])