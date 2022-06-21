import numpy as np
import openmdao.api as om

from modeshape_eig_imp1 import ModeshapeEigenImp1
from modeshape_eig_imp2 import ModeshapeEigenImp2

class EigenImp(om.Group):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']        
        nDOF = self.options['nDOF']        
    
        self.add_subsystem('modeshape_eig_imp1',
            ModeshapeEigenImp1(nNode=nNode,nElem=nElem,nDOF=nDOF),
            promotes_inputs=['M_mode'],
            promotes_outputs=['eig_vectors'])
            
        self.add_subsystem('modeshape_eig_imp2',
            ModeshapeEigenImp2(nNode=nNode,nElem=nElem,nDOF=nDOF),
            promotes_inputs=['M_mode', 'K_mode', 'eig_vectors'],
            promotes_outputs=['eig_vals'])
        
        