import numpy as np
import scipy
import openmdao.api as om
import openmdao.func_api as omf

from modeshape_beam_nodes import ModeshapeBeamNodes
from modeshape_elem_length import ModeshapeElemLength
from modeshape_elem_mass import ModeshapeElemMass
from modeshape_elem_EI import ModeshapeElemEI
from modeshape_elem_normforce import ModeshapeElemNormforce
from modeshape_elem_stiff import ModeshapeElemStiff
from modeshape_glob_mass import ModeshapeGlobMass
from modeshape_glob_stiff import ModeshapeGlobStiff

from modeshape_M_inv import ModeshapeMInv
from modeshape_eigmatrix import ModeshapeEigmatrix
from eigenproblem_nelson import EigenproblemNelson

from eigenproblem import Eigenproblem
from modeshape_eig_select import ModeshapeEigSelect

from eigen_to_mode_group import Eig2Mode

from modal_mass import ModalMass
from modal_stiffness import ModalStiffness

class Modeshape(om.Group):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']        
        nDOF = self.options['nDOF']        

        self.add_subsystem('modeshape_elem_mass', 
            ModeshapeElemMass(nNode=nNode,nElem=nElem), 
            promotes_inputs=['L_beam', 'M_beam', 'L_mode_elem'], 
            promotes_outputs=['mel'])

        self.add_subsystem('modeshape_elem_EI', 
            ModeshapeElemEI(nNode=nNode,nElem=nElem), 
            promotes_inputs=['D_beam', 'wt_beam'], 
            promotes_outputs=['EI_mode_elem'])

        self.add_subsystem('modeshape_elem_normforce', 
            ModeshapeElemNormforce(nNode=nNode,nElem=nElem), 
            promotes_inputs=['M_beam', 'tot_M_beam'], 
            promotes_outputs=['normforce_mode_elem'])
        
        self.add_subsystem('modeshape_elem_stiff', 
            ModeshapeElemStiff(nNode=nNode,nElem=nElem), 
            promotes_inputs=['EI_mode_elem', 'L_mode_elem', 'normforce_mode_elem'], 
            promotes_outputs=['kel'])

        self.add_subsystem('modeshape_glob_mass', 
            ModeshapeGlobMass(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['mel'], 
            promotes_outputs=['M_mode'])

        self.add_subsystem('modeshape_glob_stiff', 
            ModeshapeGlobStiff(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['kel'], 
            promotes_outputs=['K_mode'])