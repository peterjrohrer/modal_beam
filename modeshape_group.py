import numpy as np
import openmdao.api as om

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
from modeshape_eigvector import ModeshapeEigvector
from modeshape_eig_full import ModeshapeEigen

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

        self.add_subsystem('modeshape_beam_nodes',
            ModeshapeBeamNodes(nNode=nNode,nElem=nElem), 
            promotes_inputs=['Z_beam'], 
            promotes_outputs=['z_beamnode', 'z_beamelem'])

        self.add_subsystem('modeshape_elem_length', 
            ModeshapeElemLength(nNode=nNode,nElem=nElem), 
            promotes_inputs=['z_beamnode'], 
            promotes_outputs=['L_mode_elem'])

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

        self.add_subsystem('modeshape_eig_full',
            ModeshapeEigen(nNode=nNode,nElem=nElem,nDOF=nDOF),
            promotes_inputs=['M_mode', 'K_mode'],
            promotes_outputs=['full_eig_vector', 'full_eig_val'])

        self.add_subsystem('modeshape_M_inv', 
            ModeshapeMInv(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['M_mode'], 
            promotes_outputs=['M_mode_inv'])

        self.add_subsystem('modeshape_eigmatrix', 
            ModeshapeEigmatrix(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['K_mode', 'M_mode_inv'], 
            promotes_outputs=['A_eig'])

        self.add_subsystem('modeshape_eigvector', 
            ModeshapeEigvector(nNode=nNode,nElem=nElem,nDOF=nDOF), 
            promotes_inputs=['A_eig'], 
            promotes_outputs=['eig_vector_1', 'eig_freq_1', 'eig_vector_2', 'eig_freq_2', 'eig_vector_3', 'eig_freq_3'])

        numModes = 3
        for i in range(1,numModes+1):
            self.add_subsystem('modeshape_%d' % i,
                Eig2Mode(mode=i,nNode=nNode,nElem=nElem,nDOF=nDOF),
                promotes_inputs=['eig_vector_1', 'eig_vector_2', 'eig_vector_3', 'z_beamnode', 'z_beamelem'],
                promotes_outputs=['x_beamnode_%d' % i, 'x_d_beamnode_%d' % i, 'x_beamelem_%d' % i, 'x_d_beamelem_%d' % i, 'x_dd_beamelem_%d' % i])
        
        self.add_subsystem('modal_mass', 
            ModalMass(nNode=nNode,nElem=nElem), 
            promotes_inputs=['M_beam', 'x_beamelem_*'], 
            promotes_outputs=['M11', 'M12', 'M13', 'M22', 'M23', 'M33'])

        self.add_subsystem('modal_stiffness', 
            ModalStiffness(nNode=nNode,nElem=nElem), 
            promotes_inputs=['L_beam', 'x_d_beamelem_*', 'x_dd_beamelem_*', 'normforce_mode_elem', 'EI_mode_elem'], 
            promotes_outputs=['K11', 'K12', 'K13', 'K22', 'K23', 'K33']) 