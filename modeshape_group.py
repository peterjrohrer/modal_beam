import numpy as np
import scipy
import openmdao.api as om

from modeshape_elem_mass import ModeshapeElemMass
from modeshape_elem_mat_stiff import ModeshapeElemMatStiff
from modeshape_elem_geom_stiff import ModeshapeElemGeomStiff
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
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        nodal_data = self.options['nodal_data']  

        self.add_subsystem('modeshape_elem_mass', 
            ModeshapeElemMass(nodal_data=nodal_data), 
            promotes_inputs=['L_beam', 'A_beam', 'Ix_beam', 'M_beam', 'dir_cosines'], 
            promotes_outputs=['mel'])

        self.add_subsystem('modeshape_elem_mat_stiff', 
            ModeshapeElemMatStiff(nodal_data=nodal_data), 
            promotes_inputs=['D_beam', 'wt_beam'], 
            promotes_outputs=['kel_mat'])

        self.add_subsystem('modeshape_elem_geom_stiff', 
            ModeshapeElemGeomStiff(nodal_data=nodal_data), 
            promotes_inputs=['M_beam', 'tot_M_beam'], 
            promotes_outputs=['kel_geom'])
        
        self.add_subsystem('modeshape_elem_stiff', 
            ModeshapeElemStiff(nodal_data=nodal_data), 
            promotes_inputs=['kel_mat', 'kel_geom'], 
            promotes_outputs=['kel'])

        self.add_subsystem('modeshape_glob_mass', 
            ModeshapeGlobMass(nodal_data=nodal_data), 
            promotes_inputs=['mel'], 
            promotes_outputs=['M_mode'])

        self.add_subsystem('modeshape_glob_stiff', 
            ModeshapeGlobStiff(nodal_data=nodal_data), 
            promotes_inputs=['kel'], 
            promotes_outputs=['K_mode'])

        self.add_subsystem('eigenproblem',
            Eigenproblem(),
            promotes_inputs=['M_mode', 'K_mode'],
            promotes_outputs=['eig_vectors', 'eig_vals'])

        self.add_subsystem('modeshape_eig_select', 
            ModeshapeEigSelect(nodal_data=nodal_data),
            promotes_inputs=['eig_vectors', 'eig_vals'], 
            promotes_outputs=['eig_vector_1', 'eig_freq_1', 'eig_vector_2', 'eig_freq_2', 'eig_vector_3', 'eig_freq_3'])

        numModes = 3
        for i in range(1,numModes+1):
            self.add_subsystem('modeshape_%d' % i,
                Eig2Mode(mode=i,nodal_data=nodal_data),
                promotes_inputs=['eig_vector_1', 'eig_vector_2', 'eig_vector_3', 'z_beamnode', 'z_beamelem'],
                promotes_outputs=['x_beamnode_%d' % i, 'x_d_beamnode_%d' % i, 'x_beamelem_%d' % i, 'x_d_beamelem_%d' % i, 'x_dd_beamelem_%d' % i])
        
        self.add_subsystem('modal_mass', 
            ModalMass(nodal_data=nodal_data), 
            promotes_inputs=['M_beam', 'x_beamelem_*'], 
            promotes_outputs=['M11', 'M12', 'M13', 'M22', 'M23', 'M33'])

        self.add_subsystem('modal_stiffness', 
            ModalStiffness(nodal_data=nodal_data), 
            promotes_inputs=['L_beam', 'x_d_beamelem_*', 'x_dd_beamelem_*', 'normforce_mode_elem', 'EI_mode_elem'], 
            promotes_outputs=['K11', 'K12', 'K13', 'K22', 'K23', 'K33']) 