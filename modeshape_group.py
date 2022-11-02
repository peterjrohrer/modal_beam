import numpy as np
import scipy
import openmdao.api as om

from modeshape_elem_mass import ModeshapeElemMass
from modeshape_elem_mat_stiff import ModeshapeElemMatStiff
from modeshape_elem_geom_stiff import ModeshapeElemGeomStiff
from modeshape_elem_stiff import ModeshapeElemStiff
from modeshape_glob_mass import ModeshapeGlobMass
from modeshape_glob_stiff import ModeshapeGlobStiff
from modeshape_dof_reduce import ModeshapeDOFReduce

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
            promotes_inputs=['L_beam', 'A_beam', 'Iy_beam', 'dir_cosines'], 
            promotes_outputs=['kel_mat'])

        self.add_subsystem('modeshape_elem_geom_stiff', 
            ModeshapeElemGeomStiff(nodal_data=nodal_data), 
            promotes_inputs=['L_beam', 'P_beam', 'dir_cosines'], 
            promotes_outputs=['kel_geom'])
        
        self.add_subsystem('modeshape_elem_stiff', 
            ModeshapeElemStiff(nodal_data=nodal_data), 
            promotes_inputs=['kel_mat', 'kel_geom'], 
            promotes_outputs=['kel'])

        self.add_subsystem('modeshape_glob_mass', 
            ModeshapeGlobMass(nodal_data=nodal_data), 
            promotes_inputs=['mel'], 
            promotes_outputs=['M_glob'])

        self.add_subsystem('modeshape_glob_stiff', 
            ModeshapeGlobStiff(nodal_data=nodal_data), 
            promotes_inputs=['kel'], 
            promotes_outputs=['K_glob'])

        self.add_subsystem('modeshape_dof_reduce',
            ModeshapeDOFReduce(nodal_data=nodal_data),
            promotes_inputs=['M_glob','K_glob'], 
            promotes_outputs=['Mr_glob', 'Kr_glob'])

        self.add_subsystem('eigenproblem',
            Eigenproblem(nodal_data=nodal_data),
            promotes_inputs=['Mr_glob', 'Kr_glob'],
            promotes_outputs=['Q', 'eig_freqs'])

        # for m in range(1,nodal_data['nMode']+1):
        #     n = m-1
        #     self.connect
        #     self.add_subsystem('modeshape_%d' % m,
        #         Eig2Mode(mode=m,nodal_data=nodal_data),
        #         promotes_inputs=['eig_vector', 'x_beamnode', 'y_beamnode', 'z_beamnode'],
        #         promotes_outputs=['x_beamnode_%d' % i, 'x_d_beamnode_%d' % i, 'x_beamelem_%d' % i, 'x_d_beamelem_%d' % i, 'x_dd_beamelem_%d' % i])
        
        self.add_subsystem('modal_mass', 
            ModalMass(nodal_data=nodal_data), 
            promotes_inputs=['Q', 'M_glob'], 
            promotes_outputs=['M_modal'])

        self.add_subsystem('modal_stiffness', 
            ModalStiffness(nodal_data=nodal_data), 
            promotes_inputs=['Q', 'K_glob'], 
            promotes_outputs=['K_modal']) 