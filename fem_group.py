import numpy as np
import scipy
import openmdao.api as om
from modeshape_block_rotation import ModeshapeBlockRotation

from modeshape_elem_mass import ModeshapeElemMass
from modeshape_elem_mat_stiff import ModeshapeElemMatStiff
from modeshape_elem_geom_stiff import ModeshapeElemGeomStiff
from modeshape_elem_stiff import ModeshapeElemStiff
from modeshape_elem_txform import ModeshapeElemTransform
from modeshape_glob_mass import ModeshapeGlobMass
from modeshape_glob_stiff import ModeshapeGlobStiff
from modeshape_dof_reduce import ModeshapeDOFReduce

from modeshape_M_inv import ModeshapeMInv
from modeshape_eigmatrix import ModeshapeEigmatrix
from eigenproblem_nelson import EigenproblemNelson

from eigenproblem import Eigenproblem
from modal_reduction import ModalReduction
from modeshape_eig_select import ModeshapeEigSelect

from eigen_to_mode_group import Eig2Mode

from modal_mass import ModalMass
from modal_stiffness import ModalStiffness

class FEM(om.Group):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        nodal_data = self.options['nodal_data']  

        self.add_subsystem('modeshape_elem_mass', 
            ModeshapeElemMass(nodal_data=nodal_data), 
            promotes_inputs=['L_beam', 'A_beam', 'Ix_beam', 'M_beam'], 
            promotes_outputs=['mel_loc'])

        self.add_subsystem('modeshape_elem_mat_stiff', 
            ModeshapeElemMatStiff(nodal_data=nodal_data), 
            promotes_inputs=['L_beam', 'A_beam', 'Iy_beam', 'Iz_beam'], 
            promotes_outputs=['kel_mat'])

        self.add_subsystem('modeshape_elem_geom_stiff', 
            ModeshapeElemGeomStiff(nodal_data=nodal_data), 
            promotes_inputs=['L_beam', 'P_beam'], 
            promotes_outputs=['kel_geom'])
        
        self.add_subsystem('modeshape_elem_stiff', 
            ModeshapeElemStiff(nodal_data=nodal_data), 
            promotes_inputs=['kel_mat', 'kel_geom'], 
            promotes_outputs=['kel_loc'])
        
        self.add_subsystem('modeshape_block_rotation',
            ModeshapeBlockRotation(nodal_data=nodal_data),
            promotes_inputs=['dir_cosines'],
            promotes_outputs=['block_rot_mat'])

        self.add_subsystem('modeshape_elem_txform',
            ModeshapeElemTransform(nodal_data=nodal_data),
            promotes_inputs=['mel_loc', 'kel_loc', 'block_rot_mat'],
            promotes_outputs=['mel', 'kel'])

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
            promotes_outputs=['Q_full', 'eig_freqs_full'])

        self.add_subsystem('reduce_modes',
            ModalReduction(nodal_data=nodal_data),
            promotes_inputs=['Q_full', 'eig_freqs_full'],
            promotes_outputs=['Q', 'eig_freqs'])

        self.add_subsystem('modeshapes',
            Eig2Mode(nodal_data=nodal_data),
            promotes_inputs=['Q', 'x_beamnode', 'y_beamnode', 'z_beamnode'],
            promotes_outputs=['x_nodes', 'y_nodes', 'z_nodes'])
        
        self.add_subsystem('modal_mass', 
            ModalMass(nodal_data=nodal_data), 
            promotes_inputs=['Q', 'M_glob'], 
            promotes_outputs=['M_modal'])

        self.add_subsystem('modal_stiffness', 
            ModalStiffness(nodal_data=nodal_data), 
            promotes_inputs=['Q', 'K_glob'], 
            promotes_outputs=['K_modal']) 