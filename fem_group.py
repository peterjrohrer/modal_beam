import numpy as np

import openmdao.api as om
import openmdao.func_api as omf

from modeshape_block_rotation import ModeshapeBlockRotation

from modeshape_elem_mass import ModeshapeElemMass
from total_mass import TotalMass
from modeshape_elem_mat_stiff import ModeshapeElemMatStiff
from modeshape_elem_geom_stiff import ModeshapeElemGeomStiff
from modeshape_elem_stiff import ModeshapeElemStiff
from modeshape_elem_txform import ModeshapeElemTransform
from modeshape_glob_mass import ModeshapeGlobMass
from modeshape_glob_stiff import ModeshapeGlobStiff
from modeshape_point_mass import ModeshapePointMass
from modeshape_point_stiff import ModeshapePointStiff
from transform_matrix import TransformMatrix
from modeshape_dof_reduce import ModeshapeDOFReduce

from modeshape_M_inv import ModeshapeMInv
from modeshape_eigmatrix import ModeshapeEigmatrix
from modeshape_eigmatrix_imp import ModeshapeEigmatrixImp
# from eigenproblem import Eigenproblem

from eigenvectors import Eigenvecs
from modeshape_eig_sort import EigenvecsSort
from modeshape_eig_select import EigenvecsSelect
from eigenvectors_modal_mass import EigenvecsModalMass
from eigenvectors_mass_norm import EigenvecsMassNorm
from eigenvalues import Eigenvals
from eigenval_to_freq import Eigen2Freq
from eigenproblem_sort import EigenSort
from eigenvecs_removed import EigenRemoved
from eigenproblem_santize import EigenSantize
from modal_reduction import ModalReduction

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
        
        self.add_subsystem('total_mass', 
            TotalMass(nodal_data=nodal_data), 
            promotes_inputs=['M_beam','tip_mass'], 
            promotes_outputs=['tot_M'])

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

        self.add_subsystem('modeshape_elem_txform',
            ModeshapeElemTransform(nodal_data=nodal_data),
            promotes_inputs=['mel_loc', 'kel_loc'],
            promotes_outputs=['mel', 'kel'])

        self.add_subsystem('modeshape_glob_mass', 
            ModeshapeGlobMass(nodal_data=nodal_data), 
            promotes_inputs=['mel'], 
            promotes_outputs=['M_glob_pre'])

        self.add_subsystem('modeshape_glob_stiff', 
            ModeshapeGlobStiff(nodal_data=nodal_data), 
            promotes_inputs=['kel'], 
            promotes_outputs=['K_glob_pre'])

        self.add_subsystem('modeshape_point_mass',
            ModeshapePointMass(nodal_data=nodal_data), 
            promotes_inputs=['M_glob_pre', 'tip_mass_mat'], 
            promotes_outputs=['M_glob'])

        self.add_subsystem('modeshape_point_stiff',
            ModeshapePointStiff(nodal_data=nodal_data), 
            promotes_inputs=['K_glob_pre'], 
            promotes_outputs=['K_glob'])

        self.add_subsystem('transform_matrix',
            TransformMatrix(nodal_data=nodal_data),
            promotes_inputs=[],
            promotes_outputs=['Tr'])

        self.add_subsystem('modeshape_dof_reduce',
            ModeshapeDOFReduce(nodal_data=nodal_data),
            promotes_inputs=['Tr','M_glob','K_glob'], 
            promotes_outputs=['Mr_glob', 'Kr_glob', 'A_glob'])

        # self.add_subsystem('modeshape_M_inv',
        #     ModeshapeMInv(nodal_data=nodal_data),
        #     promotes_inputs=['Mr_glob'],
        #     promotes_outputs=['Mr_glob_inv'])

        # self.add_subsystem('modeshape_eigmatrix',
        #     ModeshapeEigmatrix(nodal_data=nodal_data),
        #     promotes_inputs=['Mr_glob_inv', 'Kr_glob'],
        #     promotes_outputs=['Ar_eig'])

        self.add_subsystem('modeshape_eigmatrix',
            ModeshapeEigmatrixImp(nodal_data=nodal_data),
            promotes_inputs=['Mr_glob', 'Kr_glob'],
            promotes_outputs=['Ar_eig'])
        
        self.add_subsystem('eigenvectors',
            Eigenvecs(nodal_data=nodal_data),
            promotes_inputs=['Ar_eig'],
            promotes_outputs=['Q_raw', 'sort_idx'])

        self.add_subsystem('eigenvectors_sort',
            EigenvecsSort(nodal_data=nodal_data),
            promotes_inputs=['Q_raw', 'sort_idx'],
            promotes_outputs=['Q_sort'])

        self.add_subsystem('eigenvectors_select',
            EigenvecsSelect(nodal_data=nodal_data),
            promotes_inputs=['Q_sort'],
            promotes_outputs=['Q_basis'])       

        # def func(Ar_eig=np.eye(nodal_data['nDOF_r'])):
        #     _, Q_raw = np.linalg.eig(Ar_eig)
        #     return Q_raw
        
        # f = (omf.wrap(func)
        #         .add_input('Ar_eig', shape=(nodal_data['nDOF_r'], nodal_data['nDOF_r']))
        #         .add_output('Q_raw', shape=(nodal_data['nDOF_r'], nodal_data['nDOF_r']))
        #         .declare_partials(of='Q_raw', wrt='Ar_eig'))
        
        # self.add_subsystem('eigenvectors',
        #     om.ExplicitFuncComp(f, use_jit=True),
        #     promotes_inputs=['Ar_eig'],
        #     promotes_outputs=['Q_raw'])

        self.add_subsystem('eigenvectors_modal_mass',
            EigenvecsModalMass(nodal_data=nodal_data),
            promotes_inputs=['Mr_glob', 'Q_raw'],
            promotes_outputs=['M_mode_eig'])

        self.add_subsystem('eigenvectors_mass_norm',
            EigenvecsMassNorm(nodal_data=nodal_data),
            promotes_inputs=['M_mode_eig', 'Q_raw'],
            promotes_outputs=['Q_mass_norm'])
        
        self.add_subsystem('eigenvalues',
            Eigenvals(nodal_data=nodal_data),
            promotes_inputs=['Kr_glob', 'Q_mass_norm'],
            promotes_outputs=['eigenvals_raw'])

        # self.add_subsystem('eigenproblem',
        #     Eigenproblem(nodal_data=nodal_data),
        #     promotes_inputs=['Ar_eig'],
        #     promotes_outputs=['Q_raw', 'eigfreqs_raw'])

        self.add_subsystem('eigen_sort',
            EigenSort(nodal_data=nodal_data),
            promotes_inputs=['Q_mass_norm', 'eigenvals_raw'],
            promotes_outputs=['Q_sorted', 'eigenvals_sorted'])
        
        self.add_subsystem('eignval_to_freq',
            Eigen2Freq(nodal_data=nodal_data),
            promotes_inputs=['eigenvals_sorted'],
            promotes_outputs=['eigfreqs_all'])

        self.add_subsystem('eigen_removed',
            EigenRemoved(nodal_data=nodal_data),
            promotes_inputs=['Tr', 'Q_sorted'],
            promotes_outputs=['Q_all'])
        
        self.add_subsystem('reduce_modes',
            ModalReduction(nodal_data=nodal_data),
            promotes_inputs=['Q_all', 'eigfreqs_all'],
            promotes_outputs=['Q_unnorm', 'eigfreqs'])

        self.add_subsystem('eigen_sanitize',
            EigenSantize(nodal_data=nodal_data),
            promotes_inputs=['Q_unnorm'],
            promotes_outputs=['Q'])

        self.add_subsystem('modeshapes',
            Eig2Mode(nodal_data=nodal_data),
            promotes_inputs=['Q', 'x_beamnode', 'y_beamnode', 'z_beamnode'],
            promotes_outputs=['x_nodes', 'y_nodes', 'z_nodes',
                              'y_d_nodes', 'z_d_nodes',
                              'y_dd_nodes', 'z_dd_nodes'])
        
        self.add_subsystem('modal_mass', 
            ModalMass(nodal_data=nodal_data), 
            promotes_inputs=['Q', 'M_glob'], 
            promotes_outputs=['M_modal'])

        self.add_subsystem('modal_stiffness', 
            ModalStiffness(nodal_data=nodal_data), 
            promotes_inputs=['Q', 'K_glob'], 
            promotes_outputs=['K_modal']) 