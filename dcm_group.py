import openmdao.api as om

from beam_directional_cosines import BeamDirectionalCosines

from delta_nodes import DeltaNodes
from tangent_vector import TangentVector
from first_tangent_vector import FirstTangentVector
from tangent_svd import TangentSVD
from svd_to_normal import SVD2Normal

class DCMGroup(om.Group):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        nodal_data = self.options['nodal_data']  

        self.add_subsystem('beam_dir_cosines',
            BeamDirectionalCosines(nodal_data=nodal_data),
            promotes_inputs=['x_beamnode', 'y_beamnode', 'z_beamnode'],
            promotes_outputs=['dir_cosines'])

        # self.add_subsystem('delta_nodes',
        #     DeltaNodes(nodal_data=nodal_data),
        #     promotes_inputs=['x_beamnode', 'y_beamnode', 'z_beamnode'],
        #     promotes_outputs=['d_node', 'elem_norm'])

        # self.add_subsystem('tangent_vector',
        #     TangentVector(nodal_data=nodal_data),
        #     promotes_inputs=['d_node', 'elem_norm'],
        #     promotes_outputs=['tangent_vecs'])

        # self.add_subsystem('first_tangent_vector',
        #     FirstTangentVector(nodal_data=nodal_data),
        #     promotes_inputs=['tangent_vecs'],
        #     promotes_outputs=['first_tangent_vec'])

        # self.add_subsystem('tangent_svd',
        #     TangentSVD(nodal_data=nodal_data),
        #     promotes_inputs=['first_tangent_vec'],
        #     promotes_outputs=['tangent_u', 'tangent_s', 'tangent_v'])

        # self.add_subsystem('svd_to_normal',
        #     SVD2Normal(nodal_data=nodal_data),
        #     promotes_inputs=['tangent_v'],
        #     promotes_outputs=['normal_vecs'])


