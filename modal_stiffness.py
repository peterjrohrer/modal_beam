
import numpy as np
import scipy.interpolate as si

from openmdao.api import ExplicitComponent


class ModalStiffness(ExplicitComponent):
    # Calculate modal stiffness for column with tendon contribution
    
    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        self.add_input('z_beamnode', val=np.zeros(nNode), units='m/m')
        self.add_input('z_beamelem', val=np.zeros(nElem), units='m/m')
        self.add_input('x_beamelem_1', val=np.zeros(nElem), units='m/m')
        self.add_input('x_beamelem_2', val=np.zeros(nElem), units='m/m')
        self.add_input('x_beamelem_3', val=np.zeros(nElem), units='m/m')
        self.add_input('x_d_beamelem_1', val=np.zeros(nElem), units='1/m')
        self.add_input('x_d_beamelem_2', val=np.zeros(nElem), units='1/m')
        self.add_input('x_d_beamelem_3', val=np.zeros(nElem), units='1/m')
        self.add_input('x_dd_beamelem_1', val=np.zeros(nElem), units='1/(m**2)')
        self.add_input('x_dd_beamelem_2', val=np.zeros(nElem), units='1/(m**2)')
        self.add_input('x_dd_beamelem_3', val=np.zeros(nElem), units='1/(m**2)')

        self.add_input('normforce_mode_elem', val=np.zeros(nElem), units='N')
        self.add_input('EI_mode_elem', val=np.zeros(nElem), units='N*m**2')

        self.add_output('K11', val=0., units='N/m')
        self.add_output('K12', val=0., units='N/m')
        self.add_output('K13', val=0., units='N/m')
        self.add_output('K22', val=0., units='N/m')
        self.add_output('K23', val=0., units='N/m')
        self.add_output('K33', val=0., units='N/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        
        z_beamnode = inputs['z_beamnode']
        z_beamelem = inputs['z_beamelem']
        
        x_beamelem_1 = inputs['x_beamelem_1']
        x_beamelem_2 = inputs['x_beamelem_2']
        x_beamelem_3 = inputs['x_beamelem_3']

        # # --- SUBSTITUTE IN SPLINES
        # spar_cubicspline_1 = si.CubicSpline(z_sparelem, x_sparelem_1, bc_type='not-a-knot')
        # beam_cubicspline_1 = si.CubicSpline(z_beamelem, x_beamelem_1, bc_type='not-a-knot')
        # spar_cubicspline_2 = si.CubicSpline(z_sparelem, x_sparelem_2, bc_type='not-a-knot')
        # beam_cubicspline_2 = si.CubicSpline(z_beamelem, x_beamelem_2, bc_type='not-a-knot')
        # spar_cubicspline_3 = si.CubicSpline(z_sparelem, x_sparelem_3, bc_type='not-a-knot')
        # beam_cubicspline_3 = si.CubicSpline(z_beamelem, x_beamelem_3, bc_type='not-a-knot')
        
        # x_d_sparelem_1 = spar_cubicspline_1(z_sparelem, 1)
        # x_d_sparelem_2 = spar_cubicspline_2(z_sparelem, 1)
        # x_d_sparelem_3 = spar_cubicspline_3(z_sparelem, 1)
        # x_d_beamelem_1 = beam_cubicspline_1(z_beamelem, 1)
        # x_d_beamelem_2 = beam_cubicspline_2(z_beamelem, 1)
        # x_d_beamelem_3 = beam_cubicspline_3(z_beamelem, 1)

        # x_dd_sparelem_1 = spar_cubicspline_1(z_sparelem, 2)
        # x_dd_sparelem_2 = spar_cubicspline_2(z_sparelem, 2)
        # x_dd_sparelem_3 = spar_cubicspline_3(z_sparelem, 2)
        # x_dd_beamelem_1 = beam_cubicspline_1(z_beamelem, 2)
        # x_dd_beamelem_2 = beam_cubicspline_2(z_beamelem, 2)
        # x_dd_beamelem_3 = beam_cubicspline_3(z_beamelem, 2)

        x_d_beamelem_1 = inputs['x_d_beamelem_1']
        x_d_beamelem_2 = inputs['x_d_beamelem_2']
        x_d_beamelem_3 = inputs['x_d_beamelem_3']
        x_dd_beamelem_1 = inputs['x_dd_beamelem_1']
        x_dd_beamelem_2 = inputs['x_dd_beamelem_2']
        x_dd_beamelem_3 = inputs['x_dd_beamelem_3']
        
        norm_force = inputs['normforce_mode_elem']
        EI = inputs['EI_mode_elem']

        outputs['K11'] = 0.
        outputs['K12'] = 0.
        outputs['K13'] = 0.
        outputs['K22'] = 0.
        outputs['K23'] = 0.
        outputs['K33'] = 0.

        for i in range(nElem):
            dz = z_beamnode[i + 1] - z_beamnode[i]

            # EI term
            outputs['K11'] += dz * EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_1[i] 
            outputs['K12'] += dz * EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_2[i] 
            outputs['K13'] += dz * EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_3[i] 
            outputs['K22'] += dz * EI[i] * x_dd_beamelem_2[i] * x_dd_beamelem_2[i] 
            outputs['K23'] += dz * EI[i] * x_dd_beamelem_2[i] * x_dd_beamelem_3[i] 
            outputs['K33'] += dz * EI[i] * x_dd_beamelem_3[i] * x_dd_beamelem_3[i] 

            # Norm Force Term
            outputs['K11'] += dz * norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_1[i] 
            outputs['K12'] += dz * norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_2[i] 
            outputs['K13'] += dz * norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_3[i] 
            outputs['K22'] += dz * norm_force[i] * x_d_beamelem_2[i] * x_d_beamelem_2[i] 
            outputs['K23'] += dz * norm_force[i] * x_d_beamelem_2[i] * x_d_beamelem_3[i] 
            outputs['K33'] += dz * norm_force[i] * x_d_beamelem_3[i] * x_d_beamelem_3[i] 

        a = 1
        
    ##TODO add Z_beam partials
    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        
        z_beamnode = inputs['z_beamnode']
        x_d_beamelem_1 = inputs['x_d_beamelem_1']
        x_d_beamelem_2 = inputs['x_d_beamelem_2']
        x_d_beamelem_3 = inputs['x_d_beamelem_3']
        x_dd_beamelem_1 = inputs['x_dd_beamelem_1']
        x_dd_beamelem_2 = inputs['x_dd_beamelem_2']
        x_dd_beamelem_3 = inputs['x_dd_beamelem_3']
        
        norm_force = inputs['normforce_mode_elem']
        EI = inputs['EI_mode_elem']

        partials['K11', 'z_beamnode'] = np.zeros((1, nNode))
        partials['K11', 'x_d_beamelem_1'] = np.zeros((1, nElem))
        partials['K11', 'x_d_beamelem_2'] = np.zeros((1, nElem))
        partials['K11', 'x_d_beamelem_3'] = np.zeros((1, nElem))
        partials['K11', 'x_dd_beamelem_1'] = np.zeros((1, nElem))
        partials['K11', 'x_dd_beamelem_2'] = np.zeros((1, nElem))
        partials['K11', 'x_dd_beamelem_3'] = np.zeros((1, nElem))
        partials['K11', 'normforce_mode_elem'] = np.zeros((1, 22))
        partials['K11', 'EI_mode_elem'] = np.zeros((1, 22))

        partials['K12', 'z_beamnode'] = np.zeros((1, nNode))
        partials['K12', 'x_d_beamelem_1'] = np.zeros((1, nElem))
        partials['K12', 'x_d_beamelem_2'] = np.zeros((1, nElem))
        partials['K12', 'x_d_beamelem_3'] = np.zeros((1, nElem))
        partials['K12', 'x_dd_beamelem_1'] = np.zeros((1, nElem))
        partials['K12', 'x_dd_beamelem_2'] = np.zeros((1, nElem))
        partials['K12', 'x_dd_beamelem_3'] = np.zeros((1, nElem))
        partials['K12', 'normforce_mode_elem'] = np.zeros((1, 22))
        partials['K12', 'EI_mode_elem'] = np.zeros((1, 22))

        partials['K13', 'z_beamnode'] = np.zeros((1, nNode))
        partials['K13', 'x_d_beamelem_1'] = np.zeros((1, nElem))
        partials['K13', 'x_d_beamelem_2'] = np.zeros((1, nElem))
        partials['K13', 'x_d_beamelem_3'] = np.zeros((1, nElem))
        partials['K13', 'x_dd_beamelem_1'] = np.zeros((1, nElem))
        partials['K13', 'x_dd_beamelem_2'] = np.zeros((1, nElem))
        partials['K13', 'x_dd_beamelem_3'] = np.zeros((1, nElem))
        partials['K13', 'normforce_mode_elem'] = np.zeros((1, 22))
        partials['K13', 'EI_mode_elem'] = np.zeros((1, 22))

        partials['K22', 'z_beamnode'] = np.zeros((1, nNode))
        partials['K22', 'x_d_beamelem_1'] = np.zeros((1, nElem))
        partials['K22', 'x_d_beamelem_2'] = np.zeros((1, nElem))
        partials['K22', 'x_d_beamelem_3'] = np.zeros((1, nElem))
        partials['K22', 'x_dd_beamelem_1'] = np.zeros((1, nElem))
        partials['K22', 'x_dd_beamelem_2'] = np.zeros((1, nElem))
        partials['K22', 'x_dd_beamelem_3'] = np.zeros((1, nElem))
        partials['K22', 'normforce_mode_elem'] = np.zeros((1, 22))
        partials['K22', 'EI_mode_elem'] = np.zeros((1, 22))

        partials['K23', 'z_beamnode'] = np.zeros((1, nNode))
        partials['K23', 'x_d_beamelem_1'] = np.zeros((1, nElem))
        partials['K23', 'x_d_beamelem_2'] = np.zeros((1, nElem))
        partials['K23', 'x_d_beamelem_3'] = np.zeros((1, nElem))
        partials['K23', 'x_dd_beamelem_1'] = np.zeros((1, nElem))
        partials['K23', 'x_dd_beamelem_2'] = np.zeros((1, nElem))
        partials['K23', 'x_dd_beamelem_3'] = np.zeros((1, nElem))
        partials['K23', 'normforce_mode_elem'] = np.zeros((1, 22))
        partials['K23', 'EI_mode_elem'] = np.zeros((1, 22))

        partials['K33', 'z_beamnode'] = np.zeros((1, nNode))
        partials['K33', 'x_d_beamelem_1'] = np.zeros((1, nElem))
        partials['K33', 'x_d_beamelem_2'] = np.zeros((1, nElem))
        partials['K33', 'x_d_beamelem_3'] = np.zeros((1, nElem))
        partials['K33', 'x_dd_beamelem_1'] = np.zeros((1, nElem))
        partials['K33', 'x_dd_beamelem_2'] = np.zeros((1, nElem))
        partials['K33', 'x_dd_beamelem_3'] = np.zeros((1, nElem))
        partials['K33', 'normforce_mode_elem'] = np.zeros((1, 22))
        partials['K33', 'EI_mode_elem'] = np.zeros((1, 22))

        N_beamelem = len(x_d_beamelem_1)
        
        for i in range(N_beamelem):
            dz = z_beamnode[i + 1] - z_beamnode[i]

            # EI term
            partials['K11', 'z_beamnode'][0, i] += -1. * EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_1[i]
            partials['K11', 'z_beamnode'][0, i + 1] += EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_1[i]
            partials['K11', 'x_dd_beamelem_1'][0, i] += dz * EI[i] * 2 * x_dd_beamelem_1[i]
            partials['K11', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_1[i] * x_dd_beamelem_1[i]

            partials['K12', 'z_beamnode'][0, i] += -1. * EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_2[i]
            partials['K12', 'z_beamnode'][0, i + 1] += EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_2[i]
            partials['K12', 'x_dd_beamelem_1'][0, i] += dz * EI[i] * x_dd_beamelem_2[i]
            partials['K12', 'x_dd_beamelem_2'][0, i] += dz * EI[i] * x_dd_beamelem_1[i]
            partials['K12', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_1[i] * x_dd_beamelem_2[i]
            
            partials['K13', 'z_beamnode'][0, i] += -1. * EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_3[i]
            partials['K13', 'z_beamnode'][0, i + 1] += EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_3[i]
            partials['K13', 'x_dd_beamelem_1'][0, i] += dz * EI[i] * x_dd_beamelem_3[i]
            partials['K13', 'x_dd_beamelem_3'][0, i] += dz * EI[i] * x_dd_beamelem_1[i]
            partials['K13', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_1[i] * x_dd_beamelem_3[i]

            partials['K22', 'z_beamnode'][0, i] += -1. * EI[i] * x_dd_beamelem_2[i] * x_dd_beamelem_2[i]
            partials['K22', 'z_beamnode'][0, i + 1] += EI[i] * x_dd_beamelem_2[i] * x_dd_beamelem_2[i]
            partials['K22', 'x_dd_beamelem_2'][0, i] += dz * EI[i] * 2 * x_dd_beamelem_2[i]
            partials['K22', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_2[i] * x_dd_beamelem_2[i]

            partials['K23', 'z_beamnode'][0, i] += -1. * EI[i] * x_dd_beamelem_2[i] * x_dd_beamelem_3[i]
            partials['K23', 'z_beamnode'][0, i + 1] += EI[i] * x_dd_beamelem_2[i] * x_dd_beamelem_3[i]
            partials['K23', 'x_dd_beamelem_2'][0, i] += dz * EI[i] * x_dd_beamelem_3[i]
            partials['K23', 'x_dd_beamelem_3'][0, i] += dz * EI[i] * x_dd_beamelem_2[i]
            partials['K23', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_2[i] * x_dd_beamelem_3[i]

            partials['K33', 'z_beamnode'][0, i] += -1. * EI[i] * x_dd_beamelem_3[i] * x_dd_beamelem_3[i]
            partials['K33', 'z_beamnode'][0, i + 1] += EI[i] * x_dd_beamelem_3[i] * x_dd_beamelem_3[i]
            partials['K33', 'x_dd_beamelem_3'][0, i] += dz * EI[i] * 2 * x_dd_beamelem_3[i]
            partials['K33', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_3[i] * x_dd_beamelem_3[i]

            # Norm Force term
            partials['K11', 'z_beamnode'][0, i] += -1. * norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_1[i]
            partials['K11', 'z_beamnode'][0, i + 1] += norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_1[i]
            partials['K11', 'x_d_beamelem_1'][0, i] += dz * norm_force[i] * 2 * x_d_beamelem_1[i]
            partials['K11', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_1[i] * x_d_beamelem_1[i]

            partials['K12', 'z_beamnode'][0, i] += -1. * norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_2[i]
            partials['K12', 'z_beamnode'][0, i + 1] += norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_2[i]
            partials['K12', 'x_d_beamelem_1'][0, i] += dz * norm_force[i] * x_d_beamelem_2[i]
            partials['K12', 'x_d_beamelem_2'][0, i] += dz * norm_force[i] * x_d_beamelem_1[i]
            partials['K12', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_1[i] * x_d_beamelem_2[i]

            partials['K13', 'z_beamnode'][0, i] += -1. * norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_3[i]
            partials['K13', 'z_beamnode'][0, i + 1] += norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_3[i]
            partials['K13', 'x_d_beamelem_1'][0, i] += dz * norm_force[i] * x_d_beamelem_3[i]
            partials['K13', 'x_d_beamelem_3'][0, i] += dz * norm_force[i] * x_d_beamelem_1[i]
            partials['K13', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_1[i] * x_d_beamelem_3[i]

            partials['K22', 'z_beamnode'][0, i] += -1. * norm_force[i] * x_d_beamelem_2[i] * x_d_beamelem_2[i]
            partials['K22', 'z_beamnode'][0, i + 1] += norm_force[i] * x_d_beamelem_2[i] * x_d_beamelem_2[i]
            partials['K22', 'x_d_beamelem_2'][0, i] += dz * norm_force[i] * 2 * x_d_beamelem_2[i]
            partials['K22', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_2[i] * x_d_beamelem_2[i]

            partials['K23', 'z_beamnode'][0, i] += -1. * norm_force[i] * x_d_beamelem_2[i] * x_d_beamelem_3[i]
            partials['K23', 'z_beamnode'][0, i + 1] += norm_force[i] * x_d_beamelem_2[i] * x_d_beamelem_3[i]
            partials['K23', 'x_d_beamelem_2'][0, i] += dz * norm_force[i] * x_d_beamelem_3[i]
            partials['K23', 'x_d_beamelem_3'][0, i] += dz * norm_force[i] * x_d_beamelem_2[i]
            partials['K23', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_2[i] * x_d_beamelem_3[i]

            partials['K33', 'z_beamnode'][0, i] += -1. * norm_force[i] * x_d_beamelem_3[i] * x_d_beamelem_3[i]
            partials['K33', 'z_beamnode'][0, i + 1] += norm_force[i] * x_d_beamelem_3[i] * x_d_beamelem_3[i]
            partials['K33', 'x_d_beamelem_3'][0, i] += dz * norm_force[i] * 2 * x_d_beamelem_3[i]
            partials['K33', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_3[i] * x_d_beamelem_3[i]