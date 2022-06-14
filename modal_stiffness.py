
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

        self.add_input('L_beam', val=np.zeros(nElem), units='m')
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

    def setup_partials(self):
        self.declare_partials('K11', ['x_d_beamelem_1', 'x_dd_beamelem_1', 'L_beam', 'normforce_mode_elem', 'EI_mode_elem',])
        self.declare_partials('K12', ['x_d_beamelem_1', 'x_dd_beamelem_1', 'x_d_beamelem_2', 'x_dd_beamelem_2', 'L_beam', 'normforce_mode_elem', 'EI_mode_elem'])
        self.declare_partials('K13', ['x_d_beamelem_1', 'x_dd_beamelem_1', 'x_d_beamelem_3', 'x_dd_beamelem_3', 'L_beam', 'normforce_mode_elem', 'EI_mode_elem'])
        self.declare_partials('K22', ['x_d_beamelem_2', 'x_dd_beamelem_2', 'L_beam', 'normforce_mode_elem', 'EI_mode_elem',])
        self.declare_partials('K23', ['x_d_beamelem_2', 'x_dd_beamelem_2', 'x_d_beamelem_3', 'x_dd_beamelem_3', 'L_beam', 'normforce_mode_elem', 'EI_mode_elem'])
        self.declare_partials('K33', ['x_d_beamelem_3', 'x_dd_beamelem_3', 'L_beam', 'normforce_mode_elem', 'EI_mode_elem',])

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        
        L_beam = inputs['L_beam']

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
            dz = L_beam[i]

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
        
        L_beam = inputs['L_beam']
        x_d_beamelem_1 = inputs['x_d_beamelem_1']
        x_d_beamelem_2 = inputs['x_d_beamelem_2']
        x_d_beamelem_3 = inputs['x_d_beamelem_3']
        x_dd_beamelem_1 = inputs['x_dd_beamelem_1']
        x_dd_beamelem_2 = inputs['x_dd_beamelem_2']
        x_dd_beamelem_3 = inputs['x_dd_beamelem_3']
        
        norm_force = inputs['normforce_mode_elem']
        EI = inputs['EI_mode_elem']

        partials['K11', 'L_beam'] = np.zeros((1, nElem))
        partials['K11', 'x_d_beamelem_1'] = np.zeros((1, nElem))
        partials['K11', 'x_dd_beamelem_1'] = np.zeros((1, nElem))
        partials['K11', 'normforce_mode_elem'] = np.zeros((1, nElem))
        partials['K11', 'EI_mode_elem'] = np.zeros((1, nElem))

        partials['K12', 'L_beam'] = np.zeros((1, nElem))
        partials['K12', 'x_d_beamelem_1'] = np.zeros((1, nElem))
        partials['K12', 'x_d_beamelem_2'] = np.zeros((1, nElem))
        partials['K12', 'x_dd_beamelem_1'] = np.zeros((1, nElem))
        partials['K12', 'x_dd_beamelem_2'] = np.zeros((1, nElem))
        partials['K12', 'normforce_mode_elem'] = np.zeros((1, nElem))
        partials['K12', 'EI_mode_elem'] = np.zeros((1, nElem))

        partials['K13', 'L_beam'] = np.zeros((1, nElem))
        partials['K13', 'x_d_beamelem_1'] = np.zeros((1, nElem))
        partials['K13', 'x_d_beamelem_3'] = np.zeros((1, nElem))
        partials['K13', 'x_dd_beamelem_1'] = np.zeros((1, nElem))
        partials['K13', 'x_dd_beamelem_3'] = np.zeros((1, nElem))
        partials['K13', 'normforce_mode_elem'] = np.zeros((1, nElem))
        partials['K13', 'EI_mode_elem'] = np.zeros((1, nElem))

        partials['K22', 'L_beam'] = np.zeros((1, nElem))
        partials['K22', 'x_d_beamelem_2'] = np.zeros((1, nElem))
        partials['K22', 'x_dd_beamelem_2'] = np.zeros((1, nElem))
        partials['K22', 'normforce_mode_elem'] = np.zeros((1, nElem))
        partials['K22', 'EI_mode_elem'] = np.zeros((1, nElem))

        partials['K23', 'L_beam'] = np.zeros((1, nElem))
        partials['K23', 'x_d_beamelem_2'] = np.zeros((1, nElem))
        partials['K23', 'x_d_beamelem_3'] = np.zeros((1, nElem))
        partials['K23', 'x_dd_beamelem_2'] = np.zeros((1, nElem))
        partials['K23', 'x_dd_beamelem_3'] = np.zeros((1, nElem))
        partials['K23', 'normforce_mode_elem'] = np.zeros((1, nElem))
        partials['K23', 'EI_mode_elem'] = np.zeros((1, nElem))

        partials['K33', 'L_beam'] = np.zeros((1, nElem))
        partials['K33', 'x_d_beamelem_3'] = np.zeros((1, nElem))
        partials['K33', 'x_dd_beamelem_3'] = np.zeros((1, nElem))
        partials['K33', 'normforce_mode_elem'] = np.zeros((1, nElem))
        partials['K33', 'EI_mode_elem'] = np.zeros((1, nElem))

       
        for i in range(nElem):
            dz = L_beam[i]

            # EI term
            partials['K11', 'L_beam'][0, i] += EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_1[i]
            partials['K11', 'x_dd_beamelem_1'][0, i] += dz * EI[i] * 2 * x_dd_beamelem_1[i]
            partials['K11', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_1[i] * x_dd_beamelem_1[i]

            partials['K12', 'L_beam'][0, i] += EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_2[i]
            partials['K12', 'x_dd_beamelem_1'][0, i] += dz * EI[i] * x_dd_beamelem_2[i]
            partials['K12', 'x_dd_beamelem_2'][0, i] += dz * EI[i] * x_dd_beamelem_1[i]
            partials['K12', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_1[i] * x_dd_beamelem_2[i]
            
            partials['K13', 'L_beam'][0, i] += EI[i] * x_dd_beamelem_1[i] * x_dd_beamelem_3[i]
            partials['K13', 'x_dd_beamelem_1'][0, i] += dz * EI[i] * x_dd_beamelem_3[i]
            partials['K13', 'x_dd_beamelem_3'][0, i] += dz * EI[i] * x_dd_beamelem_1[i]
            partials['K13', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_1[i] * x_dd_beamelem_3[i]

            partials['K22', 'L_beam'][0, i] += EI[i] * x_dd_beamelem_2[i] * x_dd_beamelem_2[i]
            partials['K22', 'x_dd_beamelem_2'][0, i] += dz * EI[i] * 2 * x_dd_beamelem_2[i]
            partials['K22', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_2[i] * x_dd_beamelem_2[i]

            partials['K23', 'L_beam'][0, i] += EI[i] * x_dd_beamelem_2[i] * x_dd_beamelem_3[i]
            partials['K23', 'x_dd_beamelem_2'][0, i] += dz * EI[i] * x_dd_beamelem_3[i]
            partials['K23', 'x_dd_beamelem_3'][0, i] += dz * EI[i] * x_dd_beamelem_2[i]
            partials['K23', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_2[i] * x_dd_beamelem_3[i]

            partials['K33', 'L_beam'][0, i] += EI[i] * x_dd_beamelem_3[i] * x_dd_beamelem_3[i]
            partials['K33', 'x_dd_beamelem_3'][0, i] += dz * EI[i] * 2 * x_dd_beamelem_3[i]
            partials['K33', 'EI_mode_elem'][0, i] += dz * x_dd_beamelem_3[i] * x_dd_beamelem_3[i]

            # Norm Force term
            partials['K11', 'L_beam'][0, i] += norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_1[i]
            partials['K11', 'x_d_beamelem_1'][0, i] += dz * norm_force[i] * 2 * x_d_beamelem_1[i]
            partials['K11', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_1[i] * x_d_beamelem_1[i]

            partials['K12', 'L_beam'][0, i] += norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_2[i]
            partials['K12', 'x_d_beamelem_1'][0, i] += dz * norm_force[i] * x_d_beamelem_2[i]
            partials['K12', 'x_d_beamelem_2'][0, i] += dz * norm_force[i] * x_d_beamelem_1[i]
            partials['K12', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_1[i] * x_d_beamelem_2[i]

            partials['K13', 'L_beam'][0, i] += norm_force[i] * x_d_beamelem_1[i] * x_d_beamelem_3[i]
            partials['K13', 'x_d_beamelem_1'][0, i] += dz * norm_force[i] * x_d_beamelem_3[i]
            partials['K13', 'x_d_beamelem_3'][0, i] += dz * norm_force[i] * x_d_beamelem_1[i]
            partials['K13', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_1[i] * x_d_beamelem_3[i]

            partials['K22', 'L_beam'][0, i] += norm_force[i] * x_d_beamelem_2[i] * x_d_beamelem_2[i]
            partials['K22', 'x_d_beamelem_2'][0, i] += dz * norm_force[i] * 2 * x_d_beamelem_2[i]
            partials['K22', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_2[i] * x_d_beamelem_2[i]

            partials['K23', 'L_beam'][0, i] += norm_force[i] * x_d_beamelem_2[i] * x_d_beamelem_3[i]
            partials['K23', 'x_d_beamelem_2'][0, i] += dz * norm_force[i] * x_d_beamelem_3[i]
            partials['K23', 'x_d_beamelem_3'][0, i] += dz * norm_force[i] * x_d_beamelem_2[i]
            partials['K23', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_2[i] * x_d_beamelem_3[i]

            partials['K33', 'L_beam'][0, i] += norm_force[i] * x_d_beamelem_3[i] * x_d_beamelem_3[i]
            partials['K33', 'x_d_beamelem_3'][0, i] += dz * norm_force[i] * 2 * x_d_beamelem_3[i]
            partials['K33', 'normforce_mode_elem'][0, i] += dz * x_d_beamelem_3[i] * x_d_beamelem_3[i]