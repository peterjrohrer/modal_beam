import numpy as np
import scipy.interpolate as si
from openmdao.api import ExplicitComponent

class ModalMass(ExplicitComponent):
    # Calculate modal mass for TLPWT in first three modes
    
    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        self.add_input('M_beam', val=np.zeros(nElem), units='kg')
        self.add_input('x_beamelem_1', val=np.zeros(nElem), units='m/m')
        self.add_input('x_beamelem_2', val=np.zeros(nElem), units='m/m')
        self.add_input('x_beamelem_3', val=np.zeros(nElem), units='m/m')

        self.add_output('M11', val=0., units='kg')
        self.add_output('M12', val=0., units='kg')
        self.add_output('M13', val=0., units='kg')
        self.add_output('M22', val=0., units='kg')
        self.add_output('M23', val=0., units='kg')
        self.add_output('M33', val=0., units='kg')

    def	setup_partials(self):
        self.declare_partials('M11', ['x_beamelem_1', 'M_beam'])
        self.declare_partials('M12', ['x_beamelem_1', 'x_beamelem_2', 'M_beam'])
        self.declare_partials('M13', ['x_beamelem_1', 'x_beamelem_3', 'M_beam'])
        self.declare_partials('M22', ['x_beamelem_2', 'M_beam'])
        self.declare_partials('M23', ['x_beamelem_2', 'x_beamelem_3', 'M_beam'])
        self.declare_partials('M33', ['x_beamelem_3', 'M_beam'])


    def compute(self, inputs, outputs):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        M_beam = inputs['M_beam']
        
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
        
        # x_d_beamtop_1 = beam_cubicspline_1(z_beamelem[-1], 1)
        # x_d_beamtop_2 = beam_cubicspline_2(z_beamelem[-1], 1)
        # x_d_beamtop_3 = beam_cubicspline_3(z_beamelem[-1], 1)
        # x_d_pontcen_1 = spar_cubicspline_1(z_sparelem[0], 1)
        # x_d_pontcen_2 = spar_cubicspline_2(z_sparelem[0], 1)
        # x_d_pontcen_3 = spar_cubicspline_3(z_sparelem[0], 1)
        
        # Beam Contributions
        for i in range(nElem):
            outputs['M11'] += M_beam[i] * x_beamelem_1[i] * x_beamelem_1[i]
            outputs['M12'] += M_beam[i] * x_beamelem_1[i] * x_beamelem_2[i]
            outputs['M13'] += M_beam[i] * x_beamelem_1[i] * x_beamelem_3[i]
            outputs['M22'] += M_beam[i] * x_beamelem_2[i] * x_beamelem_2[i]
            outputs['M23'] += M_beam[i] * x_beamelem_2[i] * x_beamelem_3[i]
            outputs['M33'] += M_beam[i] * x_beamelem_3[i] * x_beamelem_3[i]
    
    ##TODO Redo Z_beam partials
    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        M_beam = inputs['M_beam']
        
        x_beamelem_1 = inputs['x_beamelem_1']
        x_beamelem_2 = inputs['x_beamelem_2']
        x_beamelem_3 = inputs['x_beamelem_3']

        partials['M11', 'x_beamelem_1'] = np.zeros((1, nElem))
        partials['M11', 'M_beam'] = np.zeros((1, nElem))

        partials['M12', 'x_beamelem_1'] = np.zeros((1, nElem))
        partials['M12', 'x_beamelem_2'] = np.zeros((1, nElem))
        partials['M12', 'M_beam'] = np.zeros((1, nElem))

        partials['M13', 'x_beamelem_1'] = np.zeros((1, nElem))
        partials['M13', 'x_beamelem_3'] = np.zeros((1, nElem))
        partials['M13', 'M_beam'] = np.zeros((1, nElem))

        partials['M22', 'x_beamelem_2'] = np.zeros((1, nElem))
        partials['M22', 'M_beam'] = np.zeros((1, nElem))

        partials['M23', 'x_beamelem_2'] = np.zeros((1, nElem))
        partials['M23', 'x_beamelem_3'] = np.zeros((1, nElem))
        partials['M23', 'M_beam'] = np.zeros((1, nElem))

        partials['M33', 'x_beamelem_3'] = np.zeros((1, nElem))
        partials['M33', 'M_beam'] = np.zeros((1, nElem))
        
        # Beam Contributions
        for i in range(nElem):
            partials['M11', 'M_beam'][0, i] += x_beamelem_1[i] * x_beamelem_1[i]
            partials['M12', 'M_beam'][0, i] += x_beamelem_1[i] * x_beamelem_2[i]
            partials['M13', 'M_beam'][0, i] += x_beamelem_1[i] * x_beamelem_3[i]
            partials['M22', 'M_beam'][0, i] += x_beamelem_2[i] * x_beamelem_2[i]
            partials['M23', 'M_beam'][0, i] += x_beamelem_2[i] * x_beamelem_3[i]
            partials['M33', 'M_beam'][0, i] += x_beamelem_3[i] * x_beamelem_3[i]

            partials['M11', 'x_beamelem_1'][0, i] += M_beam[i] * 2. * x_beamelem_1[i]
            partials['M12', 'x_beamelem_1'][0, i] += M_beam[i] * x_beamelem_2[i]
            partials['M12', 'x_beamelem_2'][0, i] += M_beam[i] * x_beamelem_1[i]
            partials['M13', 'x_beamelem_1'][0, i] += M_beam[i] * x_beamelem_3[i]
            partials['M13', 'x_beamelem_3'][0, i] += M_beam[i] * x_beamelem_1[i]
            partials['M22', 'x_beamelem_2'][0, i] += M_beam[i] * 2. * x_beamelem_2[i]
            partials['M23', 'x_beamelem_2'][0, i] += M_beam[i] * x_beamelem_3[i]
            partials['M23', 'x_beamelem_3'][0, i] += M_beam[i] * x_beamelem_2[i]
            partials['M33', 'x_beamelem_3'][0, i] += M_beam[i] * 2. * x_beamelem_3[i]