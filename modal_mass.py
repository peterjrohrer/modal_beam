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

        self.add_input('z_beamnode', val=np.zeros(nNode), units='m/m')
        self.add_input('z_beamelem', val=np.zeros(nElem), units='m/m')
        self.add_input('Z_beam', val=np.zeros(nNode), units='m')
        self.add_input('x_beamelem_1', val=np.zeros(nElem), units='m/m')
        self.add_input('x_beamelem_2', val=np.zeros(nElem), units='m/m')
        self.add_input('x_beamelem_3', val=np.zeros(nElem), units='m/m')
        self.add_input('M_beam', val=np.zeros(nElem), units='kg')
        self.add_input('L_beam', val=np.zeros(nElem), units='m')

        self.add_output('M11', val=0., units='kg')
        self.add_output('M12', val=0., units='kg')
        self.add_output('M13', val=0., units='kg')
        self.add_output('M22', val=0., units='kg')
        self.add_output('M23', val=0., units='kg')
        self.add_output('M33', val=0., units='kg')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        M_beam = inputs['M_beam']
        L_beam = inputs['L_beam']
        Z_beam = inputs['Z_beam']

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
        
        # x_d_beamtop_1 = beam_cubicspline_1(z_beamelem[-1], 1)
        # x_d_beamtop_2 = beam_cubicspline_2(z_beamelem[-1], 1)
        # x_d_beamtop_3 = beam_cubicspline_3(z_beamelem[-1], 1)
        # x_d_pontcen_1 = spar_cubicspline_1(z_sparelem[0], 1)
        # x_d_pontcen_2 = spar_cubicspline_2(z_sparelem[0], 1)
        # x_d_pontcen_3 = spar_cubicspline_3(z_sparelem[0], 1)
        
        m_elem_beam = M_beam / L_beam

        N_elem_beam = len(x_beamelem_1)

        # Beam Contributions
        for i in range(N_elem_beam):
            dz = z_beamnode[i + 1] - z_beamnode[i]
            m = m_elem_beam[i]

            outputs['M11'] += dz * m * x_beamelem_1[i] * x_beamelem_1[i]
            outputs['M12'] += dz * m * x_beamelem_1[i] * x_beamelem_2[i]
            outputs['M13'] += dz * m * x_beamelem_1[i] * x_beamelem_3[i]
            outputs['M22'] += dz * m * x_beamelem_2[i] * x_beamelem_2[i]
            outputs['M23'] += dz * m * x_beamelem_2[i] * x_beamelem_3[i]
            outputs['M33'] += dz * m * x_beamelem_3[i] * x_beamelem_3[i]

        a = 1
    
    ##TODO Redo Z_beam partials
    def compute_partials(self, inputs, partials):
        
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        M_beam = inputs['M_beam']
        L_beam = inputs['L_beam']
        Z_beam = inputs['Z_beam']

        z_beamnode = inputs['z_beamnode']

        x_beamelem_1 = inputs['x_beamelem_1']
        x_beamelem_2 = inputs['x_beamelem_2']
        x_beamelem_3 = inputs['x_beamelem_3']

        m_elem_beam = M_beam / L_beam

        partials['M11', 'z_beamnode'] = np.zeros((1, nNode))
        partials['M11', 'x_beamelem_1'] = np.zeros((1, nElem))
        partials['M11', 'x_beamelem_2'] = np.zeros((1, nElem))
        partials['M11', 'x_beamelem_3'] = np.zeros((1, nElem))
        partials['M11', 'x_d_beamtop_1'] = 0.
        partials['M11', 'x_d_beamtop_2'] = 0.
        partials['M11', 'x_d_beamtop_3'] = 0.
        partials['M11', 'M_beam'] = np.zeros((1, nElem))
        partials['M11', 'L_beam'] = np.zeros((1, nElem))

        partials['M12', 'z_beamnode'] = np.zeros((1, nNode))
        partials['M12', 'x_beamelem_1'] = np.zeros((1, nElem))
        partials['M12', 'x_beamelem_2'] = np.zeros((1, nElem))
        partials['M12', 'x_beamelem_3'] = np.zeros((1, nElem))
        partials['M12', 'M_beam'] = np.zeros((1, nElem))
        partials['M12', 'L_beam'] = np.zeros((1, nElem))

        partials['M13', 'z_beamnode'] = np.zeros((1, nNode))
        partials['M13', 'x_beamelem_1'] = np.zeros((1, nElem))
        partials['M13', 'x_beamelem_2'] = np.zeros((1, nElem))
        partials['M13', 'x_beamelem_3'] = np.zeros((1, nElem))
        partials['M13', 'M_beam'] = np.zeros((1, nElem))
        partials['M13', 'L_beam'] = np.zeros((1, nElem))

        partials['M22', 'z_beamnode'] = np.zeros((1, nNode))
        partials['M22', 'x_beamelem_1'] = np.zeros((1, nElem))
        partials['M22', 'x_beamelem_2'] = np.zeros((1, nElem))
        partials['M22', 'x_beamelem_3'] = np.zeros((1, nElem))
        partials['M22', 'M_beam'] = np.zeros((1, nElem))
        partials['M22', 'L_beam'] = np.zeros((1, nElem))

        partials['M23', 'z_beamnode'] = np.zeros((1, nNode))
        partials['M23', 'x_beamelem_1'] = np.zeros((1, nElem))
        partials['M23', 'x_beamelem_2'] = np.zeros((1, nElem))
        partials['M23', 'x_beamelem_3'] = np.zeros((1, nElem))
        partials['M23', 'M_beam'] = np.zeros((1, nElem))
        partials['M23', 'L_beam'] = np.zeros((1, nElem))

        partials['M33', 'z_beamnode'] = np.zeros((1, nNode))
        partials['M33', 'x_beamelem_1'] = np.zeros((1, nElem))
        partials['M33', 'x_beamelem_2'] = np.zeros((1, nElem))
        partials['M33', 'x_beamelem_3'] = np.zeros((1, nElem))
        partials['M33', 'M_beam'] = np.zeros((1, nElem))
        partials['M33', 'L_beam'] = np.zeros((1, nElem))
        
        m = 0.

        N_elem_beam = len(x_beamelem_1)

        # Beam Contributions
        for i in range(N_elem_beam):
            z = (z_beamnode[i] + z_beamnode[i + 1]) / 2
            dz = z_beamnode[i + 1] - z_beamnode[i]
            for j in range(len(Z_beam) - 1):
                if (z < Z_beam[j + 1]) and (z >= Z_beam[j]):
                    m = m_elem_beam[j]
                    partials['M11', 'M_beam'][0, j] += dz * x_beamelem_1[i]**2. * 1. / L_beam[j]
                    partials['M11', 'L_beam'][0, j] += -1. * dz * x_beamelem_1[i]**2. * M_beam[j] / L_beam[j]**2. 
                    partials['M12', 'M_beam'][0, j] += dz * x_beamelem_1[i] * x_beamelem_2[i] * 1. / L_beam[j]
                    partials['M12', 'L_beam'][0, j] += -1. * dz * x_beamelem_1[i] * x_beamelem_2[i] * M_beam[j] / L_beam[j]**2.                    
                    partials['M13', 'M_beam'][0, j] += dz * x_beamelem_1[i] * x_beamelem_3[i] * 1. / L_beam[j]
                    partials['M13', 'L_beam'][0, j] += -1. * dz * x_beamelem_1[i] * x_beamelem_3[i] * M_beam[j] / L_beam[j]**2.                    
                    partials['M22', 'M_beam'][0, j] += dz * x_beamelem_2[i]**2. * 1. / L_beam[j]
                    partials['M22', 'L_beam'][0, j] += -1. * dz * x_beamelem_2[i]**2. * M_beam[j] / L_beam[j]**2.                    
                    partials['M23', 'M_beam'][0, j] += dz * x_beamelem_2[i] * x_beamelem_3[i] * 1. / L_beam[j]
                    partials['M23', 'L_beam'][0, j] += -1. * dz * x_beamelem_2[i] * x_beamelem_3[i] * M_beam[j] / L_beam[j]**2.
                    partials['M33', 'M_beam'][0, j] += dz * x_beamelem_3[i]**2. * 1. / L_beam[j]
                    partials['M33', 'L_beam'][0, j] += -1. * dz * x_beamelem_3[i]**2. * M_beam[j] / L_beam[j]**2.
                    break

            partials['M11', 'z_beamnode'][0, i] += -1. * m * x_beamelem_1[i]**2.
            partials['M11', 'z_beamnode'][0, i + 1] += m * x_beamelem_1[i]**2.
            partials['M11', 'x_beamelem_1'][0, i] += dz * m * 2. * x_beamelem_1[i]
            partials['M12', 'z_beamnode'][0, i] += -1. * m * x_beamelem_1[i] * x_beamelem_2[i]
            partials['M12', 'z_beamnode'][0, i + 1] += m * x_beamelem_1[i] * x_beamelem_2[i]
            partials['M12', 'x_beamelem_1'][0, i] += dz * m * x_beamelem_2[i]
            partials['M12', 'x_beamelem_2'][0, i] += dz * m * x_beamelem_1[i]
            partials['M13', 'z_beamnode'][0, i] += -1. * m * x_beamelem_1[i] * x_beamelem_3[i]
            partials['M13', 'z_beamnode'][0, i + 1] += m * x_beamelem_1[i] * x_beamelem_3[i]
            partials['M13', 'x_beamelem_1'][0, i] += dz * m * x_beamelem_3[i]
            partials['M13', 'x_beamelem_3'][0, i] += dz * m * x_beamelem_1[i]
            partials['M22', 'z_beamnode'][0, i] += -1. * m * x_beamelem_2[i]**2.
            partials['M22', 'z_beamnode'][0, i + 1] += m * x_beamelem_2[i]**2.
            partials['M22', 'x_beamelem_2'][0, i] += dz * m * 2. * x_beamelem_2[i]
            partials['M23', 'z_beamnode'][0, i] += -1. * m * x_beamelem_2[i] * x_beamelem_3[i]
            partials['M23', 'z_beamnode'][0, i + 1] += m * x_beamelem_2[i] * x_beamelem_3[i]
            partials['M23', 'x_beamelem_2'][0, i] += dz * m * x_beamelem_3[i]
            partials['M23', 'x_beamelem_3'][0, i] += dz * m * x_beamelem_2[i]
            partials['M33', 'z_beamnode'][0, i] += -1. * m * x_beamelem_3[i]**2.
            partials['M33', 'z_beamnode'][0, i + 1] += m * x_beamelem_3[i]**2.
            partials['M33', 'x_beamelem_3'][0, i] += dz * m * 2. * x_beamelem_3[i]