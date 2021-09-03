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

        self.add_input('z_towernode', val=np.zeros(nNode), units='m/m')
        self.add_input('z_towerelem', val=np.zeros(nElem), units='m/m')
        self.add_input('x_towerelem_1', val=np.zeros(nElem), units='m/m')
        self.add_input('x_towerelem_2', val=np.zeros(nElem), units='m/m')
        self.add_input('x_towerelem_3', val=np.zeros(nElem), units='m/m')
        self.add_input('M_tower', val=np.zeros(nElem), units='kg')
        self.add_input('L_tower', val=np.zeros(nElem), units='m')
        self.add_input('Z_tower', val=np.zeros(nNode), units='m')

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

        M_tower = inputs['M_tower']
        L_tower = inputs['L_tower']
        Z_tower = inputs['Z_tower']

        z_towernode = inputs['z_towernode'] * inputs['Z_tower'][-1] # Add back in dimensionality
        z_towerelem = inputs['z_towerelem']
        
        x_towerelem_1 = inputs['x_towerelem_1']
        x_towerelem_2 = inputs['x_towerelem_2']
        x_towerelem_3 = inputs['x_towerelem_3']

        # # --- SUBSTITUTE IN SPLINES
        # spar_cubicspline_1 = si.CubicSpline(z_sparelem, x_sparelem_1, bc_type='not-a-knot')
        # tower_cubicspline_1 = si.CubicSpline(z_towerelem, x_towerelem_1, bc_type='not-a-knot')
        # spar_cubicspline_2 = si.CubicSpline(z_sparelem, x_sparelem_2, bc_type='not-a-knot')
        # tower_cubicspline_2 = si.CubicSpline(z_towerelem, x_towerelem_2, bc_type='not-a-knot')
        # spar_cubicspline_3 = si.CubicSpline(z_sparelem, x_sparelem_3, bc_type='not-a-knot')
        # tower_cubicspline_3 = si.CubicSpline(z_towerelem, x_towerelem_3, bc_type='not-a-knot')
        
        # x_d_towertop_1 = tower_cubicspline_1(z_towerelem[-1], 1)
        # x_d_towertop_2 = tower_cubicspline_2(z_towerelem[-1], 1)
        # x_d_towertop_3 = tower_cubicspline_3(z_towerelem[-1], 1)
        # x_d_pontcen_1 = spar_cubicspline_1(z_sparelem[0], 1)
        # x_d_pontcen_2 = spar_cubicspline_2(z_sparelem[0], 1)
        # x_d_pontcen_3 = spar_cubicspline_3(z_sparelem[0], 1)
        
        m_elem_tower = M_tower / L_tower

        N_elem_tower = len(x_towerelem_1)

        # Tower Contributions
        for i in range(N_elem_tower):
            dz = z_towernode[i + 1] - z_towernode[i]
            m = m_elem_tower[i]

            outputs['M11'] += dz * m * x_towerelem_1[i] * x_towerelem_1[i]
            outputs['M12'] += dz * m * x_towerelem_1[i] * x_towerelem_2[i]
            outputs['M13'] += dz * m * x_towerelem_1[i] * x_towerelem_3[i]
            outputs['M22'] += dz * m * x_towerelem_2[i] * x_towerelem_2[i]
            outputs['M23'] += dz * m * x_towerelem_2[i] * x_towerelem_3[i]
            outputs['M33'] += dz * m * x_towerelem_3[i] * x_towerelem_3[i]

        a = 1
    
    ##TODO Redo Z_tower partials
    def compute_partials(self, inputs, partials):
        
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        M_tower = inputs['M_tower']
        L_tower = inputs['L_tower']
        Z_tower = inputs['Z_tower']

        z_towernode = inputs['z_towernode'] * inputs['Z_tower'][-1] # Add back in dimensionality

        x_towerelem_1 = inputs['x_towerelem_1']
        x_towerelem_2 = inputs['x_towerelem_2']
        x_towerelem_3 = inputs['x_towerelem_3']

        m_elem_tower = M_tower / L_tower

        partials['M11', 'z_towernode'] = np.zeros((1, nNode))
        partials['M11', 'x_towerelem_1'] = np.zeros((1, nElem))
        partials['M11', 'x_towerelem_2'] = np.zeros((1, nElem))
        partials['M11', 'x_towerelem_3'] = np.zeros((1, nElem))
        partials['M11', 'x_d_towertop_1'] = 0.
        partials['M11', 'x_d_towertop_2'] = 0.
        partials['M11', 'x_d_towertop_3'] = 0.
        partials['M11', 'M_tower'] = np.zeros((1, nElem))
        partials['M11', 'L_tower'] = np.zeros((1, nElem))

        partials['M12', 'z_towernode'] = np.zeros((1, nNode))
        partials['M12', 'x_towerelem_1'] = np.zeros((1, nElem))
        partials['M12', 'x_towerelem_2'] = np.zeros((1, nElem))
        partials['M12', 'x_towerelem_3'] = np.zeros((1, nElem))
        partials['M12', 'M_tower'] = np.zeros((1, nElem))
        partials['M12', 'L_tower'] = np.zeros((1, nElem))

        partials['M13', 'z_towernode'] = np.zeros((1, nNode))
        partials['M13', 'x_towerelem_1'] = np.zeros((1, nElem))
        partials['M13', 'x_towerelem_2'] = np.zeros((1, nElem))
        partials['M13', 'x_towerelem_3'] = np.zeros((1, nElem))
        partials['M13', 'M_tower'] = np.zeros((1, nElem))
        partials['M13', 'L_tower'] = np.zeros((1, nElem))

        partials['M22', 'z_towernode'] = np.zeros((1, nNode))
        partials['M22', 'x_towerelem_1'] = np.zeros((1, nElem))
        partials['M22', 'x_towerelem_2'] = np.zeros((1, nElem))
        partials['M22', 'x_towerelem_3'] = np.zeros((1, nElem))
        partials['M22', 'M_tower'] = np.zeros((1, nElem))
        partials['M22', 'L_tower'] = np.zeros((1, nElem))

        partials['M23', 'z_towernode'] = np.zeros((1, nNode))
        partials['M23', 'x_towerelem_1'] = np.zeros((1, nElem))
        partials['M23', 'x_towerelem_2'] = np.zeros((1, nElem))
        partials['M23', 'x_towerelem_3'] = np.zeros((1, nElem))
        partials['M23', 'M_tower'] = np.zeros((1, nElem))
        partials['M23', 'L_tower'] = np.zeros((1, nElem))

        partials['M33', 'z_towernode'] = np.zeros((1, nNode))
        partials['M33', 'x_towerelem_1'] = np.zeros((1, nElem))
        partials['M33', 'x_towerelem_2'] = np.zeros((1, nElem))
        partials['M33', 'x_towerelem_3'] = np.zeros((1, nElem))
        partials['M33', 'M_tower'] = np.zeros((1, nElem))
        partials['M33', 'L_tower'] = np.zeros((1, nElem))
        
        m = 0.

        N_elem_tower = len(x_towerelem_1)

        # Tower Contributions
        for i in range(N_elem_tower):
            z = (z_towernode[i] + z_towernode[i + 1]) / 2
            dz = z_towernode[i + 1] - z_towernode[i]
            for j in range(len(Z_tower) - 1):
                if (z < Z_tower[j + 1]) and (z >= Z_tower[j]):
                    m = m_elem_tower[j]
                    partials['M11', 'M_tower'][0, j] += dz * x_towerelem_1[i]**2. * 1. / L_tower[j]
                    partials['M11', 'L_tower'][0, j] += -1. * dz * x_towerelem_1[i]**2. * M_tower[j] / L_tower[j]**2. 
                    partials['M12', 'M_tower'][0, j] += dz * x_towerelem_1[i] * x_towerelem_2[i] * 1. / L_tower[j]
                    partials['M12', 'L_tower'][0, j] += -1. * dz * x_towerelem_1[i] * x_towerelem_2[i] * M_tower[j] / L_tower[j]**2.                    
                    partials['M13', 'M_tower'][0, j] += dz * x_towerelem_1[i] * x_towerelem_3[i] * 1. / L_tower[j]
                    partials['M13', 'L_tower'][0, j] += -1. * dz * x_towerelem_1[i] * x_towerelem_3[i] * M_tower[j] / L_tower[j]**2.                    
                    partials['M22', 'M_tower'][0, j] += dz * x_towerelem_2[i]**2. * 1. / L_tower[j]
                    partials['M22', 'L_tower'][0, j] += -1. * dz * x_towerelem_2[i]**2. * M_tower[j] / L_tower[j]**2.                    
                    partials['M23', 'M_tower'][0, j] += dz * x_towerelem_2[i] * x_towerelem_3[i] * 1. / L_tower[j]
                    partials['M23', 'L_tower'][0, j] += -1. * dz * x_towerelem_2[i] * x_towerelem_3[i] * M_tower[j] / L_tower[j]**2.
                    partials['M33', 'M_tower'][0, j] += dz * x_towerelem_3[i]**2. * 1. / L_tower[j]
                    partials['M33', 'L_tower'][0, j] += -1. * dz * x_towerelem_3[i]**2. * M_tower[j] / L_tower[j]**2.
                    break

            partials['M11', 'z_towernode'][0, i] += -1. * m * x_towerelem_1[i]**2.
            partials['M11', 'z_towernode'][0, i + 1] += m * x_towerelem_1[i]**2.
            partials['M11', 'x_towerelem_1'][0, i] += dz * m * 2. * x_towerelem_1[i]
            partials['M12', 'z_towernode'][0, i] += -1. * m * x_towerelem_1[i] * x_towerelem_2[i]
            partials['M12', 'z_towernode'][0, i + 1] += m * x_towerelem_1[i] * x_towerelem_2[i]
            partials['M12', 'x_towerelem_1'][0, i] += dz * m * x_towerelem_2[i]
            partials['M12', 'x_towerelem_2'][0, i] += dz * m * x_towerelem_1[i]
            partials['M13', 'z_towernode'][0, i] += -1. * m * x_towerelem_1[i] * x_towerelem_3[i]
            partials['M13', 'z_towernode'][0, i + 1] += m * x_towerelem_1[i] * x_towerelem_3[i]
            partials['M13', 'x_towerelem_1'][0, i] += dz * m * x_towerelem_3[i]
            partials['M13', 'x_towerelem_3'][0, i] += dz * m * x_towerelem_1[i]
            partials['M22', 'z_towernode'][0, i] += -1. * m * x_towerelem_2[i]**2.
            partials['M22', 'z_towernode'][0, i + 1] += m * x_towerelem_2[i]**2.
            partials['M22', 'x_towerelem_2'][0, i] += dz * m * 2. * x_towerelem_2[i]
            partials['M23', 'z_towernode'][0, i] += -1. * m * x_towerelem_2[i] * x_towerelem_3[i]
            partials['M23', 'z_towernode'][0, i + 1] += m * x_towerelem_2[i] * x_towerelem_3[i]
            partials['M23', 'x_towerelem_2'][0, i] += dz * m * x_towerelem_3[i]
            partials['M23', 'x_towerelem_3'][0, i] += dz * m * x_towerelem_2[i]
            partials['M33', 'z_towernode'][0, i] += -1. * m * x_towerelem_3[i]**2.
            partials['M33', 'z_towernode'][0, i + 1] += m * x_towerelem_3[i]**2.
            partials['M33', 'x_towerelem_3'][0, i] += dz * m * 2. * x_towerelem_3[i]