import numpy as np
from openmdao.api import ExplicitComponent


class GlobalMass(ExplicitComponent):
    # A 3 DOF (surge, pitch, bending) calculation of the M matrix (inertia) without a turbine

    def setup(self):
        self.nNode = 3

        self.add_input('M11', val=0., units='kg')
        self.add_input('M12', val=0., units='kg')
        self.add_input('M13', val=0., units='kg')
        self.add_input('M22', val=0., units='kg')
        self.add_input('M23', val=0., units='kg')
        self.add_input('M33', val=0., units='kg')

        self.add_output('M_global', val=np.zeros((3, 3)), units='kg')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        
        M11 = inputs['M11']
        M12 = inputs['M12']
        M13 = inputs['M13']
        M22 = inputs['M22']
        M23 = inputs['M23']
        M33 = inputs['M33']

        outputs['M_global'] = np.zeros((3, 3))
        
        outputs['M_global'][0, 0] += M11
        outputs['M_global'][0, 1] += M12
        outputs['M_global'][0, 2] += M13
        outputs['M_global'][1, 0] += M12
        outputs['M_global'][1, 1] += M22
        outputs['M_global'][1, 2] += M23
        outputs['M_global'][2, 0] += M13
        outputs['M_global'][2, 1] += M23
        outputs['M_global'][2, 2] += M33  

    def compute_partials(self, inputs, partials):
        
        partials['M_global', 'M11'] = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.])
        partials['M_global', 'M12'] = np.array([0., 1., 0., 1., 0., 0., 0., 0., 0.])
        partials['M_global', 'M13'] = np.array([0., 0., 1., 0., 0., 0., 1., 0., 0.])
        partials['M_global', 'M22'] = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.])
        partials['M_global', 'M23'] = np.array([0., 0., 0., 0., 0., 1., 0., 1., 0.])
        partials['M_global', 'M33'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])





        
        
