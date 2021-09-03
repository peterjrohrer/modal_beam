import numpy as np
import myconstants as myconst
from openmdao.api import ExplicitComponent


class ModeshapePontMass(ExplicitComponent):

    def initialize(self):
        self.options.declare('pont_data', types=dict)

    def setup(self):

        pont_data = self.options['pont_data']
        self.shape_pont = pont_data['shape_pont']
        self.N_pont = pont_data['N_pont']
        self.N_pontelem = pont_data['N_pontelem']
        self.theta = pont_data['theta']

        self.add_input('D_pont', val=0., units='m')
        self.add_input('wt_pont', val=0., units='m')
        self.add_input('pontelem', val=np.zeros(self.N_pontelem), units='m')
        self.add_input('M_pont', val=0., units='kg')

        self.add_output('mode_M_pont', val=0., units='kg')
        self.add_output('mode_I_pont', val=0., units='kg*m**2')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        N_pont = self.N_pont
        theta = self.theta

        D_pont = inputs['D_pont']
        wt_pont = inputs['wt_pont']
        elem = inputs['pontelem']
        M_pont = inputs['M_pont']

        outputs['mode_M_pont'] = M_pont
        
        M_pont_each = M_pont/N_pont
        L_pont = (elem[-1]-elem[0]) + (elem[1]-elem[0])
        inertia = 0.
        ##TODO Implement better inertia estimation
        for i in range(N_pont):
            L_pont_reduced = np.cos(theta[i]) * L_pont
            inertia += (1/12.) * M_pont_each * (D_pont**2 + 4 * L_pont_reduced**2)
        
        outputs['mode_I_pont'] = inertia
        

    def compute_partials(self, inputs, partials):
        
        N_pont = self.N_pont
        theta = self.theta

        D_pont = inputs['D_pont']
        wt_pont = inputs['wt_pont']
        elem = inputs['pontelem']
        M_pont = inputs['M_pont']

        partials['mode_M_pont', 'D_pont'] = 0.
        partials['mode_M_pont', 'wt_pont'] = 0.
        partials['mode_M_pont', 'pontelem'] = np.zeros(self.N_pontelem)
        partials['mode_M_pont', 'M_pont'] = 1.

        partials['mode_I_pont', 'D_pont'] = 0.
        partials['mode_I_pont', 'wt_pont'] = 0.
        partials['mode_I_pont', 'pontelem'] = np.zeros(self.N_pontelem)
        partials['mode_I_pont', 'M_pont'] = 0.

        M_pont_each = M_pont/N_pont
        L_pont = (elem[-1]-elem[0]) + (elem[1]-elem[0])
        dL_delem = np.zeros(self.N_pontelem)
        dL_delem[0] += -2.
        dL_delem[1] += 1.
        dL_delem[-1] += 1.

        inertia = 0.
        for i in range(N_pont):
            L_pont_reduced = np.cos(theta[i]) * L_pont
            inertia += (1/12.) * M_pont_each * (D_pont**2 + 4 * L_pont_reduced**2)

            partials['mode_I_pont', 'D_pont'] += (1/12.) * M_pont_each * (2*D_pont)
            partials['mode_I_pont', 'wt_pont'] += 0.
            partials['mode_I_pont', 'pontelem'] += (1/12.) * M_pont_each * 8 * (L_pont_reduced * dL_delem * np.cos(theta[i]))
            partials['mode_I_pont', 'M_pont'] += (1/12.) * (1/N_pont) * (D_pont**2 + 4 * L_pont_reduced**2)

