import numpy as np
import myconstants as myconst
from openmdao.api import ExplicitComponent


class ModeshapePontAddedMass(ExplicitComponent):

    def initialize(self):
        self.options.declare('pont_data', types=dict)

    def setup(self):

        pont_data = self.options['pont_data']
        self.shape_pont = pont_data['shape_pont']
        self.N_pont = pont_data['N_pont']
        self.N_pontelem = pont_data['N_pontelem']
        self.theta = pont_data['theta']

        self.add_input('D_pont', val=0., units='m')
        self.add_input('pontelem', val=np.zeros(self.N_pontelem), units='m')

        self.add_output('mode_addM_pont', val=0., units='kg')
        self.add_output('mode_addI_pont', val=0., units='kg*m**2')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        
        shape_pont = self.shape_pont
        N_pont = self.N_pont
        theta = self.theta

        D_pont = inputs['D_pont']
        elem = inputs['pontelem']

        if shape_pont == 'rect' :
                a_t = myconst.RHO_SW * (4.754) * (D_pont/2.)**2
        elif shape_pont == 'round' :       
            a_t = myconst.RHO_SW * np.pi * (0.25) * D_pont**2            
        else :
            raise Exception("invalid pontoon shape!")

        # Span of elements plus the difference to account for midpoint to midpoint
        L_pont = (elem[-1]-elem[0]) + (elem[1]-elem[0])

        a11_pont = 0.
        a22_pont = 0.
        for i in range(N_pont):
            a11_pont += a_t * L_pont * (np.cos(theta[i]))**2
            ##TODO this is actually a slight underestimate of length, probably insignificant
            a22_pont += (a_t * ((1./3.)*(elem[-1]**3 - elem[0]**3)) * (np.cos(theta[i]))**2)

        outputs['mode_addM_pont'] = a11_pont      
        outputs['mode_addI_pont'] = a22_pont

    def compute_partials(self, inputs, partials):
        
        shape_pont = self.shape_pont
        N_pont = self.N_pont
        theta = self.theta

        D_pont = inputs['D_pont']
        elem = inputs['pontelem']

        if shape_pont == 'rect' :
            a_t = myconst.RHO_SW * (4.754) * (D_pont/2.)**2
            da_dD = myconst.RHO_SW * (4.754) * (D_pont/2.)
        elif shape_pont == 'round' :       
            a_t = myconst.RHO_SW * np.pi * (0.25) * D_pont**2     
            da_dD = myconst.RHO_SW * np.pi * (0.5) * D_pont
        else :
            raise Exception("invalid pontoon shape!")

        L_pont = (elem[-1]-elem[0]) + (elem[1]-elem[0])
        dL_delem = np.zeros(self.N_pontelem)
        dL_delem[0] += -2.
        dL_delem[1] += 1.
        dL_delem[-1] += 1.

        a11_pont = 0.
        a22_pont = 0.

        partials['mode_addM_pont', 'D_pont'] = 0.
        partials['mode_addM_pont', 'pontelem'] = np.zeros(self.N_pontelem)

        partials['mode_addI_pont', 'D_pont'] = 0.
        partials['mode_addI_pont', 'pontelem'] = np.zeros(self.N_pontelem)

        for i in range(N_pont):
            a11_pont += a_t * L_pont * (np.cos(theta[i]))**2
            partials['mode_addM_pont', 'D_pont'] += da_dD * L_pont * (np.cos(theta[i]))**2
            partials['mode_addM_pont', 'pontelem'] += a_t * dL_delem * (np.cos(theta[i]))**2
            
            a22_pont += (a_t * ((1./3.)*(elem[-1]**3 - elem[0]**3)) * (np.cos(theta[i]))**2)
            partials['mode_addI_pont', 'D_pont'] += (da_dD * ((1./3.)*(elem[-1]**3 - elem[0]**3)) * (np.cos(theta[i]))**2)
            partials['mode_addI_pont', 'pontelem'][0,-1] += a_t * (elem[-1]**2) * (np.cos(theta[i]))**2
            partials['mode_addI_pont', 'pontelem'][0,0] += a_t * -1. * (elem[0]**2) * (np.cos(theta[i]))**2

