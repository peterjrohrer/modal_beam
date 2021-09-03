import numpy as np
from openmdao.api import ExplicitComponent


class TotalMass(ExplicitComponent):
    # Sum masses without converting to matrix form

    def setup(self):
        self.nNode = 3

        self.add_input('M_sub', val=0., units='kg')
        self.add_input('CoG_sub', val=0., units='m')
        self.add_input('I_sub', val=0., units='kg*m**2')
        self.add_input('M_turb', val=0., units='kg')
        self.add_input('CoG_turb', val=0., units='m')
        self.add_input('I_turb', val=0., units='kg*m**2')

        self.add_output('M_tot', val=0., units='kg')
        self.add_output('CoG_tot', val=0., units='m')
        self.add_output('I_tot', val=0., units='kg*m**2')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        M_sub = inputs['M_sub'][0]
        CoG_sub = inputs['CoG_sub'][0]
        I_sub = inputs['I_sub'][0]
        M_turb = inputs['M_turb'][0]
        CoG_turb = inputs['CoG_turb'][0]
        I_turb = inputs['I_turb'][0]

        M_tot = (M_sub + M_turb)
        CoG_tot = ((M_sub * CoG_sub) + (M_turb * CoG_turb))/M_tot
        I_tot = (I_sub + I_turb)

        outputs['M_tot'] = M_tot
        outputs['CoG_tot'] = CoG_tot
        outputs['I_tot'] = I_tot

    def compute_partials(self, inputs, partials):
        M_sub = inputs['M_sub'][0]
        CoG_sub = inputs['CoG_sub'][0]
        I_sub = inputs['I_sub'][0]
        M_turb = inputs['M_turb'][0]
        CoG_turb = inputs['CoG_turb'][0]
        I_turb = inputs['I_turb'][0]
        addedmass = inputs['A_global']

        partials['M_tot','M_sub'] = 1.
        partials['M_tot','CoG_sub'] = 0.
        partials['M_tot','I_sub'] = 0.
        partials['M_tot','M_turb'] = 1.
        partials['M_tot','CoG_turb'] = 0. 
        partials['M_tot','I_turb'] = 0.

        partials['CoG_tot','M_sub'] = CoG_sub/(M_sub+M_turb) - ((M_sub * CoG_sub) + (M_turb * CoG_turb))/(M_sub + M_turb)**2
        partials['CoG_tot','CoG_sub'] = M_sub/(M_sub+M_turb)
        partials['CoG_tot','I_sub'] = 0.
        partials['CoG_tot','M_turb'] = CoG_turb/(M_sub+M_turb) - ((M_sub * CoG_sub) + (M_turb * CoG_turb))/(M_sub + M_turb)**2
        partials['CoG_tot','CoG_turb'] = M_turb/(M_sub+M_turb)
        partials['CoG_tot','I_turb'] = 0.
        
        partials['I_tot','M_sub'] = 0.
        partials['I_tot','CoG_sub'] = 0.
        partials['I_tot','I_sub'] = 1.
        partials['I_tot','M_turb'] = 0.
        partials['I_tot','CoG_turb'] = 0. 
        partials['I_tot','I_turb'] = 1.




        
        
