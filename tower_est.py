import numpy as np
import myconstants as myconst

from openmdao.api import ExplicitComponent

class TowerEst(ExplicitComponent):
    # A component to provide basic information about DTU 10MW turbine for development
        # Combining turbine and tower at this point

    def setup(self):
        self.add_output('M_turb', val=0., units='kg')
        self.add_output('CoG_turb', val=0., units='m')
        self.add_output('I_turb', val=0., units='kg*m**2')    

    def compute(self, inputs, outputs):
 
        # weight estimates from DTU 10MW reference in JMH thesis
        mass_tower = (628.4)*1000.
        mass_rotor = (230.7)*1000.
        mass_nacelle = (446.0)*1000.

        M_turb = mass_tower + mass_rotor + mass_nacelle

        # CoG Estimation, Roughly calculated by hand
        CoG_turb = 90.35

        # Mass moment of inertia estimation, calculation for tower, point mass for rotor/nacelle
        D_tower = 10. # rough approximation
        L_tower = 109. 
        I_tower = (1/12.)*mass_tower*(3*(D_tower/2.)**2 + L_tower**2) + mass_tower*((L_tower/2.)+10.)**2
        I_RNA = (mass_rotor+mass_nacelle)*(L_tower+10.)**2

        I_turb = I_tower + I_RNA
        
        outputs['M_turb'] = M_turb
        outputs['CoG_turb'] = CoG_turb
        outputs['I_turb'] = I_turb
