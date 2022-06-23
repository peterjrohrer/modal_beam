import numpy as np
import myconstants as myconst
import openmdao.api as om


class Beam(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        self.add_input('D_beam', val=np.zeros(nElem), units='m')
        
        self.add_output('L_beam', val=np.zeros(nElem), units='m')
        self.add_output('Z_beam', val=np.zeros(nNode), units='m')
        self.add_output('M_beam', val=np.zeros(nElem), units='kg')
        self.add_output('tot_M_beam', val=0., units='kg')

    def setup_partials(self):
        self.declare_partials('M_beam', ['D_beam'])
        self.declare_partials('tot_M_beam', ['D_beam'])

    def compute(self, inputs, outputs):
        nElem = self.options['nElem']
        
        D_beam = inputs['D_beam']

        L_overall = myconst.L_BEAM
        L_per_elem = L_overall/nElem 
        L_beam = np.ones(nElem)*L_per_elem

        Z_beam = np.concatenate(([0.],np.cumsum(L_beam)))
        
        M_beam = np.zeros_like(D_beam)
        for i in range(nElem):                
            M_beam[i] = L_beam[i] * myconst.RHO_STL * np.pi * 0.25 * (D_beam[i]*D_beam[i]) 

        outputs['L_beam'] = L_beam
        outputs['Z_beam'] = Z_beam
        outputs['M_beam'] = M_beam
        outputs['tot_M_beam'] = np.sum(M_beam)

    def compute_partials(self, inputs, partials):
        nElem = self.options['nElem']
        
        D_beam = inputs['D_beam']

        L_overall = 75.
        L_per_elem = L_overall/nElem 
        L_beam = np.ones(nElem)*L_per_elem
        
        M_beam = np.zeros_like(D_beam)
        dM_dD = np.zeros_like(D_beam)
        dM_dt = np.zeros_like(D_beam)

        for i in range(nElem):                
            M_beam[i] = L_beam[i] * myconst.RHO_STL * np.pi * 0.25 * (D_beam[i]*D_beam[i])
            dM_dD[i] = L_beam[i] * myconst.RHO_STL * np.pi * 0.25 * (2.*D_beam[i])
        
        partials['M_beam', 'D_beam'] = np.diag(dM_dD)

        partials['tot_M_beam', 'D_beam'] = dM_dD
