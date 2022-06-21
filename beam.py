import numpy as np
import myconstants as myconst
from openmdao.api import ExplicitComponent


class Beam(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        self.add_input('diameter', val=1., units='m')
        self.add_input('thickness', val=1., units='m')

        self.add_output('L_beam', val=np.zeros(nElem), units='m')
        self.add_output('Z_beam', val=np.zeros(nNode), units='m')
        self.add_output('D_beam', val=np.zeros(nElem), units='m')
        self.add_output('wt_beam', val=np.zeros(nElem), units='m')
        self.add_output('M_beam', val=np.zeros(nElem), units='kg')
        self.add_output('tot_M_beam', val=0., units='kg')

    def setup_partials(self):
        self.declare_partials('D_beam', 'diameter')
        self.declare_partials('wt_beam', 'thickness')
        self.declare_partials('M_beam', ['diameter', 'thickness'])
        self.declare_partials('tot_M_beam', ['diameter', 'thickness'])

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        
        D = inputs['diameter']
        wt = inputs['thickness']

        L_overall = 75.
        L_per_elem = L_overall/nElem 
        L_beam = np.ones(nElem)*L_per_elem

        Z_beam = np.concatenate(([0.],np.cumsum(L_beam)))
        
        D_beam = D * np.ones(nElem)
        wt_beam = wt * np.ones(nElem)
        
        M_per_elem = L_per_elem * myconst.RHO_STL * np.pi * 0.25 * ((D*D) - ((D - 2.*wt)*(D - 2.*wt)))
        M_beam = M_per_elem * np.ones(nElem)

        outputs['L_beam'] = L_beam
        outputs['Z_beam'] = Z_beam
        outputs['D_beam'] = D_beam
        outputs['wt_beam'] = wt_beam
        outputs['M_beam'] = M_beam
        outputs['tot_M_beam'] = np.sum(M_beam)

    def compute_partials(self, inputs, partials):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        
        D = inputs['diameter']
        wt = inputs['thickness']
        
        L_overall = 75.
        L_per_elem = L_overall/nElem 
        
        dM_dD = L_per_elem * myconst.RHO_STL * np.pi * 0.25 * ((2.*D) - (2.*(D - 2.*wt)))
        dM_dt = L_per_elem * myconst.RHO_STL * np.pi * 0.25 * -1. * ((8.*wt)-(4.*D))
        
        partials['D_beam', 'diameter'] = np.ones(nElem)
        partials['wt_beam', 'thickness'] = np.ones(nElem)

        partials['M_beam', 'diameter'] = dM_dD * np.ones(nElem)
        partials['M_beam', 'thickness'] = dM_dt * np.ones(nElem)

        partials['tot_M_beam', 'diameter'] = dM_dD * nElem
        partials['tot_M_beam', 'thickness'] = dM_dt * nElem