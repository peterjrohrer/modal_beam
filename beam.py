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

        # self.add_input('L_spar', val=np.zeros(10), units='m')
        # self.add_input('A_pont', val=0., units='m**2')

        self.add_output('Z_tower', val=np.zeros(nNode), units='m')
        self.add_output('D_tower', val=np.zeros(nElem), units='m')
        self.add_output('L_tower', val=np.zeros(nElem), units='m')
        self.add_output('M_tower', val=np.zeros(nElem), units='kg')
        self.add_output('wt_tower', val=np.zeros(nElem), units='m')
        self.add_output('tot_M_tower', val=0., units='kg')

        # self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']

        L_per_elem = 3.3
        L_tower = np.ones(nElem)*L_per_elem

        Z_tower = np.concatenate(([0.],np.cumsum(L_tower)))
        
        D = .25
        D_tower = D * np.ones(nElem)
        
        wt = 0.01
        wt_tower = wt * np.ones(nElem)
        
        M_per_elem = L_per_elem * myconst.RHO_STL * np.pi * 0.25 * ((D*D) - ((D - 2.*wt)*(D - 2.*wt)))
        M_tower = M_per_elem * np.ones(nElem)

        outputs['Z_tower'] = Z_tower
        outputs['D_tower'] = D_tower
        outputs['L_tower'] = L_tower
        outputs['M_tower'] = M_tower
        outputs['wt_tower'] = wt_tower
        outputs['tot_M_tower'] = np.sum(M_tower)

    # def compute_partials(self, inputs, partials):
       