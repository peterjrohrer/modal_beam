import numpy as np
import myconstants as myconst

from openmdao.api import ExplicitComponent


class ModeshapeElemNormforce(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        
        self.add_input('M_tower', val=np.zeros(nElem), units='kg')
        self.add_input('tot_M_tower', val=0., units='kg')

        self.add_output('normforce_mode_elem', val=np.zeros(nElem), units='N')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        nElem = self.options['nElem']
                
        M_tower = inputs['M_tower']
        tot_M_tower = inputs['tot_M_tower']


        # for i in range(N_towerelem):
        #     outputs['normforce_mode_elem'][i] = (np.sum(M_tower[:i]) - tot_M_tower) * myconst.G
        
        # # Simply Supported
        # outputs['normforce_mode_elem'][0] = (tot_M_tower/2.) * myconst.G
        # outputs['normforce_mode_elem'][-1] = (tot_M_tower/2.) * myconst.G

        # Cantilevered
        outputs['normforce_mode_elem'] = np.zeros(nElem)


    ##TODO Fix these partials!!
    def compute_partials(self, inputs, partials):

        M_tower = inputs['M_tower']
        tot_M_tower = inputs['tot_M_tower']

        N_towerelem = len(M_tower)

        # partials['normforce_mode_elem', 'M_tower'] = np.zeros((22, 10))
        partials['normforce_mode_elem', 'tot_M_tower'] = np.zeros(22)

        for i in range(N_towerelem):
            partials['normforce_mode_elem', 'tot_M_tower'][i] = -1. * myconst.G

            for j in range(i):
                partials['normforce_mode_elem', 'M_tower'][i, j] += myconst.G
