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
        
        self.add_input('M_beam', val=np.zeros(nElem), units='kg')
        self.add_input('tot_M_beam', val=0., units='kg')

        self.add_output('normforce_mode_elem', val=np.zeros(nElem), units='N')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        nElem = self.options['nElem']
                
        M_beam = inputs['M_beam']
        tot_M_beam = inputs['tot_M_beam']


        # for i in range(N_beamelem):
        #     outputs['normforce_mode_elem'][i] = (np.sum(M_beam[:i]) - tot_M_beam) * myconst.G
        
        # # Simply Supported
        # outputs['normforce_mode_elem'][0] = (tot_M_beam/2.) * myconst.G
        # outputs['normforce_mode_elem'][-1] = (tot_M_beam/2.) * myconst.G

        # Cantilevered
        outputs['normforce_mode_elem'] = np.zeros(nElem)


    ##TODO Fix these partials!!
    def compute_partials(self, inputs, partials):

        M_beam = inputs['M_beam']
        tot_M_beam = inputs['tot_M_beam']

        N_beamelem = len(M_beam)

        # partials['normforce_mode_elem', 'M_beam'] = np.zeros((22, 10))
        partials['normforce_mode_elem', 'tot_M_beam'] = np.zeros(22)

        for i in range(N_beamelem):
            partials['normforce_mode_elem', 'tot_M_beam'][i] = -1. * myconst.G

            for j in range(i):
                partials['normforce_mode_elem', 'M_beam'][i, j] += myconst.G
