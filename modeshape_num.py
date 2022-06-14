import numpy as np
import matplotlib.pyplot as plt
from openmdao.api import ExplicitComponent

class ModeshapeNum(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('mode', types=int)
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        mode_num = self.options['mode']
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        self.add_input('x_beamnode', val=np.zeros(nNode), units='m/m')
        self.add_input('x_d_beamnode', val=np.zeros(nNode), units='1/m')
        self.add_input('x_beamelem', val=np.zeros(nElem), units='m/m')     
        self.add_input('x_d_beamelem', val=np.zeros(nElem), units='1/m')
        self.add_input('x_dd_beamelem', val=np.zeros(nElem), units='1/(m**2)')

        self.add_output('x_beamnode_%d' % mode_num, val=np.zeros(nNode), units='m/m')
        self.add_output('x_d_beamnode_%d' % mode_num, val=np.zeros(nNode), units='1/m')
        self.add_output('x_beamelem_%d' % mode_num, val=np.zeros(nElem), units='m/m')
        self.add_output('x_d_beamelem_%d' % mode_num, val=np.zeros(nElem), units='1/m')
        self.add_output('x_dd_beamelem_%d' % mode_num, val=np.zeros(nElem), units='1/(m**2)')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        mode_num = self.options['mode']
        
        outputs['x_beamnode_%d' % mode_num] = inputs['x_beamnode']
        outputs['x_d_beamnode_%d' % mode_num] = inputs['x_d_beamnode']
        outputs['x_beamelem_%d' % mode_num] = inputs['x_beamelem']
        outputs['x_d_beamelem_%d' % mode_num] = inputs['x_d_beamelem']
        outputs['x_dd_beamelem_%d' % mode_num] = inputs['x_dd_beamelem']

    def compute_partials(self, inputs, partials):
        mode_num = self.options['mode']
        nNode = self.options['nNode']
        nElem = self.options['nElem']

        partials['x_beamnode_%d' % mode_num, 'x_beamnode'] = np.identity(nNode)
        partials['x_d_beamnode_%d' % mode_num, 'x_d_beamnode'] = np.identity(nNode)
        partials['x_beamelem_%d' % mode_num, 'x_beamelem'] = np.identity(nElem)
        partials['x_d_beamelem_%d' % mode_num, 'x_d_beamelem'] = np.identity(nElem)
        partials['x_dd_beamelem_%d' % mode_num, 'x_dd_beamelem'] = np.identity(nElem)
