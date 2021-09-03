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

        self.add_input('x_towernode', val=np.zeros(nNode), units='m/m')
        self.add_input('x_d_towernode', val=np.zeros(nNode), units='1/m')
        self.add_input('x_towerelem', val=np.zeros(nElem), units='m/m')     
        self.add_input('x_d_towerelem', val=np.zeros(nElem), units='1/m')
        self.add_input('x_dd_towerelem', val=np.zeros(nElem), units='1/(m**2)')

        self.add_output('x_towernode_%d' % mode_num, val=np.zeros(nNode), units='m/m')
        self.add_output('x_d_towernode_%d' % mode_num, val=np.zeros(nNode), units='1/m')
        self.add_output('x_towerelem_%d' % mode_num, val=np.zeros(nElem), units='m/m')
        self.add_output('x_d_towerelem_%d' % mode_num, val=np.zeros(nElem), units='1/m')
        self.add_output('x_dd_towerelem_%d' % mode_num, val=np.zeros(nElem), units='1/(m**2)')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        mode_num = self.options['mode']
        
        outputs['x_towernode_%d' % mode_num] = inputs['x_towernode']
        outputs['x_d_towernode_%d' % mode_num] = inputs['x_d_towernode']
        outputs['x_towerelem_%d' % mode_num] = inputs['x_towerelem']
        outputs['x_d_towerelem_%d' % mode_num] = inputs['x_d_towerelem']
        outputs['x_dd_towerelem_%d' % mode_num] = inputs['x_dd_towerelem']

    def compute_partials(self, inputs, partials):
        mode_num = self.options['mode']

        partials['x_towernode_%d' % mode_num, 'x_towernode'] = np.ones(11)
        partials['x_d_towernode_%d' % mode_num, 'x_d_towernode'] = np.ones(11)
        partials['x_towerelem_%d' % mode_num, 'x_towerelem'] = np.ones(10)
        partials['x_d_towerelem_%d' % mode_num, 'x_d_towerelem'] = np.ones(10)
        partials['x_dd_towerelem_%d' % mode_num, 'x_dd_towerelem'] = np.ones(10)
