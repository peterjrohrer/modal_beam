import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeGlobStiff(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']
        
        self.add_input('kel', val=np.zeros((nElem, 4, 4)), units='N/m')
        
        self.add_output('K_mode', val=np.zeros((nDOF, nDOF)), units='N/m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']
        nElem = self.options['nElem']        
        nDOF = self.options['nDOF']

        kel = inputs['kel']
        
        K_mode_all = np.zeros((2*nNode,2*nNode))

        LD = np.zeros((nElem, 4))

        for i in range(nElem):
            for j in range(4):
                LD[i, j] = j + 2 * i

        for i in range(nElem):
            for j in range(4):
                row = int(LD[i][j])
                if row > -1:
                    for p in range(4):
                        col = int(LD[i][p])
                        if col > -1:
                            K_mode_all[row][col] += kel[i][j][p]
        
        # Drop first two DOF
        outputs['K_mode'] = K_mode_all[2:,2:]
        
    def compute_partials(self, inputs, partials):
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']
        partials['K_mode', 'kel'] = np.zeros(((nDOF * nDOF), (nElem*16)))
        LD = np.zeros((nElem, 4))

        for i in range(nElem):
            for j in range(4):
                LD[i, j] = j + 2 * i

        for i in range(nElem-1):
            for j in range(4):
                row = int(LD[i][j])
                if row > -1:
                    for p in range(4):
                        col = int(LD[i][p])
                        if col > -1:
                            partials['K_mode', 'kel'][(nDOF * row + col)][16 * i + 4 * j + p] += 1.