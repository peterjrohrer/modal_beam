import numpy as np

from openmdao.api import ExplicitComponent


class ModeshapeGlobMass(ExplicitComponent):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']

        self.add_input('mel', val=np.zeros((nElem, 4, 4)), units='kg')
    
        self.add_output('M_mode', val=np.zeros((nDOF, nDOF)), units='kg')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']

        mel = inputs['mel']
       
        M_mode_all = np.zeros((2*nNode,2*nNode))

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
                            M_mode_all[row][col] += mel[i][j][p]
        
        # Drop first two DOF
        outputs['M_mode'] = M_mode_all[2:,2:]

    def compute_partials(self, inputs, partials):
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']
        partials['M_mode', 'mel'] = np.zeros(((nDOF * nDOF), (nElem*16)))
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
                            partials['M_mode', 'mel'][(nDOF * row + col)][16 * i + 4 * j + p] += 1.
