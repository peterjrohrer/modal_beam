import numpy as np
import scipy.linalg
from openmdao.api import ExplicitComponent

class EigenSort(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF = self.nodal_data['nDOF_r']

        self.add_input('Q_mass_norm', val=np.zeros((nDOF, nDOF)))
        self.add_input('eigenvals_raw', val=np.zeros((nDOF, nDOF)))
    
        self.add_output('Q_sorted', val=np.zeros((nDOF, nDOF)))
        self.add_output('eigenvals_sorted', val=np.zeros(nDOF))

    def setup_partials(self):
        nDOF = self.nodal_data['nDOF_r']
        
        self.declare_partials('Q_sorted', 'Q_mass_norm')
        self.declare_partials('eigenvals_sorted', 'eigenvals_raw')

    def compute(self, inputs, outputs):
        nDOF = self.nodal_data['nDOF_r']
        Q = inputs['Q_mass_norm']
        D = inputs['eigenvals_raw']

        # Sort and diagonalize
        lambdaDiag = np.diag(D)
        self.sort_idx = I = np.argsort(lambdaDiag)
        Q = Q[:,I]
        lambdaDiag = lambdaDiag[I]

        outputs['Q_sorted'] = Q
        outputs['eigenvals_sorted'] = lambdaDiag

    def compute_partials(self, inputs, partials):
        nDOF = self.nodal_data['nDOF_r']

        vals_part = np.zeros((nDOF, nDOF, nDOF))
        vecs_part = np.eye(nDOF)[self.sort_idx,:]
        vecs_part_blocks = []

        for i in range(nDOF):
            vals_part[i, i, i] += 1.
            vecs_part_blocks.append(vecs_part)

        vals_part = np.reshape(vals_part,((nDOF), (nDOF * nDOF)))
        partials['eigenvals_sorted', 'eigenvals_raw'] = vals_part[self.sort_idx,:]
        partials['Q_sorted', 'Q_mass_norm']= scipy.linalg.block_diag(*vecs_part_blocks)