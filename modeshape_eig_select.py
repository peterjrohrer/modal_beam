import numpy as np
from openmdao.api import ExplicitComponent


class ModeshapeEigSelect(ExplicitComponent):
    # Compute modeshape eigenvalues (eigen freq) and eigenvectors for bending

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nDOF = self.options['nDOF']

        self.add_input('eig_vectors', val=np.zeros((nDOF, nDOF)))
        self.add_input('eig_vals', val=np.zeros((nDOF, nDOF)))

        # Setup to export first three eigenvectors
        self.add_output('eig_vector_1', val=np.ones(nDOF), units='m')
        self.add_output('eig_freq_1', val=0., units='1/s')
        self.add_output('eig_vector_2', val=np.ones(nDOF), units='m')
        self.add_output('eig_freq_2', val=0., units='1/s')
        self.add_output('eig_vector_3', val=np.ones(nDOF), units='m')
        self.add_output('eig_freq_3', val=0., units='1/s')

        self.declare_partials('eig_vector_1', 'eig_vectors', method='fd')
        self.declare_partials('eig_freq_1', 'eig_vals', method='fd')
        self.declare_partials('eig_vector_2', 'eig_vectors', method='fd')
        self.declare_partials('eig_freq_2', 'eig_vals', method='fd')
        self.declare_partials('eig_vector_3', 'eig_vectors', method='fd')
        self.declare_partials('eig_freq_3', 'eig_vals', method='fd')

    def compute(self, inputs, outputs):
        nDOF = self.options['nDOF']
         
        eig_vecs = inputs['eig_vectors']
        eig_vals = np.diag(inputs['eig_vals'])

        outputs['eig_freq_1'] = np.sqrt(np.real(eig_vals[0])) / (2*np.pi) # Export in Hz
        outputs['eig_vector_1'] = eig_vecs[:, 0]
        outputs['eig_freq_2'] = np.sqrt(np.real(eig_vals[1])) / (2*np.pi) # Export in Hz
        outputs['eig_vector_2'] = eig_vecs[:, 1]
        outputs['eig_freq_3'] = np.sqrt(np.real(eig_vals[2])) / (2*np.pi) # Export in Hz
        outputs['eig_vector_3'] = eig_vecs[:, 2]