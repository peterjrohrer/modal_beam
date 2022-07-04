import numpy as np
from openmdao.api import ExplicitComponent


class ModeshapeEigSelect(ExplicitComponent):
    # Select lowest three modeshape eigenvalues (eigen freq) and eigenvectors for bending

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

        self.declare_partials('eig_vector_1', 'eig_vectors')
        self.declare_partials('eig_freq_1', 'eig_vals')
        self.declare_partials('eig_vector_2', 'eig_vectors')
        self.declare_partials('eig_freq_2', 'eig_vals')
        self.declare_partials('eig_vector_3', 'eig_vectors')
        self.declare_partials('eig_freq_3', 'eig_vals')

    def compute(self, inputs, outputs):
        nDOF = self.options['nDOF']
         
        eig_vecs = inputs['eig_vectors']
        eig_vals = np.diag(inputs['eig_vals'])

        # # "Conventional" Eigenproblem definition
        # outputs['eig_freq_1'] = np.sqrt(eig_vals[-1]) / (2*np.pi) # Export in Hz
        # outputs['eig_vector_1'] = eig_vecs[:, -1]
        # outputs['eig_freq_2'] = np.sqrt(eig_vals[-2]) / (2*np.pi) # Export in Hz
        # outputs['eig_vector_2'] = eig_vecs[:, -2]
        # outputs['eig_freq_3'] = np.sqrt(eig_vals[-3]) / (2*np.pi) # Export in Hz
        # outputs['eig_vector_3'] = eig_vecs[:, -3]

        # He (2022) Eigenproblem definition
        outputs['eig_freq_1'] = np.sqrt(1./eig_vals[0]) / (2*np.pi) # Export in Hz
        outputs['eig_vector_1'] = eig_vecs[:, 0]
        outputs['eig_freq_2'] = np.sqrt(1./eig_vals[1]) / (2*np.pi) # Export in Hz
        outputs['eig_vector_2'] = eig_vecs[:, 1]
        outputs['eig_freq_3'] = np.sqrt(1./eig_vals[2]) / (2*np.pi) # Export in Hz
        outputs['eig_vector_3'] = eig_vecs[:, 2]

    def compute_partials(self, inputs, partials):
        nDOF = self.options['nDOF']
        eig_vecs = inputs['eig_vectors']
        eig_vals = np.diag(inputs['eig_vals'])

        # # "Conventional" Eigenproblem definition
        # dfreq1_dvals = np.zeros((nDOF,nDOF))
        # dfreq2_dvals = np.zeros((nDOF,nDOF))  
        # dfreq3_dvals = np.zeros((nDOF,nDOF))

        # dfreq1_dvals[-1,-1] += 1. / (2.*(2.*np.pi)*np.sqrt(eig_vals[-1]))
        # dfreq2_dvals[-2,-2] += 1. / (2.*(2.*np.pi)*np.sqrt(eig_vals[-2]))
        # dfreq3_dvals[-3,-3] += 1. / (2.*(2.*np.pi)*np.sqrt(eig_vals[-3]))
        
        # partials['eig_freq_1', 'eig_vals'] = dfreq1_dvals
        # partials['eig_freq_2', 'eig_vals'] = dfreq2_dvals
        # partials['eig_freq_3', 'eig_vals'] = dfreq3_dvals

        # dvec1_dvecs = np.zeros((1,nDOF))
        # dvec1_dvecs[0,-1] += 1.
        # dvec2_dvecs = np.zeros((1,nDOF))
        # dvec2_dvecs[0,-2] += 1.
        # dvec3_dvecs = np.zeros((1,nDOF))
        # dvec3_dvecs[0,-3] += 1.

        # partials['eig_vector_1', 'eig_vectors'] = np.kron(np.eye(nDOF),dvec1_dvecs)
        # partials['eig_vector_2', 'eig_vectors'] = np.kron(np.eye(nDOF),dvec2_dvecs)
        # partials['eig_vector_3', 'eig_vectors'] = np.kron(np.eye(nDOF),dvec3_dvecs)

        # He (2022) Eigenproblem definition
        dfreq1_dvals = np.zeros((nDOF,nDOF))
        dfreq2_dvals = np.zeros((nDOF,nDOF))  
        dfreq3_dvals = np.zeros((nDOF,nDOF))

        dfreq1_dvals[0,0] += -1. * (1./eig_vals[0])**(3./2.) / (2.*(2.*np.pi))
        dfreq2_dvals[1,1] += -1. * (1./eig_vals[1])**(3./2.) / (2.*(2.*np.pi))
        dfreq3_dvals[2,2] += -1. * (1./eig_vals[2])**(3./2.) / (2.*(2.*np.pi))
        
        partials['eig_freq_1', 'eig_vals'] = dfreq1_dvals
        partials['eig_freq_2', 'eig_vals'] = dfreq2_dvals
        partials['eig_freq_3', 'eig_vals'] = dfreq3_dvals

        dvec1_dvecs = np.zeros((1,nDOF))
        dvec1_dvecs[0,0] += 1.
        dvec2_dvecs = np.zeros((1,nDOF))
        dvec2_dvecs[0,1] += 1.
        dvec3_dvecs = np.zeros((1,nDOF))
        dvec3_dvecs[0,2] += 1.

        partials['eig_vector_1', 'eig_vectors'] = np.kron(np.eye(nDOF),dvec1_dvecs)
        partials['eig_vector_2', 'eig_vectors'] = np.kron(np.eye(nDOF),dvec2_dvecs)
        partials['eig_vector_3', 'eig_vectors'] = np.kron(np.eye(nDOF),dvec3_dvecs)