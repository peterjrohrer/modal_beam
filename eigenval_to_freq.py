import numpy as np
from openmdao.api import ExplicitComponent

class Eigen2Freq(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_r = self.nodal_data['nDOF_r']

        self.add_input('eigenvals_sorted', val=np.zeros(nDOF_r))
    
        self.add_output('eigfreqs_all', val=np.zeros(nDOF_r), units='1/s')

    def setup_partials(self):
        nDOF_r = self.nodal_data['nDOF_r']
        
        self.declare_partials('eigfreqs_all', 'eigenvals_sorted', rows=np.arange(nDOF_r), cols=np.arange(nDOF_r))

    def compute(self, inputs, outputs):
        LambdaDiag = inputs['eigenvals_sorted']

        # Export frequencies
        Lambda = np.sqrt(np.real(LambdaDiag))/(2*np.pi) # frequencies [Hz]
        outputs['eigfreqs_all'] = Lambda
    
        # # Throw errors for unexpected eigenvalues
        # if any(np.imag(vals) != 0.) :
        #     raise om.AnalysisError('Imaginary eigenvalues')
        # if any(np.real(vals) < -1.e-03) :
        #     raise om.AnalysisError('Negative eigenvalues')

    def compute_partials(self, inputs, partials):
        nDOF_r = self.nodal_data['nDOF_r']
        LambdaDiag = inputs['eigenvals_sorted']

        partials['eigfreqs_all', 'eigenvals_sorted'] = np.zeros(nDOF_r)
        for i in range(nDOF_r):
            partials['eigfreqs_all', 'eigenvals_sorted'][i] += 1. / ((4.*np.pi)*np.sqrt(np.real(LambdaDiag[i])))
