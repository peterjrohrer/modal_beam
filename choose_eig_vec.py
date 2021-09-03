import numpy as np

from openmdao.api import ExplicitComponent

class ChooseEigVec(ExplicitComponent):
    ## Extract single eigenvector from three provided
    
    def initialize(self):
        self.options.declare('mode', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        mode_num = self.options['mode']
        nDOF = self.options['nDOF']        

        self.add_input('eig_vector_1', val=np.ones(nDOF), units='m')
        self.add_input('eig_vector_2', val=np.ones(nDOF), units='m')
        self.add_input('eig_vector_3', val=np.ones(nDOF), units='m')
        
        self.add_output('eig_vector', val=np.ones(nDOF), units='m')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        mode_num = self.options['mode']
        nDOF = self.options['nDOF']        
        
        if mode_num == 1 :
            outputs['eig_vector'] = inputs['eig_vector_1']
        elif mode_num == 2 :
            outputs['eig_vector'] = inputs['eig_vector_2']
        elif mode_num == 3 :
            outputs['eig_vector'] = inputs['eig_vector_3']
        else :
            raise Exception("Invalid Mode!")

    def compute_partials(self, inputs, partials):
        mode_num = self.options['mode']
        nDOF = self.options['nDOF']        

        partials['eig_vector', 'eig_vector_1'] = np.zeros((nDOF,nDOF))
        partials['eig_vector', 'eig_vector_2'] = np.zeros((nDOF,nDOF))
        partials['eig_vector', 'eig_vector_3'] = np.zeros((nDOF,nDOF))
        
        if mode_num == 1 :
            partials['eig_vector', 'eig_vector_1'] = np.eye(nDOF)
        elif mode_num == 2 :
            partials['eig_vector', 'eig_vector_2'] = np.eye(nDOF)
        elif mode_num == 3 :
            partials['eig_vector', 'eig_vector_3'] = np.eye(nDOF)
        else :
            raise Exception("Invalid Mode!")
        
