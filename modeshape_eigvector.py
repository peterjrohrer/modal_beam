import numpy as np
from scipy.linalg import det, eig, solve
import pandas as pd

from openmdao.api import ExplicitComponent


class ModeshapeEigvector(ExplicitComponent):
    # Compute modeshape eigenvalues (eigen freq) and eigenvectors for bending

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']

        self.add_input('A_eig', val=np.zeros((nDOF, nDOF)))

        # Setup to export first three eigenvectors
        self.add_output('eig_vector_1', val=np.ones(nDOF), units='m')
        self.add_output('eig_freq_1', val=0., units='1/s')
        self.add_output('eig_vector_2', val=np.ones(nDOF), units='m')
        self.add_output('eig_freq_2', val=0., units='1/s')
        self.add_output('eig_vector_3', val=np.ones(nDOF), units='m')
        self.add_output('eig_freq_3', val=0., units='1/s')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        nNode = self.options['nNode']
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']
        A = inputs['A_eig']

        eig_vals_usrt, eig_vecs_usrt = np.linalg.eig(A)
        idx = eig_vals_usrt.argsort()   
        eig_vals = eig_vals_usrt[idx]
        eig_vecs = eig_vecs_usrt[:,idx]

        if any(eig_vals<0.):
            raise Exception("Negative Eigenvalues!")

        # --- DEBUGGING EIGs ---
        # Export eigenvals/vecs
        cols = []
        for i in range(1,len(eig_vals)+1) :
            cols.append('Mode %1d' %i)
        # Create dataframe with vectors followed by freqs
        df = pd.DataFrame(eig_vecs,columns=cols,index=np.arange(nDOF))
        df.loc['freq'] = np.sqrt(np.real(eig_vals)) / (2*np.pi) # Export in Hz
        # Save dataframe
        # df.to_csv('eig_plots/eig_data.csv')
        # print(eig_vals[-3:])
        # print(2. * np.pi / np.sqrt(eig_vals[-3:]))

        outputs['eig_freq_1'] = np.sqrt(np.real(eig_vals[0])) / (2*np.pi) # Export in Hz
        outputs['eig_vector_1'] = eig_vecs[:, 0]
        outputs['eig_freq_2'] = np.sqrt(np.real(eig_vals[1])) / (2*np.pi) # Export in Hz
        outputs['eig_vector_2'] = eig_vecs[:, 1]
        outputs['eig_freq_3'] = np.sqrt(np.real(eig_vals[2])) / (2*np.pi) # Export in Hz
        outputs['eig_vector_3'] = eig_vecs[:, 2]
        
    ##TODO Better understand these partials, think they are okay despite check
    def compute_partials(self, inputs, partials):
        A = inputs['A_eig']

        partials['eig_vector_1', 'A_eig'] = np.zeros((len(A),A.size))
        partials['eig_freq_1', 'A_eig'] = np.zeros((1,A.size))
        partials['eig_vector_2', 'A_eig'] = np.zeros((len(A),A.size))
        partials['eig_freq_2', 'A_eig'] = np.zeros((1,A.size))
        partials['eig_vector_3', 'A_eig'] = np.zeros((len(A),A.size))
        partials['eig_freq_3', 'A_eig'] = np.zeros((1,A.size))

        eig_vals_usrt, eig_vecs_usrt = np.linalg.eig(A)
        idx = eig_vals_usrt.argsort()   
        eig_vals = eig_vals_usrt[idx]
        eig_vecs = eig_vecs_usrt[:,idx]

        E = np.zeros_like(A)
        F = np.zeros_like(A)

        for i in range(len(A)):
            for j in range(len(A)):
                E[i, j] = eig_vals[j] - eig_vals[i]
                if i != j:
                    F[i, j] = 1. / (eig_vals[j] - eig_vals[i])

        for i in range(len(A)):
            for j in range(len(A)):
                dA = np.zeros_like(A)
                dA[i,j] = 1.

                P = solve(eig_vecs, np.dot(dA, eig_vecs))

                dD = np.diag(np.identity(len(A)) * P)
                dU = np.dot(eig_vecs, (F * P))

                partials['eig_vector_1', 'A_eig'][:,len(A)*i+j] = dU[:,0]
                partials['eig_vector_2', 'A_eig'][:,len(A)*i+j] = dU[:,1]
                partials['eig_vector_3', 'A_eig'][:,len(A)*i+j] = dU[:,2]
                
                partials['eig_freq_1', 'A_eig'][0,len(A)*i+j] = 0.5 / np.sqrt(np.real(eig_vals[0])) * np.real(dD[0])                
                partials['eig_freq_2', 'A_eig'][0,len(A)*i+j] = 0.5 / np.sqrt(np.real(eig_vals[1])) * np.real(dD[1])
                partials['eig_freq_3', 'A_eig'][0,len(A)*i+j] = 0.5 / np.sqrt(np.real(eig_vals[2])) * np.real(dD[2])