import numpy as np
from openmdao.api import ExplicitComponent

class EigenRemoved(ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        nMode = self.nodal_data['nMode']

        self.add_input('Q_sorted', val=np.zeros((nDOF_r, nDOF_r)))
    
        self.add_output('Q_all', val=np.zeros((nDOF_tot, nDOF_r)))

    def setup_partials(self):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        nPart = nDOF_r * nDOF_r
    
        Hrows0 = np.arange(0,(nDOF_r*nDOF_tot))
        mask = np.ones(len(Hrows0), dtype=bool)
        for i in range(nDOF_tot):
            if i in self.nodal_data['IDOF_removed']:
                mask[(i*nDOF_r):((i+1)*nDOF_r)] = False
                a=1
        Hrows = Hrows0[mask,...]
    
        self.declare_partials('Q_all', 'Q_sorted', rows=Hrows, cols=np.arange(nPart), val=np.ones(nPart))

    def compute(self, inputs, outputs):
        nDOF_tot = self.nodal_data['nDOF_tot']
        nDOF_r = self.nodal_data['nDOF_r']
        Tr = self.nodal_data['Tr']
        Q = inputs['Q_sorted']

        # --- Add removed DOF back into eigenvectors
        Qr = Q
        Q = Tr.dot(Qr)

        outputs['Q_all'] = Q