import numpy as np
import openmdao.api as om

class TangentSVD(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        self.add_input('first_tangent_vec', val=np.zeros((1,3)))

        self.add_output('tangent_u', val=np.zeros((1,1)))
        self.add_output('tangent_s', val=np.zeros((1,3)))
        self.add_output('tangent_v', val=np.zeros((3,3)))

    def compute(self, inputs, outputs):
        
        e1 = inputs['first_tangent_vec']
        u, s1, v = np.linalg.svd(e1)
        s = np.zeros((1,3))
        s[0] = s1

        outputs['tangent_u'] = u
        outputs['tangent_s'] = s
        outputs['tangent_v'] = v

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        e1 = inputs['first_tangent_vec']
        u, s1, v = np.linalg.svd(e1)
        s = np.zeros((1,3))
        s[0,0] = float(s1)
        F = np.zeros((1,3))
        F[0,1] = -1./(s[0,0]**2.)
        F[0,2] = -1./(s[0,0]**2.)

        dP = (u.T @ d_inputs['first_tangent_vec'] @ v)
        dP_1 = dP[0,0]
        dP_2 = np.reshape(dP[0,1:],(1,2))
        dD_1 = np.zeros((1,1))
        dD_2 = (1./s1) * dP_2
        dD_3 = np.zeros((2,2))
        dD = np.block([[dD_1,(-1*dD_2)],[dD_2.T,dD_3]])
        dC = np.zeros((1,1))

        ## Based on Giles (2008), https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf (singular value decomposition)
        ##TODO check these??
        if mode == 'rev':    
            if 'tangent_u' in d_outputs:
                if 'first_tangent_vec' in d_inputs:
                    d_inputs['first_tangent_vec'] += (d_outputs['tangent_u'] @ s @ v.T)
            if 'tangent_s' in d_outputs:
                if 'first_tangent_vec' in d_inputs:
                    s1 = np.zeros((1,3))
                    s1[0] = d_outputs['tangent_s']
                    d_inputs['first_tangent_vec'] += (u @ s1 @ v.T)
            if 'tangent_v' in d_outputs:
                if 'first_tangent_vec' in d_inputs:
                    d_inputs['first_tangent_vec'] += (u @ s @ d_outputs['tangent_v'].T)

        elif mode == 'fwd':
            if 'tangent_u' in d_outputs:
                if 'first_tangent_vec' in d_inputs:
                    d_outputs['tangent_u'] += u @ dC
            if 'tangent_s' in d_outputs:
                if 'first_tangent_vec' in d_inputs:
                    I = np.zeros((1,3))
                    I[0] = 1.
                    d_outputs['tangent_s'] += np.multiply(I, dP)
            if 'tangent_v' in d_outputs:
                if 'first_tangent_vec' in d_inputs:
                    d_outputs['tangent_v'] += v @ dD

