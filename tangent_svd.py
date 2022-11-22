import numpy as np
import openmdao.api as om
import tensorflow as tf

class TangentSVD(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        self.add_input('first_tangent_vec', val=np.zeros((1,3)))

        self.add_output('tangent_u', val=np.zeros((1,1)))
        self.add_output('tangent_s', val=np.zeros((1,1)))
        self.add_output('tangent_v', val=np.zeros((3,1)))

        self.declare_partials('*','*')

    def compute(self, inputs, outputs):
        
        e1 = tf.Variable(inputs['first_tangent_vec'])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(e1)
            s, u, v = tf.linalg.svd(e1, full_matrices=False)

        self.du_de1 = tape.gradient(u,e1).numpy()
        self.ds_de1 = tape.gradient(s,e1).numpy()
        self.dv_de1 = tape.gradient(v,e1).numpy()

        del tape

        outputs['tangent_u'] = u.numpy()
        outputs['tangent_s'] = s.numpy()
        outputs['tangent_v'] = v.numpy()

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        
        partials['tangent_u','first_tangent_vec'] = self.du_de1
        partials['tangent_s','first_tangent_vec'] = self.ds_de1
        partials['tangent_v','first_tangent_vec'][0,:] = self.dv_de1

        a=1

    # def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
    #     e1 = inputs['first_tangent_vec']
    #     u, s1, v = np.linalg.svd(e1)
    #     s = np.zeros((1,3))
    #     s[0,0] = float(s1)
    #     F = np.zeros((1,3))
    #     F[0,1] = -1./(s[0,0]**2.)
    #     F[0,2] = -1./(s[0,0]**2.)

    #     dP = (u.T @ d_inputs['first_tangent_vec'] @ v)
    #     dP_1 = dP[0,0]
    #     dP_2 = np.reshape(dP[0,1:],(1,2))
    #     dD_1 = np.zeros((1,1))
    #     dD_2 = (1./s1) * dP_2
    #     dD_3 = np.zeros((2,2))
    #     dD = np.block([[dD_1,(-1*dD_2)],[dD_2.T,dD_3]])
    #     dC = np.zeros((1,1))

    #     ## Based on Giles (2008), https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf (singular value decomposition)
    #     ##TODO check these??
    #     if mode == 'rev':    
    #         if 'tangent_u' in d_outputs:
    #             if 'first_tangent_vec' in d_inputs:
    #                 d_inputs['first_tangent_vec'] += (d_outputs['tangent_u'] @ s @ v.T)
    #         if 'tangent_s' in d_outputs:
    #             if 'first_tangent_vec' in d_inputs:
    #                 s1 = np.zeros((1,3))
    #                 s1[0] = d_outputs['tangent_s']
    #                 d_inputs['first_tangent_vec'] += (u @ s1 @ v.T)
    #         if 'tangent_v' in d_outputs:
    #             if 'first_tangent_vec' in d_inputs:
    #                 d_inputs['first_tangent_vec'] += (u @ s @ d_outputs['tangent_v'].T)

    #     elif mode == 'fwd':
    #         if 'tangent_u' in d_outputs:
    #             if 'first_tangent_vec' in d_inputs:
    #                 d_outputs['tangent_u'] += u @ dC
    #         if 'tangent_s' in d_outputs:
    #             if 'first_tangent_vec' in d_inputs:
    #                 I = np.zeros((1,3))
    #                 I[0] = 1.
    #                 d_outputs['tangent_s'] += np.multiply(I, dP)
    #         if 'tangent_v' in d_outputs:
    #             if 'first_tangent_vec' in d_inputs:
    #                 d_outputs['tangent_v'] += v @ dD

