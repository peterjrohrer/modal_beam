import numpy as np
import myconstants as myconst
import openmdao.api as om


class Beam(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        self.nodal_data = self.options['nodal_data']        
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        self.add_input('D_beam', val=np.zeros(nElem), units='m')
        self.add_input('wt_beam', val=np.zeros(nElem), units='m')
        self.add_input('L_beam_tot', val=0., units='m')
        
        self.add_output('L_beam', val=np.zeros(nElem), units='m')
        self.add_output('A_beam', val=np.zeros(nElem), units='m**2')
        self.add_output('Ix_beam', val=np.zeros(nElem), units='m**4')
        self.add_output('Iy_beam', val=np.zeros(nElem), units='m**4')
        self.add_output('Iz_beam', val=np.zeros(nElem), units='m**4')
        self.add_output('M_beam', val=np.zeros(nElem), units='kg')
        self.add_output('x_beamnode', val=np.zeros(nNode), units='m')
        self.add_output('y_beamnode', val=np.zeros(nNode), units='m')
        self.add_output('z_beamnode', val=np.zeros(nNode), units='m')

    def setup_partials(self):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']

        self.declare_partials('L_beam', ['L_beam_tot'])
        self.declare_partials('A_beam', ['D_beam', 'wt_beam'], rows=np.arange(nElem), cols=np.arange(nElem))
        self.declare_partials('Ix_beam', ['D_beam', 'wt_beam'], rows=np.arange(nElem), cols=np.arange(nElem))
        self.declare_partials('Iy_beam', ['D_beam', 'wt_beam'], rows=np.arange(nElem), cols=np.arange(nElem))
        self.declare_partials('Iz_beam', ['D_beam', 'wt_beam'], rows=np.arange(nElem), cols=np.arange(nElem))
        self.declare_partials('M_beam', ['D_beam', 'wt_beam'], rows=np.arange(nElem), cols=np.arange(nElem))
        self.declare_partials('M_beam', ['L_beam_tot'])
        self.declare_partials('x_beamnode', ['L_beam_tot'])

    def compute(self, inputs, outputs):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']
        
        D_beam = inputs['D_beam']
        wt_beam = inputs['wt_beam']
        L_beam_tot = inputs['L_beam_tot']

        L_per_elem = L_beam_tot/nElem 
        L_beam = np.ones(nElem)*L_per_elem

        x_beamnode = np.concatenate(([0.],np.cumsum(L_beam)))
        y_beamnode = np.zeros_like(x_beamnode)
        z_beamnode = np.zeros_like(x_beamnode)
        
        A_beam = np.zeros_like(D_beam)
        Ix_beam = np.zeros_like(D_beam)
        Iy_beam = np.zeros_like(D_beam)
        Iz_beam = np.zeros_like(D_beam)
        M_beam = np.zeros_like(D_beam)
        for i in range(nElem):                
            A_beam[i] += (np.pi / 4.) * ((D_beam[i]**2.) - ((D_beam[i] - 2.*wt_beam[i])**2.))
            Ix_beam[i] += (np.pi / 32.) * ((D_beam[i]**4.) - ((D_beam[i] - 2.*wt_beam[i])**4.))
            Iy_beam[i] += (np.pi / 64.) * ((D_beam[i]**4.) - ((D_beam[i] - 2.*wt_beam[i])**4.))
            Iz_beam[i] += (np.pi / 64.) * ((D_beam[i]**4.) - ((D_beam[i] - 2.*wt_beam[i])**4.))
            M_beam[i] += L_beam[i] * myconst.RHO_STL * A_beam[i]

        outputs['L_beam'] = L_beam
        outputs['A_beam'] = A_beam
        outputs['Ix_beam'] = Ix_beam
        outputs['Iy_beam'] = Iy_beam
        outputs['Iz_beam'] = Iz_beam
        outputs['M_beam'] = M_beam
        outputs['x_beamnode'] = x_beamnode
        outputs['y_beamnode'] = y_beamnode
        outputs['z_beamnode'] = z_beamnode

    def compute_partials(self, inputs, partials):
        nElem = self.nodal_data['nElem']
        nNode = self.nodal_data['nNode']
        
        D_beam = inputs['D_beam']
        wt_beam = inputs['wt_beam']
        L_beam_tot = inputs['L_beam_tot']

        L_per_elem = L_beam_tot/nElem 
        L_beam = np.ones(nElem)*L_per_elem
        dL_dL = (1./ nElem)
        partials['L_beam', 'L_beam_tot'] = np.ones(nElem) * dL_dL

        x_beamnode = np.concatenate(([0.],np.cumsum(L_beam)))
        y_beamnode = np.zeros_like(x_beamnode)
        z_beamnode = np.zeros_like(x_beamnode)
        
        A_beam = np.zeros_like(D_beam)
        Ix_beam = np.zeros_like(D_beam)
        Iy_beam = np.zeros_like(D_beam)
        Iz_beam = np.zeros_like(D_beam)
        M_beam = np.zeros_like(D_beam)
        
        dA_dD = np.zeros_like(D_beam)
        dA_dt = np.zeros_like(D_beam)
        dIx_dD = np.zeros_like(D_beam)
        dIx_dt = np.zeros_like(D_beam)
        dIy_dD = np.zeros_like(D_beam)
        dIy_dt = np.zeros_like(D_beam)
        dIz_dD = np.zeros_like(D_beam)
        dIz_dt = np.zeros_like(D_beam)
        dM_dD = np.zeros_like(D_beam)
        dM_dt = np.zeros_like(D_beam)
        dM_dL = np.zeros_like(D_beam)

        for i in range(nElem):                
            A_beam[i] += (np.pi / 4.) * ((D_beam[i]**2.) - ((D_beam[i] - 2.*wt_beam[i])**2.))
            dA_dD[i] += (np.pi /4.) * ((2.*D_beam[i]) - (2.*(D_beam[i] - 2.*wt_beam[i])))
            dA_dt[i] += np.pi * (D_beam[i] - (2.*wt_beam[i]))
            
            Ix_beam[i] += (np.pi / 32.) * ((D_beam[i]**4.) - ((D_beam[i] - 2.*wt_beam[i])**4.))
            dIx_dD[i] += (np.pi / 8.) * ((D_beam[i]**3.) - ((D_beam[i] - 2.*wt_beam[i])**3.))
            dIx_dt[i] += (np.pi / 4.) * (D_beam[i] - (2.*wt_beam[i]))**3.
            
            Iy_beam[i] += (np.pi / 64.) * ((D_beam[i]**4.) - ((D_beam[i] - 2.*wt_beam[i])**4.))
            dIy_dD[i] += (np.pi / 16.) * ((D_beam[i]**3.) - ((D_beam[i] - 2.*wt_beam[i])**3.))
            dIy_dt[i] += (np.pi / 8.) * (D_beam[i] - (2.*wt_beam[i]))**3.

            Iz_beam[i] += (np.pi / 64.) * ((D_beam[i]**4.) - ((D_beam[i] - 2.*wt_beam[i])**4.))
            dIz_dD[i] += (np.pi / 16.) * ((D_beam[i]**3.) - ((D_beam[i] - 2.*wt_beam[i])**3.))
            dIz_dt[i] += (np.pi / 8.) * (D_beam[i] - (2.*wt_beam[i]))**3.
            
            M_beam[i] += L_beam[i] * myconst.RHO_STL * A_beam[i]
            dM_dD[i] = L_beam[i] * myconst.RHO_STL * (np.pi /4.) * ((2.*D_beam[i]) - (2.*(D_beam[i] - 2.*wt_beam[i])))
            dM_dt[i] = L_beam[i] * myconst.RHO_STL * np.pi * (D_beam[i] - (2.*wt_beam[i]))
            dM_dL[i] = dL_dL * myconst.RHO_STL * A_beam[i]
        
        partials['A_beam', 'D_beam'] = dA_dD
        partials['A_beam', 'wt_beam'] = dA_dt
        partials['Ix_beam', 'D_beam'] = dIx_dD
        partials['Ix_beam', 'wt_beam'] = dIx_dt
        partials['Iy_beam', 'D_beam'] = dIy_dD
        partials['Iy_beam', 'wt_beam'] = dIy_dt
        partials['Iz_beam', 'D_beam'] = dIz_dD
        partials['Iz_beam', 'wt_beam'] = dIz_dt
        partials['M_beam', 'D_beam'] = dM_dD
        partials['M_beam', 'wt_beam'] = dM_dt
        partials['M_beam', 'L_beam_tot'] = dM_dL

        partials['x_beamnode', 'L_beam_tot'] = np.linspace(0,1.,nNode)
        # partials['tot_M_beam', 'D_beam'] = dM_dD
        # partials['tot_M_beam', 'wt_beam'] = dM_dt