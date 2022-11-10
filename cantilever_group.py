import numpy as np
import openmdao.api as om

from beam import Beam
from tip_mass import TipMass
from beam_directional_cosines import BeamDirectionalCosines
from fem_group import FEM

class Cantilever(om.Group):

    def initialize(self):
        self.options.declare('nodal_data', types=dict)

    def setup(self):
        nodal_data = self.options['nodal_data']      

        self.add_subsystem('beam',
            Beam(nodal_data=nodal_data),
            promotes_inputs=['D_beam', 'wt_beam', 'L_beam_tot'],
            promotes_outputs=['L_beam', 'A_beam', 'Ix_beam', 'Iy_beam', 'Iz_beam', 'M_beam', 'x_beamnode', 'y_beamnode', 'z_beamnode'])

        self.add_subsystem('tip_mass',
            TipMass(),
            promotes_inputs=['tip_mass', 'ref_to_cog', 'tip_inertia'],
            promotes_outputs=['tip_mass_mat'])
        
        self.add_subsystem('beam_dir_cosines',
            BeamDirectionalCosines(nodal_data=nodal_data),
            promotes_inputs=['x_beamnode', 'y_beamnode', 'z_beamnode'],
            promotes_outputs=['dir_cosines'])

        fem_group = FEM(nodal_data=nodal_data)
        # fem_group.linear_solver = om.ScipyKrylov()
        # fem_group.linear_solver = om.DirectSolver(assemble_jac=True)
        # fem_group.linear_solver.precon = DirectSolver(assemble_jac=True)
        # fem_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=500, iprint=0)
        # fem_group.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        self.add_subsystem('fem_group',
            fem_group,
            promotes_inputs=['L_beam', 'A_beam', 'Ix_beam', 'Iy_beam', 'Iz_beam', 'M_beam', 'tip_mass_mat', 'dir_cosines', 'x_beamnode', 'y_beamnode', 'z_beamnode'],
            promotes_outputs=['Q', 'eig_freqs', 'x_nodes', 'y_nodes', 'z_nodes', 'y_d_nodes', 'z_d_nodes', 'y_dd_nodes', 'z_dd_nodes', 'M_modal', 'K_modal'])      
