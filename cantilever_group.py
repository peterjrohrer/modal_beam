import numpy as np
import openmdao.api as om

from beam import Beam
from modeshape_group import Modeshape

from global_mass import GlobalMass
from global_stiffness import GlobalStiffness


class Cantilever(om.Group):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']        

        self.add_subsystem('beam',
            Beam(nNode=nNode,nElem=nElem),
            promotes_inputs=['D_beam'],
            promotes_outputs=['Z_beam', 'L_beam', 'M_beam', 'tot_M_beam'])

        modeshape_group = Modeshape(nNode=nNode, nElem=nElem, nDOF=nDOF)
        # modeshape_group.linear_solver = om.ScipyKrylov()
        # modeshape_group.linear_solver = om.DirectSolver(assemble_jac=True)
        # modeshape_group.linear_solver.precon = DirectSolver(assemble_jac=True)
        # modeshape_group.nonlinear_solver = om.NonlinearBlockGS(maxiter=500, iprint=0)
        # modeshape_group.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        self.add_subsystem('modeshape_group',
            modeshape_group,
            promotes_inputs=['Z_beam', 'D_beam', 'L_beam', 'M_beam', 'tot_M_beam'],
            promotes_outputs=['eig_vector_*', 'eig_freq_*', 'z_beamnode', 'z_beamelem',
                'x_beamnode_*', 'x_d_beamnode_*', 
                'x_beamelem_*', 'x_d_beamelem_*', 'x_dd_beamelem_*',
                'M11', 'M12', 'M13', 'M22', 'M23', 'M33', 
                'K11', 'K12', 'K13', 'K22', 'K23', 'K33',])

        self.add_subsystem('global_mass',
            GlobalMass(),
            promotes_inputs=['M11', 'M12', 'M13', 'M22', 'M23', 'M33'],
            promotes_outputs=['M_global'])

        self.add_subsystem('global_stiffness',
            GlobalStiffness(),
            promotes_inputs=['K11', 'K12', 'K13', 'K22', 'K23', 'K33'],
            promotes_outputs=['K_global'])

        
