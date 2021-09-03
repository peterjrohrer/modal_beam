import numpy as np
from openmdao.api import Group, DirectSolver, ScipyKrylov

from beam import Beam
from modeshape_group import Modeshape

from global_mass import GlobalMass
from global_stiffness import GlobalStiffness


class Cantilever(Group):

    def initialize(self):
        self.options.declare('nNode', types=int)
        self.options.declare('nElem', types=int)
        self.options.declare('nDOF', types=int)

    def setup(self):
        nNode = self.options['nNode']        
        nElem = self.options['nElem']
        nDOF = self.options['nDOF']        

        self.add_subsystem('beam_group',
            Beam(nNode=nNode,nElem=nElem),
            promotes_inputs=[],
            promotes_outputs=['Z_tower', 'D_tower', 'L_tower', 'M_tower', 'tot_M_tower', 'wt_tower'])

        modeshape_group = Modeshape(nNode=nNode, nElem=nElem, nDOF=nDOF)
        # modeshape_group.linear_solver = ScipyKrylov()
        # # modeshape_group.linear_solver = DirectSolver(assemble_jac=True)
        # modeshape_group.linear_solver.precon = DirectSolver(assemble_jac=True)

        self.add_subsystem('modeshape_group',
            modeshape_group,
            promotes_inputs=['Z_tower', 'D_tower', 'L_tower', 'M_tower', 'tot_M_tower', 'wt_tower'],
            promotes_outputs=['eig_vector_*', 'eig_freq_*', 'z_towernode', 'z_towerelem',
                'x_towernode_*', 'x_d_towernode_*', 
                'x_towerelem_*', 'x_d_towerelem_*', 'x_dd_towerelem_*',
                'M11', 'M12', 'M13', 'M22', 'M23', 'M33', 
                'K11', 'K12', 'K13', 'K22', 'K23', 'K33'])

        self.add_subsystem('global_mass',
            GlobalMass(),
            promotes_inputs=['M11', 'M12', 'M13', 'M22', 'M23', 'M33'],
            promotes_outputs=['M_global'])

        self.add_subsystem('global_stiffness',
            GlobalStiffness(),
            promotes_inputs=['K11', 'K12', 'K13', 'K22', 'K23', 'K33'],
            promotes_outputs=['K_global'])

        
