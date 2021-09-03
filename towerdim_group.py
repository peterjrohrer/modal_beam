import numpy as np

from openmdao.api import Group

from Tower.tower_diameter import TowerDiameter
from Tower.z_tower import ZTower
from Tower.taper_tower import TaperTower


class Towerdim(Group):

    def setup(self):

        self.add_subsystem('tower_diameter',
            TowerDiameter(),
            promotes_inputs=['D_tower_p'],
            promotes_outputs=['D_tower'])

        self.add_subsystem('Z_tower',
            ZTower(),
            promotes_inputs=['L_tower'],
            promotes_outputs=['Z_tower'])

        self.add_subsystem('taper_tower',
            TaperTower(),
            promotes_inputs=['D_tower_p', 'L_tower'],
            promotes_outputs=['taper_angle_tower'])
