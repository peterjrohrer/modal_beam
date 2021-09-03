import numpy as np
from openmdao.api import ExplicitComponent 

class ModeshapeSparNodes(ExplicitComponent):

	def setup(self):
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('L_ball', val=0., units='m')

		self.add_output('z_sparnode', val=np.zeros(13), units='m/m')
		self.add_output('z_sparelem', val=np.zeros(12), units='m/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		Z_spar = inputs['Z_spar']
		hub_height = inputs['Z_tower'][-1]
		L_ball = inputs['L_ball'][0]
		z_ball = -1. * inputs['spar_draft'][0] + L_ball #top of ballast
		z_SWL = 0.

		if len(np.where(Z_spar==z_ball)[0]) != 0:
			z_ball += 0.1
		if len(np.where(Z_spar==z_SWL)[0]) != 0:
			z_SWL += 0.1
		if z_ball == z_SWL:
			z_ball += 0.1
		
		z_aux = np.array([z_ball, z_SWL])

		z_node = np.sort(np.concatenate((Z_spar, z_aux),0)) /hub_height # normalized spar node locations
		
		h = np.zeros(12)
		z_elem = np.zeros(len(h))

		for i in range(len(h)):
			h[i] = z_node[i + 1] - z_node[i]
			z_elem[i] = z_node[i]+h[i]

		outputs['z_sparnode'] = z_node
		outputs['z_sparelem'] = z_elem


	def compute_partials(self, inputs, partials):
		Z_spar = inputs['Z_spar']
		Z_tower = inputs['Z_tower']
		hub_height = inputs['Z_tower'][-1]
		L_ball = inputs['L_ball'][0]
		z_ball = -1. * inputs['spar_draft'][0] + L_ball
		z_SWL = 0.

		if len(np.where(Z_spar==z_ball)[0]) != 0:
			z_ball += 0.1
		if len(np.where(Z_spar==z_SWL)[0]) != 0:
			z_SWL += 0.1
		if z_ball == z_SWL:
			z_ball += 0.1

		z_aux = np.array([z_ball, z_SWL])

		z_sparnode = np.concatenate((Z_spar, z_aux),0)
		z_sparnode = np.sort(z_sparnode)/hub_height
		z_ball = z_ball/hub_height
		z_SWL = z_SWL/hub_height

		partials['z_sparnode', 'Z_spar'] = np.zeros((13,11))
		partials['z_sparnode', 'Z_tower'] = np.zeros((13,11))
		partials['z_sparnode', 'spar_draft'] = np.zeros(13)
		partials['z_sparnode', 'L_ball'] = np.zeros(13)

		ballidx = np.concatenate(np.where(z_sparnode==z_ball))
		SWLidx = np.concatenate(np.where(z_sparnode==z_SWL))

		partials['z_sparnode', 'spar_draft'][ballidx] = -1./hub_height
		partials['z_sparnode', 'L_ball'][ballidx] = 1./hub_height

		count = 0
		for i in range(13):
			partials['z_sparnode', 'Z_tower'][i, -1] += -1. * z_sparnode[i]/hub_height
			if i != ballidx and i != SWLidx:
				partials['z_sparnode', 'Z_spar'][i,count] += 1./hub_height
				count += 1
