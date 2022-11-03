import numpy as np
from openmdao.api import ExplicitComponent

class ModeshapeElemStiff(ExplicitComponent):
	# Add total stiffness 
	
	def initialize(self):
		self.options.declare('nodal_data', types=dict)

	def setup(self):
		self.nodal_data = self.options['nodal_data']
		nElem = self.nodal_data['nElem']

		self.add_input('kel_mat', val=np.zeros((nElem,12,12)), units='N/m')
		self.add_input('kel_geom', val=np.zeros((nElem,12,12)), units='N/m')

		self.add_output('kel_loc', val=np.zeros((nElem,12,12)), units='N/m')

	def setup_partials(self):
		self.declare_partials('kel_loc', ['kel_mat', 'kel_geom'])

	def compute(self, inputs, outputs):
		nElem = self.nodal_data['nElem']

		kem = inputs['kel_mat']
		keg = inputs['kel_geom']

		outputs['kel_loc'] += kem + keg

	def compute_partials(self, inputs, partials):
		nElem = self.nodal_data['nElem']
		
		partials['kel_loc', 'kel_mat'] = np.zeros(((nElem * 12 * 12), (nElem * 12 * 12)))
		partials['kel_loc', 'kel_geom'] = np.zeros(((nElem * 12 * 12), (nElem * 12 * 12)))
