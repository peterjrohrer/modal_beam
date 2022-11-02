import numpy as np
from openmdao.api import ExplicitComponent

class ModeshapeElemStiff(ExplicitComponent):
	# Add total stiffness 
	
	def initialize(self):
		self.options.declare('nodal_data', types=dict)

	def setup(self):
		nElem = self.nodal_data['nElem']

		self.add_input('kel_mat', val=np.zeros((nElem,12,12)), units='N/m')
		self.add_input('kel_geom', val=np.zeros((nElem,12,12)), units='N/m')

		self.add_output('kel', val=np.zeros((nElem,12,12)), units='N/m')

	def setup_partials(self):
		self.declare_partials('kel', ['kel_mat', 'kel_geom'])

	def compute(self, inputs, outputs):
		nElem = self.options['nElem']

		kem = inputs['kel_mat']
		keg = inputs['kel_geom']

		outputs['kel'] += kem + keg

	def compute_partials(self, inputs, partials):
		nElem = self.options['nElem']
		
		partials['kel', 'kel_mat'] = np.zeros(((nElem * 12 * 12), (nElem * 12 * 12)))
		partials['kel', 'kel_geom'] = np.zeros(((nElem * 12 * 12), (nElem * 12 * 12)))
