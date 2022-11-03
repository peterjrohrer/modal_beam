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
		nElem = self.nodal_data['nElem']
		nPart = nElem * 12 * 12

		self.declare_partials('kel_loc', ['kel_mat', 'kel_geom'], rows=np.arange(nPart), cols=np.arange(nPart))

	def compute(self, inputs, outputs):
		outputs['kel_loc'] = inputs['kel_mat'] + inputs['kel_geom']

	def compute_partials(self, inputs, partials):
		nElem = self.nodal_data['nElem']
		nPart = nElem * 12 * 12
		
		partials['kel_loc', 'kel_mat'] = np.ones(nPart)
		partials['kel_loc', 'kel_geom'] = np.ones(nPart)
