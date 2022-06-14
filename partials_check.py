import numpy as np
import myconstants as myconst
from openmdao.api import Problem, partial_deriv_plot

from wind_wave_prob import WindWaveProb

# Load input data (starting points)
design = 'tian_10mw'
inputs_file = 'inputs/'+design+'.txt'
with open(inputs_file, 'r') as f:
    s = f.read()
    input_data = eval(s)

# Create arrays of frequencies
freqs = {
    'omega': np.linspace(0.05, 2.5, 240),
    'omega_wave': np.linspace(0.01, 3.0, 120),
    'white_noise_wave': True
}

# Create array of pontoon locations
num_pontoons = 3
pont_data = {
    'N_pont': num_pontoons,
    'N_pontelem': 10,
    'theta': np.delete(np.linspace(np.pi, 3*np.pi, num_pontoons+1),-1),
    'shape_pont': 'rect'
}

# Precomputed blade data
blades = {
    'Rtip' : 89.165, 
    'Rhub' : 2.8, 
    'N_b_elem' : 20, 
    'indfile' : 'WindLoad/DTU10MW_indfacs.dat', 
    'bladefile' : 'WindLoad/DTU10MWblade.dat', 
    'foilnames' : ['foil1', 'foil11', 'foil12', 'foil13', 'foil14', 'foil15', 'foil16', 'foil17', 'foil18', 'foil19'], 
    'foilfolder' : 'WindLoad/Airfoils/', 
    'windfolder' : 'WindLoad/Windspeeds/'
}

# Bring in problem with defined defaults
model = WindWaveProb(freqs=freqs,pont_data=pont_data,blades=blades,input_data=input_data)
prob = Problem(model)

prob.setup(force_alloc_complex=True)
prob.run_model()

comp_to_check = 'wind_wave_group.substructure.modeshape_group.modeshape_elem_normforce'

check_partials_data = prob.check_partials(method='fd',form='central',step=1e-6, includes=comp_to_check, show_only_incorrect=True, compact_print=True)
# check_partials_data = prob.check_partials(method='cs', includes=comp_to_check, show_only_incorrect=True, compact_print=True)

partial_deriv_plot('normforce_mode_elem', 'Z_beam', check_partials_data, binary=False)
