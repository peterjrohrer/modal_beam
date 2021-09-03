import numpy as np
import openmdao.api as om
import myconstants as myconst
import pandas as pd
import matplotlib.pyplot as plt

from wind_wave_prob import WindWaveProb

# Load input data (starting points)
design = 'tian_10mw'
inputs_file = 'inputs/'+design+'.txt'
with open(inputs_file, 'r') as f:
    s = f.read()
    input_data = eval(s)

# Create arrays of frequencies
freqs = {
    'omega': np.linspace(0.1, 9.5, 240),
    'omega_wave': np.linspace(0.05, 10.0, 120),
    'white_noise_wave': False
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
prob = om.Problem(model)

# Setup and run problem
prob.setup()

# prob.model.list_inputs()
# prob.model.list_outputs()

# # Add simple linear solver
# prob.model.linear_solver = om.LinearRunOnce()

#prob.set_solver_print(0)
prob.run_model()

# Total Wave Forces and phases
mode1_wave_forces = np.abs(prob['Re_wave_force_mode1'] + 1j * prob['Im_wave_force_mode1'])
mode2_wave_forces = np.abs(prob['Re_wave_force_mode2'] + 1j * prob['Im_wave_force_mode2'])
mode3_wave_forces = np.abs(prob['Re_wave_force_mode3'] + 1j * prob['Im_wave_force_mode3'])
mode1_phase = np.angle(prob['Re_wave_force_mode1'] + 1j * prob['Im_wave_force_mode1'])*(180./np.pi)
mode2_phase = np.angle(prob['Re_wave_force_mode2'] + 1j * prob['Im_wave_force_mode2'])*(180./np.pi)
mode3_phase = np.angle(prob['Re_wave_force_mode3'] + 1j * prob['Im_wave_force_mode3'])*(180./np.pi)

# RAOs
mode1_wave_RAO = np.abs(prob['Re_RAO_wave_mode1'] + 1j*prob['Im_RAO_wave_mode1'])
mode1_wave_RAO_phase = np.angle(prob['Re_RAO_wave_mode1'] + 1j*prob['Im_RAO_wave_mode1'])*(180./np.pi)
mode1_wind_RAO = np.abs(prob['Re_RAO_wind_mode1'] + 1j*prob['Im_RAO_wind_mode1'])
mode2_wave_RAO = np.abs(prob['Re_RAO_wave_mode2'] + 1j*prob['Im_RAO_wave_mode2'])
mode2_wave_RAO_phase = np.angle(prob['Re_RAO_wave_mode2'] + 1j*prob['Im_RAO_wave_mode2'])*(180./np.pi)
mode2_wind_RAO = np.abs(prob['Re_RAO_wind_mode2'] + 1j*prob['Im_RAO_wind_mode2'])
mode3_wave_RAO = np.abs(prob['Re_RAO_wave_mode3'] + 1j*prob['Im_RAO_wave_mode3'])
mode3_wave_RAO_phase = np.angle(prob['Re_RAO_wave_mode3'] + 1j*prob['Im_RAO_wave_mode3'])*(180./np.pi)
mode3_wind_RAO = np.abs(prob['Re_RAO_wind_mode3'] + 1j*prob['Im_RAO_wind_mode3'])

tt_wave_RAO = np.abs(prob['Re_RAO_wave_towertop'] + 1j*prob['Im_RAO_wave_towertop'])
tt_wave_RAO_phase = np.angle(prob['Re_RAO_wave_towertop'] + 1j*prob['Im_RAO_wave_towertop'])
tt_resp_spectra = prob['resp_towertop']
tt_resp_std_dev = prob['stddev_towertop']
tt_acc_spectra = prob['acc_towertop']
tt_acc_std_dev = prob['stddev_acc_towertop']

H_feedbk = prob['Re_H_feedbk'] + 1j * prob['Im_H_feedbk']

eval_freqs = model.options['freqs']['omega']
eval_freqs_hz = eval_freqs/(2*np.pi)
eval_pers = 1/eval_freqs_hz

## --- DEBUGGING PRINTOUTS
print('Total Cost $%3.2f' %prob['total_cost'])
print('Pretension Line: %3.3E kN' % (prob['pretension_line']/1000.))
print('Substruc Volume: %3.3E m^3' % prob['sub_vol'])
print('Substruc Mass: %3.3E kg' % prob['M_sub'])
print('Mode 1 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_1']), (2*np.pi*float(prob['eig_freq_1'])), (float(prob['eig_freq_1']))))
print('Mode 2 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_2']), (2*np.pi*float(prob['eig_freq_2'])), (float(prob['eig_freq_2']))))
print('Mode 3 Nat. Period: %3.2f s, (freq: %3.3f rad/s, %3.3f Hz)' % (1./float(prob['eig_freq_3']), (2*np.pi*float(prob['eig_freq_3'])), (float(prob['eig_freq_3']))))
# print('Pitch Stiffness: %3.3E ' % (prob['K_global'][1,1]-prob['K55_moor']) )
print('Center of Buoy.: %3.3f m' % prob['CoB'])
print('M11: %3.3E kg' %prob['M_global'][0,0])
print('M22: %3.3E kg' %prob['M_global'][1,1])
print('M33: %3.3E kg' %prob['M_global'][2,2])
print('A11: %3.3E kg' % prob['A_global'][0,0])
print('A22: %3.3E kg' % prob['A_global'][1,1])
print('A33: %3.3E kg' % prob['A_global'][2,2])
print('B11: %3.3E N/s' % prob['B_global'][0,0])
print('B22: %3.3E N/s' % prob['B_global'][1,1])
print('B33: %3.3E N/s' % prob['B_global'][2,2])
print('K11: %3.3E N/m' %prob['K_global'][0,0])
print('K22: %3.3E N/m' %prob['K_global'][1,1])
print('K33: %3.3E N/m' %prob['K_global'][2,2])
print('Center of Grav.: %3.3f m' % prob['CoG_tot'])
print('==================')
print('Towertop Response Std. Dev: %3.4f m' % tt_resp_std_dev)
print('Towertop Accel Std. Dev: %3.4f m/s^2' % tt_acc_std_dev)

# # --- VISUALIZE TRANSFER FUNCTION
# font = {'size': 16}
# plt.rc('font', **font)
# fig, ax = plt.subplots(figsize=(8,6))

# H_maxes_idx = np.argmax(np.abs(H_feedbk),axis=0)
# H_max_freqs = eval_freqs[H_maxes_idx]

# plt.imshow(H_max_freqs, cmap='plasma', interpolation='nearest', aspect='equal')
# ax = plt.gca()
# # Major ticks
# ax.set_xticks(np.arange(0, 6, 1))
# ax.set_yticks(np.arange(0, 11, 1))
# # Labels for major ticks
# # ax.set_xticklabels(np.arange(0, 6, 1))
# ax.set_xticklabels([r'$v_{F_T}$',r'$v_{M_T}$',r'$v_{Q_A}$',r'$F_{W_1}$',r'$F_{W_2}$',r'$F_{W_3}$'])
# # ax.set_yticklabels(np.arange(0, 11, 1))
# ax.set_yticklabels([r'$\chi_1$',r'$\chi_2$',r'$\chi_3$',r'$\dot{\chi}_1$',r'$\dot{\chi}_2$',r'$\dot{\chi}_3$',r'$\dot{\phi}$',r'$\phi_{LP}$',r'$\dot{\phi}_{LP}$',r'$x_{LP}$',r'$\dot{x}_{LP}$'])
# # Minor ticks
# ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
# ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
# # Gridlines based on minor ticks
# ax.grid(which='minor', color='w', linestyle='-', linewidth=1)   
# ax.set_title('Transfer Function Frequency of Maximum Response')
# plt.colorbar(label='Frequency [rad/s]')
# plt.tight_layout()
# plt.show()

## --- Plotting Mode 2 SDOF
H_mode2 = 1. / ((-1. * eval_freqs * eval_freqs * (prob['M_global'][1,1] + prob['A_global'][1,1])) + (1j * eval_freqs * prob['B_global'][1,1]) + (prob['K_global'][1,1]))
RAO_mode2 = np.abs(mode2_wave_forces * H_mode2)

font = {'size': 16}
plt.rc('font', **font)
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(14,6))

# Plot Forcing
mode1_wave_force = ax1.plot(eval_freqs, mode1_wave_forces, label='Mode 1')
mode2_wave_force = ax1.plot(eval_freqs, mode2_wave_forces, label='Mode 2')
mode3_wave_force = ax1.plot(eval_freqs, mode3_wave_forces, label='Mode 3')
# Plot Transfer Function (from matrix)
# mode2_txfer_func1 = ax2.plot(eval_freqs, np.abs(H_feedbk[:,1,3]), label='Mode 2 - matrix 21')
# mode2_txfer_func2 = ax2.plot(eval_freqs, np.abs(H_feedbk[:,1,4]), label='Mode 2 - matrix 22')
# mode2_txfer_func3 = ax2.plot(eval_freqs, np.abs(H_feedbk[:,1,5]), label='Mode 2 - matrix 23')
mode2_txfer_func_all = ax2.plot(eval_freqs, np.abs(H_feedbk[:,1,3] + H_feedbk[:,1,4] + H_feedbk[:,1,5]), label='Mode 2 - sum')
# Plot Transfer Function (calculated)
mode2_txfer_func = ax2.plot(eval_freqs, np.abs(H_mode2), label='Mode 2 - calculated')
# Plot RAOs (matrix and calculated)
mode2_rao_mat = ax3.plot(eval_freqs, mode2_wave_RAO, label='Mode 2 - matrix')
mode2_rao_calc = ax3.plot(eval_freqs, RAO_mode2, label='Mode 2 - calculated')

ax1.grid()
ax1.legend()
ax2.grid()
ax2.legend()
ax3.grid()
ax3.legend()
ax1.set_xlim([0,10.0])
ax2.set_xlim([0,10.0])
ax3.set_xlim([0,10.0])
ax1.set_title('Forcing')
ax2.set_title('Transfer Function')
ax3.set_title('RAO')
ax1.set_xlabel('Frequency [rad/s]')
ax1.set_ylabel('Amplitude [N]')
ax2.set_xlabel('Frequency [rad/s]')
ax2.set_ylabel('Amplitude [-]')
ax3.set_xlabel('Frequency [rad/s]')
ax3.set_ylabel('Amplitude [m/m]')

# Show plot
plt.tight_layout()
plt.show()

# ## --- Plotting Wave Forces
# font = {'size': 16}
# plt.rc('font', **font)
# fig, ax = plt.subplots(figsize=(10,6))

# # Plot spar
# mode1_wave_force = ax.plot(eval_freqs, mode1_wave_forces, label='Mode 1')
# mode2_wave_force = ax.plot(eval_freqs, mode2_wave_forces, label='Mode 2')
# mode3_wave_force = ax.plot(eval_freqs, mode3_wave_forces, label='Mode 3')

# ax.grid()
# ax.set_xlim([0,5.0])
# ax.set_title('Modal Wave Force Amplitude')
# ax.set_xlabel('Frequency [rad/s]')
# ax.set_ylabel('Amplitude [N]')
# plt.legend()

# # Show plot
# plt.tight_layout()
# plt.show()

## --- Plotting RAOs
font = {'size': 16}
plt.rc('font', **font)
fig, ax = plt.subplots(figsize=(10,6))

# Plot spar
mode1_wave = ax.plot(eval_freqs, mode1_wave_RAO, label='Mode 1')
mode2_wave = ax.plot(eval_freqs, mode2_wave_RAO, label='Mode 2')
mode3_wave = ax.plot(eval_freqs, mode3_wave_RAO, label='Mode 3')
tt_wave = ax.plot(eval_freqs, tt_wave_RAO, label='Towertop')

ax.grid()
ax.set_xlim([0,5.0])
if freqs['white_noise_wave'] :
    ax.set_title('Modal Response - White Noise Wave Spectrum')
else :
    ax.set_title('Modal Response - Single Wave Spectrum')
ax.set_xlabel('Frequency [rad/s]')
ax.set_ylabel('Response [m/m]')
ax.set_yscale('log')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

## --- Plotting Spectrum
# Read in from SIMA
# run_id = 'june21/WhiteNoise_mor'
# run_id = 'june21/WhiteNoise_mac'
# run_id = 'june21/SingleWave_mor'
# run_id = 'june23/WhiteNoise'
run_id = 'june23/SingleWave2'
file_motions = '/Users/peter/thesis_local/ntnu_remote/'+run_id+'/nachub_motions.csv'
file_spec = '/Users/peter/thesis_local/ntnu_remote/'+run_id+'/nachub_specdata.csv'
df_motions = pd.read_csv(file_motions)
df_spec = pd.read_csv(file_spec)
df_spec.columns = ['Mot. Freq [rad/s]', 'Mot. Spec [m^2*s/rad]', 'Acc. Freq [rad/s]', 'Acc. Spec [m^2/(s^3*rad)]']
sima_tt_resp_std_dev = np.std(df_motions['X-Motion [m]'])
sima_tt_acc_std_dev = np.std(df_motions['X-Acceleration [m/s^2]'])

font = {'size': 14}
plt.rc('font', **font)
fig, (ax1, ax2) = plt.subplots(figsize=(14,8),nrows=1,ncols=2,)

tt_resp_wave = ax1.plot(eval_freqs, tt_resp_spectra, label='Std Dev: %3.3f m' % tt_resp_std_dev)
tt_resp_wave_sima = ax1.plot(df_spec['Mot. Freq [rad/s]'], df_spec['Mot. Spec [m^2*s/rad]'], label='SIMA: Std Dev: %3.3f m' % sima_tt_resp_std_dev)

tt_acc_wave = ax2.plot(eval_freqs, tt_acc_spectra, label='TLPOPT: Std Dev: %3.3f m/s^2' % tt_acc_std_dev)
tt_acc_wave_sima = ax2.plot(df_spec['Acc. Freq [rad/s]'], df_spec['Acc. Spec [m^2/(s^3*rad)]'], label='SIMA: Std Dev: %3.3f m/s^2' % sima_tt_acc_std_dev)

ax1.grid()
ax1.set_xlim([0,5.0])
ax2.grid()
ax2.set_xlim([0,5.0])

if freqs['white_noise_wave'] :
    ax1.set_title('Nacelle Surge Motion - White Noise Wave Spectrum')
    ax2.set_title('Nacelle Surge Acceleration - White Noise Wave Spectrum')
else :
    ax1.set_title('Nacelle Surge Motion - Single Wave Spectrum')
    ax2.set_title('Nacelle Surge Acceleration - Single Wave Spectrum')

ax1.set_xlabel('Frequency [rad/s]')
ax2.set_xlabel('Frequency [rad/s]')
ax1.set_ylabel('Spectrum [m^2*s/rad]')
ax2.set_ylabel('Spectrum [m^2/(s^3*rad)]')
ax1.legend()
ax2.legend()

# Show plot
plt.tight_layout()
plt.show()