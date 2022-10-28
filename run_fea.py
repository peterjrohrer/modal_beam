#!/usr/bin/env python
# from: https://gist.github.com/calebrob6/eb98725187425c527567
import sys,os
import time
import argparse

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import linalg

from beam_props import *
from elem_props import *
from global_props import *
from eigenproblem import *
from modeshape import * 
from modal_analysis import *
from utils import *

MyDir=os.path.dirname(__file__)

def doArgs(argList, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument('-v', "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    # parser.add_argument('--input', action="store", dest="inputFn", type=str, help="Input file name", required=True)
    # parser.add_argument('--output', action="store", dest="outputFn", type=str, help="Output file name", required=True)

    return parser.parse_args(argList)

def UniformBeam():   
    # --- Parameters
    nElems = [50,15,30] # Number of elements along the beam
    nNodes = [31,11,11] # Number of nodes (one on each end of each element)
    nDOFperNode = 6
    nNodesPerElem = 2

    D_beam = 0.25 # m
    D_beam_0 = (D_beam*np.ones(nElems[0]))
    D_beam_1 = (D_beam*np.ones(nElems[1]))
    D_beam_2 = (D_beam*np.ones(nElems[2]))
    wt_beam = 0.01 # m
    wt_beam_0 = (wt_beam*np.ones(nElems[0]))
    wt_beam_1 = (wt_beam*np.ones(nElems[1]))
    wt_beam_2 = (wt_beam*np.ones(nElems[2]))
    L_tot = [5.,1.,2.] # m

    # Read inputs to beam properties
    x_beamnode_0, _, L_beam_0, M_beam_0, _ = BeamProps(nNode=nNodes[0], nElem=nElems[0], D_beam=D_beam_0, wt_beam=wt_beam_0, L_tot=L_tot[0])
    x_beamnode_1, _, L_beam_1, M_beam_1, _ = BeamProps(nNode=nNodes[1], nElem=nElems[1], D_beam=D_beam_1, wt_beam=wt_beam_1, L_tot=L_tot[1])
    x_beamnode_2, _, L_beam_2, M_beam_2, _ = BeamProps(nNode=nNodes[2], nElem=nElems[2], D_beam=D_beam_2, wt_beam=wt_beam_2, L_tot=L_tot[2])

    # Add additional beam
    x_beamnode = np.concatenate((x_beamnode_0, x_beamnode_0[12] * np.ones_like(x_beamnode_1)[1:], x_beamnode_0[7] * np.ones_like(x_beamnode_2)[1:]))
    y_beamnode = np.zeros_like(x_beamnode)
    z_beamnode = np.concatenate((np.zeros_like(x_beamnode_0),x_beamnode_1[1:],-1.*x_beamnode_2[1:]))
    
    D_beam = np.concatenate((D_beam_0, D_beam_1, D_beam_2))
    wt_beam = np.concatenate((wt_beam_0, wt_beam_1, wt_beam_2))
    L_beam = np.concatenate((L_beam_0, L_beam_1, L_beam_2))
    M_beam = np.concatenate((M_beam_0, M_beam_1, M_beam_2))
    
    nElem = sum(nElems)

    # Map nodes/DOF
    Elem2Nodes, Nodes2DOF, Elem2DOF, Layout, nNode = LinearDOFMapping(nNodesPerElem, nDOFperNode, nElems=nElems, atNodes=[10,35])

    nDOF_tot = nNode * nDOFperNode
    
    # Find DCM
    DCM = elementDCMfromBeamNodes(x_beamnode,y_beamnode,z_beamnode)

    # Preallocate mel/kel and M_glob/K_glob matrices
    mel = np.zeros((nElem, 12, 12))
    kel = np.zeros((nElem, 12, 12))
    M_glob =  np.zeros((nDOF_tot,nDOF_tot))
    K_glob =  np.zeros((nDOF_tot,nDOF_tot))

    # --- Element calculations
    for i in range(nElem):
        mel[i,:,:] = ElemMass(M_beam[i], D_beam[i], wt_beam[i], L_beam[i], R=DCM[i,:,:])
        kel_mat = ElemMatStiff(D_beam[i], wt_beam[i], L_beam[i], R=DCM[i,:,:])
        kel_geom = ElemGeomStiff(P_elem=0, L_elem=L_beam[i], R=DCM[i,:,:])
        kel[i,:,:] = ElemStiff(kel_mat, kel_geom)
    
    # --- Assembly    
    for i in range(nElem):
        DOFindex=Elem2DOF[i,:]
        M_glob = BuildGlobalMatrix(M_glob, mel[i,:,:], DOFindex)
        K_glob = BuildGlobalMatrix(K_glob, kel[i,:,:], DOFindex)

    IDOF_All = np.arange(0,nDOF_tot)
    # Tip and root degrees of freedom
    IDOF_root = Nodes2DOF[Layout[0][0]]
    IDOF_tip  = Nodes2DOF[Layout[0][-1]]
    

    # --- Handle BC and root/tip conditions
    BC_root = [0,0,0,0,0,0]
    # BC_root  = [1,1,1,1,1,1]
    BC_tip  = [1,1,1,1,1,1]
    
    M_root = None
    M_tip = PointMassMatrix(m=100000.,Ref2COG=(0,0.2,0))
    K_root = None
    K_tip = None

    # Insert tip/root inertias
    if M_root is not None:
        M_glob[np.ix_(IDOF_root, IDOF_root)] += M_root
    if M_tip is not None:
        M_glob[np.ix_(IDOF_tip, IDOF_tip)]   += M_tip

    # Insert tip/root stiffness
    if K_root is not None:
        K_glob[np.ix_(IDOF_root, IDOF_root)] += K_root
    if K_tip is not None:
        K_glob[np.ix_(IDOF_tip, IDOF_tip)] += K_tip

    # Boundary condition transformation matrix (removes row/columns)
    Tr=np.eye(nDOF_tot)

    # Root and Tip BC
    IDOF_removed = [i for i,iBC in zip(IDOF_root, BC_root) if iBC==0]
    IDOF_removed += [i for i,iBC in zip(IDOF_tip, BC_tip) if iBC==0]
    Tr = np.delete(Tr, IDOF_removed, axis=1) # removing columns

    Mr = (Tr.T).dot(M_glob).dot(Tr)
    Kr = (Tr.T).dot(K_glob).dot(Tr)

    # --- Create mapping from M to Mr
    nDOF_r = Mr.shape[0]
    IDOF_BC = list(np.setdiff1d(IDOF_All, IDOF_removed))
    IFull2BC = np.zeros(nDOF_tot,dtype=int)
    IBC2Full = np.zeros(nDOF_r,dtype=int)
    k=0
    for i in IDOF_All:
        if i in IDOF_removed:
            IFull2BC[i]=-1
        else:
            IFull2BC[i]=k
            IBC2Full[k]=i
            k+=1
    
    Qr, eigfreqs, spectr = Eigenproblem(Mr, Kr)
    # Add fixed BC back into eigenvectors
    Q = Tr.dot(Qr)
    
    nModes = 5
    # Need to add original shape of x_nodes to get displacements
    x_nodes = np.reshape(np.tile(x_beamnode,nModes),(nNode,nModes),order='F')
    y_nodes = np.reshape(np.tile(y_beamnode,nModes),(nNode,nModes),order='F')
    z_nodes = np.reshape(np.tile(z_beamnode,nModes),(nNode,nModes),order='F')
    # y_nodes = np.zeros((nNode,nModes))
    # z_nodes = np.zeros((nNode,nModes))
    z_d_nodes = np.zeros((nNode,nModes))
    z_dd_nodes = np.zeros((nNode,nModes))
    z_elems = np.zeros((nElem,nModes))

    for i in range(nModes):
        x_nodes_0, y_nodes_0, z_nodes_0 = Modeshape(nNode, nElem, nDOFperNode, Q[:,i])
        x_nodes[:,i] += x_nodes_0
        y_nodes[:,i] += y_nodes_0
        z_nodes[:,i] += z_nodes_0
        # lhs = SplineLHS(x_beamnode)
        # rhs1 = SplineRHS(x_beamnode, z_nodes[:,i])
        # z_d_nodes[:,i] = SolveSpline(lhs,rhs1)
        # rhs2 = SplineRHS(x_beamnode, z_d_nodes[:,i])
        # z_dd_nodes[:,i] = SolveSpline(lhs,rhs2)
        # z_elems[:, i] = ModeshapeElem(x_beamnode, z_nodes[:,i], z_d_nodes[:,i])

    print('----- FROM FINITE ELEMENT MODEL -----')
    print('Mode 1 Nat. Freq: %3.3f Hz' %(eigfreqs[0]))
    print('Mode 2 Nat. Freq: %3.3f Hz' %(eigfreqs[1]))
    print('Mode 3 Nat. Freq: %3.3f Hz' %(eigfreqs[2]))
    print('Mode 4 Nat. Freq: %3.3f Hz' %(eigfreqs[3]))
    print('Mode 5 Nat. Freq: %3.3f Hz' %(eigfreqs[4]))
    # print('Mode 6 Nat. Freq: %3.3f Hz' %(eigfreqs[5]))
    # print('Mode 7 Nat. Freq: %3.3f Hz' %(eigfreqs[6]))
    # print('Mode 8 Nat. Freq: %3.3f Hz' %(eigfreqs[7]))
    # print('Mode 9 Nat. Freq: %3.3f Hz' %(eigfreqs[8]))
    # print('Mode 10 Nat. Freq: %3.3f Hz' %(eigfreqs[9]))

	## --- Check Frequencies
    M_modal = Q[:,:nModes].T @ M_glob @ Q[:,:nModes]
    K_modal = Q[:,:nModes].T @ K_glob @ Q[:,:nModes]

    eig_vals, eig_vecs = scipy.linalg.eig(K_modal, M_modal)
    modal_freqs = np.sort(np.sqrt(np.real(eig_vals)) /(2*np.pi))
	
    print('----- FROM MODAL MATRICES -----')
    print('Mode 1 Nat. Freq: %3.3f Hz' %(modal_freqs[0]))
    print('Mode 2 Nat. Freq: %3.3f Hz' %(modal_freqs[1]))
    print('Mode 3 Nat. Freq: %3.3f Hz' %(modal_freqs[2]))
    print('Mode 4 Nat. Freq: %3.3f Hz' %(modal_freqs[3]))
    print('Mode 5 Nat. Freq: %3.3f Hz' %(modal_freqs[4]))
    # print('Mode 6 Nat. Freq: %3.3f Hz' %(modal_freqs[5]))
    # print('Mode 7 Nat. Freq: %3.3f Hz' %(modal_freqs[6]))
    # print('Mode 8 Nat. Freq: %3.3f Hz' %(modal_freqs[7]))
    # print('Mode 9 Nat. Freq: %3.3f Hz' %(modal_freqs[8]))
    # print('Mode 10 Nat. Freq: %3.3f Hz' %(modal_freqs[9]))

    # --- Return a dictionary
    FEM = {
        'xNodes':x_nodes,
        'yNodes':y_nodes,
        'zNodes':z_nodes, 
        'MM':Mr, 
        'KK':Kr, 
        'MM_full':M_glob, 
        'KK_full':K_glob, 
        'Tr':Tr,
        # 'IFull2BC':IFull2BC, 
        # 'IBC2Full':IBC2Full,
        'Elem2Nodes':Elem2Nodes, 
        'Nodes2DOF':Nodes2DOF, 
        'Elem2DOF':Elem2DOF,
        # 'Q':Q,
        'freq':eigfreqs, 
        'nModes':nModes,
        # 'modeNames':modeNames,
        }
    return FEM

    # self.add_subsystem('modeshape_group',
    #     modeshape_group,
    #     promotes_inputs=['Z_beam', 'D_beam', 'L_beam', 'M_beam', 'tot_M_beam', 'wt_beam'],
    #     promotes_outputs=['eig_vector_*', 'eig_freq_*', 'z_beamnode', 'z_beamelem',
    #         'x_beamnode_*', 'x_d_beamnode_*', 
    #         'x_beamelem_*', 'x_d_beamelem_*', 'x_dd_beamelem_*',
    #         'M11', 'M12', 'M13', 'M22', 'M23', 'M33', 
    #         'K11', 'K12', 'K13', 'K22', 'K23', 'K33',])

    # self.add_subsystem('global_mass',
    #     GlobalMass(),
    #     promotes_inputs=['M11', 'M12', 'M13', 'M22', 'M23', 'M33'],
    #     promotes_outputs=['M_global'])

    # self.add_subsystem('global_stiffness',
    #     GlobalStiffness(),
    #     promotes_inputs=['K11', 'K12', 'K13', 'K22', 'K23', 'K33'],
    #     promotes_outputs=['K_global'])

    # return FEM

def plotFEM(FEM):
    x_nodes = FEM['xNodes']
    y_nodes = FEM['yNodes']
    z_nodes = FEM['zNodes']
    nModes = FEM['nModes']
    eigfreqs = FEM['freq']

    ## --- Shapes Plot from FEA
    font = {'size': 16}
    plt.rc('font', **font)
    fig1, axs = plt.subplot_mosaic([['ul', '.'], ['ll', 'lr']], figsize=(12, 10), layout="constrained", sharey=True)

    for i in range(nModes):
        axs['ul'].plot(x_nodes[:,i], y_nodes[:,i], label='Mode %2d: %2.3f Hz' %((i+1, eigfreqs[i])), ls='None', marker='o', ms=5)
        axs['ll'].plot(x_nodes[:,i], z_nodes[:,i], label='Mode %2d: %2.3f Hz' %((i+1, eigfreqs[i])), ls='None', marker='o', ms=5)
        axs['lr'].plot(y_nodes[:,i], z_nodes[:,i], label='Mode %2d: %2.3f Hz' %((i+1, eigfreqs[i])), ls='None', marker='o', ms=5)

    # Set labels and legend
    axs['ul'].grid()
    axs['ul'].set_xlim(-0.1,5.1)
    axs['ul'].set_ylabel('Y-displacement')
    axs['ll'].grid()
    axs['ll'].set_xlim(-0.1,5.1)
    axs['ll'].set_xlabel('X-displacement')
    axs['ll'].set_ylabel('Z-displacement')
    # axs['ll'].set_ylim(-1,1.1)
    axs['lr'].grid()
    axs['lr'].set_xlim(-1.1,1.1)
    axs['lr'].set_ylim(-1,1.1)
    axs['lr'].set_xlabel('Y-displacement')

    handles, labels = axs['lr'].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper right')
    fig1.suptitle('Modeshapes from FEA')

def main():
    progName = "run_fea_test"
    args = doArgs(sys.argv[1:], progName)

    verbose = args.verbose
    # inputFn = args.inputFn
    # outputFn = args.outputFn

    print("Starting %s" % (progName))
    startTime = float(time.time())

    FEM = UniformBeam()
    plotFEM(FEM)

    # if not os.path.isfile(inputFn):
    #     print("Input doesn't exist, exiting")
    #     return

    # outputBase = os.path.dirname(outputFn)
    # if outputBase!='' and not os.path.exists(outputBase):
    #     print("Output directory doesn't exist, making output dirs: %s" %(outputBase))
    #     os.makedirs(outputBase)
    print("Finished in %0.4f seconds" % (time.time() - startTime))
    return

if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    main()
    plt.show()
    plt.savefig('modeshapes.png')