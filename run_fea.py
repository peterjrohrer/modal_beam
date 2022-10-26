#!/usr/bin/env python
# from: https://gist.github.com/calebrob6/eb98725187425c527567
import sys,os
import time
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

from beam_props import *
from elem_props import *
from global_props import *
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
    nElem = 10 # Number of elements along the beam
    nNode = nElem +1 # Number of nodes (one on each end of each element)
    nDOFperNode = 2
    nNodesPerElem = 2
    nDOF_tot = nNode * nDOFperNode

    D_beam = 2.3 * np.ones(nElem) # m
    wt_beam = 0.2 * np.ones(nElem) # m
    L_tot = 25 # m

    # Map nodes/DOF
    Elem2Nodes, Nodes2DOF, Elem2DOF = LinearDOFMapping(nElem, nNodesPerElem, nDOFperNode)

    # Read inputs to beam properties
    x_beamnode, x_beamelem, L_beam, M_beam, tot_M_beam = BeamProps(nNode=nNode, nElem=nElem, D_beam=D_beam, wt_beam=wt_beam, L_tot=L_tot)

    # Preallocate mel/kel and M_glob/K_glob matrices
    mel = np.zeros((nElem, 4, 4))
    kel = np.zeros((nElem, 4, 4))
    M_glob =  np.zeros((nDOF_tot,nDOF_tot))
    K_glob =  np.zeros((nDOF_tot,nDOF_tot))

    # --- Element calculations
    for i in range(nElem):
        mel[i,:,:] = ElemMass(M_beam[i], L_beam[i])
        kel_mat = ElemMatStiff(D_beam[i], wt_beam[i], L_beam[i])
        kel_geom = ElemGeomStiff(P_elem=0, L_elem=L_beam[i])
        kel[i,:,:] = ElemStiff(kel_mat, kel_geom)
    
    # --- Assembly    
    for i in range(nElem):
        DOFindex=Elem2DOF[i,:]
        M_glob = BuildGlobalMatrix(M_glob, mel[i,:,:], DOFindex)
        K_glob = BuildGlobalMatrix(K_glob, kel[i,:,:], DOFindex)

    # --- Handle BC and root/tip conditions
    # BC_root = [0,0,0,0,0,0]
    # BC_tip  = [1,1,1,1,1,1]
    BC_root = [0,0]
    BC_tip  = [1,1]

    # Tip and root degrees of freedom
    IDOF_root = Nodes2DOF[Elem2Nodes[0,:][0] ,:]
    IDOF_tip  = Nodes2DOF[Elem2Nodes[-1,:][1],:]

    # Insert tip/root inertias
    if M_root is not None:
        print('Not handled yet!')
        M_glob[np.ix_(IDOF_root, IDOF_root)] += M_root
    if M_tip is not None:
        print('Not handled yet!')
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



    self.add_subsystem('modeshape_group',
        modeshape_group,
        promotes_inputs=['Z_beam', 'D_beam', 'L_beam', 'M_beam', 'tot_M_beam', 'wt_beam'],
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

    return FEM

def plotFEM(FEM):

    return 1.

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
    