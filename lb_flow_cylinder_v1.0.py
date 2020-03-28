#!/usr/bin/env python3

# Copyright (C) 2013 FlowKit Ltd, Lausanne, Switzerland
# E-mail contact: contact@flowkit.com
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.

##
PROGRAM='LB_FLOW_CYLINDER_V1.0.PY'
##

##
VERSION='v1.0: 26-Mar-2020'
##

##
## Author: Patrick Treuthardt
##

##
## Description:	Fluid dynamics code for, eventually, simulating a springtail in air.
##
##              Original Description:
##                  2D flow around a cylinder
##                  Copyright (C) 2013 FlowKit Ltd, Lausanne, Switzerland
##                  E-mail contact: contact@flowkit.com
##
##                  This program is free software: you can redistribute it and/or
##                  modify it under the terms of the GNU General Public License, either
##                  version 3 of the License, or (at your option) any later version.
##

##
## Revision History:	v1.0: 26-Mar-2020 - first attempt to recode original program into one with a set of functions
##

##
## Requirements:	None
##

##
## Import System Libraries
##
from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import os

##
## GLOBAL VARIABLES
##
ROOT_DIR='/home/patrick/Work/16032020/Fluid_dynamics/'
IMAGE_DIR='test_temp/'

##
## Let's go!
##
def lets_go():
	print('\n-----')
	print("Running "+PROGRAM+" "+VERSION)
	print('-----\n')
	return()

##
## Initialize start time
##
def start_timer():
	start_time=time.time() #initalize time stamp
	return(start_time)

##
## Check if directory to store images exists. If not, create it.
##
def check_image_dir():
    print('\nChecking if '+ROOT_DIR+IMAGE_DIR+' exists...\n')

    if not os.path.exists(ROOT_DIR+IMAGE_DIR):
        os.makedirs(ROOT_DIR+IMAGE_DIR)
        print('     Nope. Created directory.\n')
    return()

#check_image_dir()
##
## Flow definition
##
def flow_definition():
    maxIter = 200000 # Total number of time iterations.

    Re      = 220.0  # Reynolds number.
    #Re = 245.65 # approximate Reynolds number for springtail in simulation

    # Lattice dimensions and populations.
    nx = 520
    ny = 180
    ly = ny-1.0
    q = 9

    # Coordinates of the cylinder.
    cx = nx/4
    cy = ny/2
    r = ny/9

    uLB     = 0.04                       # Velocity in lattice units.

    # Relaxation parameter.
    nulb = uLB*r/Re
    omega = 1.0 / (3.*nulb+0.5)

    return(maxIter,nx,ny,ly,q,cx,cy,r,uLB,omega)
###### Flow definition #########################################################
maxIter = 200000 # Total number of time iterations.
Re      = 220.0  # Reynolds number.
#Re = 245.65
nx = 520; ny = 180; ly=ny-1.0; q = 9 # Lattice dimensions and populations.
cx = nx/4; cy=ny/2; r=ny/9;          # Coordinates of the cylinder.
uLB     = 0.04                       # Velocity in lattice units.
nulb    = uLB*r/Re; omega = 1.0 / (3.*nulb+0.5); # Relaxation parameter.

##
## Lattice Constants
##
def lattice_constants(q):
    c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.

    t = 1./36. * ones(q)                                   # Lattice weights.
    t[asarray([norm(ci)<1.1 for ci in c])] = 1./9.
    t[0] = 4./9.

    noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)]

    i1 = arange(q)[asarray([ci[0]<0  for ci in c])] # Unknown on right wall.

    i2 = arange(q)[asarray([ci[0]==0 for ci in c])] # Vertical middle.

    i3 = arange(q)[asarray([ci[0]>0  for ci in c])] # Unknown on left wall.

    return(c,t,noslip,i1,i2,i3)
###### Lattice Constants #######################################################
c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.
t = 1./36. * ones(q)                                   # Lattice weights.
t[asarray([norm(ci)<1.1 for ci in c])] = 1./9.; t[0] = 4./9.
noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)]
i1 = arange(q)[asarray([ci[0]<0  for ci in c])] # Unknown on right wall.
i2 = arange(q)[asarray([ci[0]==0 for ci in c])] # Vertical middle.
i3 = arange(q)[asarray([ci[0]>0  for ci in c])] # Unknown on left wall.

##
## Other function definitions
##
#sumpop = lambda fin: sum(fin,axis=0) # Helper function for density computation.

#def equilibrium(rho,u,c,q,nx,ny):              # Equilibrium distribution function.
#    cu   = 3.0 * dot(c,u.transpose(1,0,2))
#    usqr = 3./2.*(u[0]**2+u[1]**2)
#    feq = zeros((q,nx,ny))
#    for i in range(q): feq[i,:,:] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
#    return(feq)
###### Function Definitions ####################################################
sumpop = lambda fin: sum(fin,axis=0) # Helper function for density computation.
def equilibrium(rho,u):              # Equilibrium distribution function.
    cu   = 3.0 * dot(c,u.transpose(1,0,2))
    usqr = 3./2.*(u[0]**2+u[1]**2)
    feq = zeros((q,nx,ny))
    for i in range(q): feq[i,:,:] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return(feq)

##
## Setup: cylindrical obstacle and velocity inlet with perturbation
##
def obstacle_perturbation(cx,cy,nx,ny,r):
    obstacle = fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (nx,ny))

    vel = fromfunction(lambda d,x,y: (1-d)*uLB*(1.0+1e-4*sin(y/ly*2*pi)),(2,nx,ny))

    feq = equilibrium(1.0,vel)
    fin = feq.copy()

    return(fin)
###### Setup: cylindrical obstacle and velocity inlet with perturbation ########
obstacle = fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (nx,ny))
vel = fromfunction(lambda d,x,y: (1-d)*uLB*(1.0+1e-4*sin(y/ly*2*pi)),(2,nx,ny))
feq = equilibrium(1.0,vel); fin = feq.copy()

##
## Main time loop
##
def time_loop():
    fin[i1,-1,:] = fin[i1,-2,:] # Right wall: outflow condition.
    rho = sumpop(fin)           # Calculate macroscopic density and velocity.
    u = dot(c.transpose(), fin.transpose((1,0,2)))/rho

    u[:,0,:] =vel[:,0,:] # Left wall: compute density from known populations.
    rho[0,:] = 1./(1.-u[0,0,:]) * (sumpop(fin[i2,0,:])+2.*sumpop(fin[i1,0,:]))

    feq = equilibrium(rho,u) # Left wall: Zou/He boundary condition.
    fin[i3,0,:] = fin[i1,0,:] + feq[i3,0,:] - fin[i1,0,:]
    fout = fin - omega * (fin - feq)  # Collision step.
    for i in range(q): fout[i,obstacle] = fin[noslip[i],obstacle]
    for i in range(q): # Streaming step.
        fin[i,:,:] = roll(roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)

    if (time%100==0): # Visualization
        plt.clf(); plt.imshow(sqrt(u[0]**2+u[1]**2).transpose(),cmap=cm.Reds)
        plt.savefig("vel."+str(time/100).zfill(4)+".png")
    return()
###### Main time loop ##########################################################
for time in range(maxIter):
    fin[i1,-1,:] = fin[i1,-2,:] # Right wall: outflow condition.
    rho = sumpop(fin)           # Calculate macroscopic density and velocity.
    u = dot(c.transpose(), fin.transpose((1,0,2)))/rho

    u[:,0,:] =vel[:,0,:] # Left wall: compute density from known populations.
    rho[0,:] = 1./(1.-u[0,0,:]) * (sumpop(fin[i2,0,:])+2.*sumpop(fin[i1,0,:]))

    feq = equilibrium(rho,u) # Left wall: Zou/He boundary condition.
    fin[i3,0,:] = fin[i1,0,:] + feq[i3,0,:] - fin[i1,0,:]
    fout = fin - omega * (fin - feq)  # Collision step.
    for i in range(q): fout[i,obstacle] = fin[noslip[i],obstacle]
    for i in range(q): # Streaming step.
        fin[i,:,:] = roll(roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)

    if (time%100==0): # Visualization
        plt.clf(); plt.imshow(sqrt(u[0]**2+u[1]**2).transpose(),cmap=cm.Reds)
        plt.savefig(ROOT_DIR+IMAGE_DIR+"vel."+str(time/100).zfill(4)+".png")

##
## Calculate and print elapsed time
##
#def elapsed_time(start_time):
	elapsed_time=(time.time()-start_time)
	print('\nThe elapsed runtime is ',elapsed_time,' seconds.\n')

##
## MAIN
##
#start_time=start_timer()

#lets_go()

#check_image_dir()

#maxIter,nx,ny,ly,q,cx,cy,r,uLB,omega=flow_definition()

#c,t,noslip,i1,i2,i3=lattice_constants(q)

#obstacle_perturbation()

#elapsed_time(start_time)
