#!/usr/bin/env python3

# Copyright (C) 2013 FlowKit Ltd, Lausanne, Switzerland
# E-mail contact: contact@flowkit.com
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.

##
PROGRAM='LB_FLOW_CYLINDER_V2.0.PY'
##

##
VERSION='v2.0: 26-Mar-2020'
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
##						v2.0: 26-Mar-2020 - attempt to build around existing code
##

##
## Requirements:	None
##

##
## Import System Libraries
##
import time
from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import subprocess
import shutil

##
## GLOBAL VARIABLES
##
ROOT_DIR='/home/patrick/Work/16032020/Fluid_dynamics/'
IMAGE_ROOT='test_temp'
IMAGE_DIR=IMAGE_ROOT+'/'

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

	if os.path.exists(ROOT_DIR+IMAGE_DIR):
		print("     Yup, it's already there. Deleting and recreating directory.\n")
		shutil.rmtree(ROOT_DIR+IMAGE_DIR)
		os.makedirs(ROOT_DIR+IMAGE_DIR)
	else:
		os.makedirs(ROOT_DIR+IMAGE_DIR)
		print('     Nope. Created directory.\n')

	return()

##
## Calculate and print elapsed time
##
def elapsed_time(start_time):
	elapsed_time=(time.time()-start_time)
	print('\nThe elapsed runtime is '+str(elapsed_time)+' seconds.\n')

##
## Create an animation of the resultant png files saved as an mp4 video file
##
def create_mp4():
	print('\nCreating '+ROOT_DIR+IMAGE_DIR+IMAGE_ROOT+'.mp4...\n')

	os.chdir(ROOT_DIR+IMAGE_DIR) # changes to appropriate directory

	# takes images with 4 digit padding (vel.0001.png)
	# -r 5 give 5 frames per second
	subprocess.call(['ffmpeg', '-f', 'image2', '-r', '10', '-i', 'vel.%04d.png', '-vcodec', 'mpeg4', '-y', IMAGE_ROOT+'.mp4'])

	return()

start_time=start_timer()
lets_go()
check_image_dir()

###### Flow definition #########################################################
#
# Original values
#
#maxIter = 200000 # Original Total number of time iterations.
Re      = 220.0  # Original Reynolds number.
#cx = nx/4; cy=ny/2; r=ny/9;          # Original Coordinates of the cylinder.

#
# User defined values
#
maxIter = 200000 # Total number of time iterations.
#Re = 245.65 # Reynolds number

# Lattice dimensions and populations.
nx = 520 	# x dimension
ny = 180	# y dimension
ly = ny-1.0	# length of y dimension?
q  = 9		# ?

# Coordinates of the cylinder.
cx = (nx/4) # center x position in lattice
cy = (ny/2) # center y position in lattice
r  = (ny/9) # radius of cylinder

uLB     = 0.04                       # Velocity in lattice units.
nulb    = uLB*r/Re; omega = 1.0 / (3.*nulb+0.5); # Relaxation parameter.


###### Lattice Constants #######################################################
c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.
t = 1./36. * ones(q)                                   # Lattice weights.
t[asarray([norm(ci)<1.1 for ci in c])] = 1./9.; t[0] = 4./9.
noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)]
i1 = arange(q)[asarray([ci[0]<0  for ci in c])] # Unknown on right wall.
i2 = arange(q)[asarray([ci[0]==0 for ci in c])] # Vertical middle.
i3 = arange(q)[asarray([ci[0]>0  for ci in c])] # Unknown on left wall.


###### Function Definitions ####################################################
sumpop = lambda fin: sum(fin,axis=0) # Helper function for density computation.
def equilibrium(rho,u):              # Equilibrium distribution function.
    cu   = 3.0 * dot(c,u.transpose(1,0,2))
    usqr = 3./2.*(u[0]**2+u[1]**2)
    feq = zeros((q,nx,ny))
    for i in range(q): feq[i,:,:] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return(feq)


###### Setup: cylindrical obstacle and velocity inlet with perturbation ########
#obstacle = fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (nx,ny)) # Original obstacle
obstacle1 = fromfunction(lambda x,y: (x-cx)**2+(y-(cy-27))**2<r**2, (nx,ny)) # lambda inputs: equation
obstacle2 = fromfunction(lambda x,y: (x-cx)**2+(y-(cy+27))**2<r**2, (nx,ny))
obstacle=obstacle1+obstacle2
vel = fromfunction(lambda d,x,y: (1-d)*uLB*(1.0+1e-4*sin(y/ly*2*pi)),(2,nx,ny))
feq = equilibrium(1.0,vel); fin = feq.copy()


###### Main time loop ##########################################################
for time_count in range(maxIter):
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

    if (time_count%100==0): # Visualization
        plt.clf(); plt.imshow(sqrt(u[0]**2+u[1]**2).transpose(),cmap=cm.gist_rainbow)
        plt.savefig(ROOT_DIR+IMAGE_DIR+"vel."+str(time_count/100).zfill(4)+".png")

create_mp4()
elapsed_time(start_time)
