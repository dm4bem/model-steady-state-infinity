# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:36:12 2024

@author: Julie LARRUY
"""

import numpy as np
import pandas as pd 

np.set_printoptions(precision=1)

# Data
# ====
# dimensions
L, l, H, w = 10, 8, 3, 0.40  # m
# L : longueur de la pièce totale : 10m
#l : largeur de la pièce totale : 8m

# thermo-physical propertites
λ1 = 1.4   # W/(m K) wall thermal conductivity materiaux 1
λ2 = 2        # W/(m K) wall thermal conductivity materiaux 2 
ρ, c = 1.2, 1000    # kg/m3, J/(kg K) density, specific heat air
hi, ho = 8, 25      # W/(m2 K) convection coefficients in, out

# short-wave solar radiation absorbed by each wall
E = 400             # W/m2

# outdoor temperature
To = 5      # °C

# ventilation rate (air-changes per hour)
ACH = 6             # volume/h

V_dot = L * l * H * ACH / 3600  # volumetric air flow rate
m_dot = ρ * V_dot               # mass air flow rate

nq, nθ = 50, 26  # number of flow-rates branches and of temperaure nodes

# Incidence matrix
##############################################################################
##############################################################################
################################  MATRICE A ##################################
##############################################################################
##############################################################################
#================
A = np.zeros([nq, nθ])

#################################################
######## EXT --> PIECE 1, PIECE 2 ET PIECE 3
A[0, 0] = A[12, 8] = A[24, 16] = 1
A[1, 0] = A[13, 8] = A[25, 16] = -1

A[1, 1] = A[13, 9] = A[25, 17] = 1
A[2, 1] = A[14, 9] = A[26, 17] = -1

A[2, 2] = A[14, 10] = A[26,18] = 1
A[3, 2] = A[15, 10] = A[27,18] = -1

A[4,3] = A[16,11] = A[28,19] = -1
A[3,3] = A[15,11] = A[27,19] = 1

A[4, 4] = A[16,12] = A[28,20] = 1
A[6,4] = A[18,12] = A[30,20] = -1
A[5,4] = A[17,12] = A[29,20] = -1

A[6,6] = A[18,14] = A[30,22] = 1
A[10,6] = A[22,14] = A[34,22] = 1
A[11,6] = A[23,14] = A[35,22] = 1

A[5,5] = A[17,13] = A[29,21] = 1
A[7,5] = A[19,13] = A[31,21] = -1
A[9,5] = A[21,13] = A[33,21] = 1

A[8,7] = A[20,15] = A[32,23] = 1
A[9,7] = A[21,15] = A[33,23] = -1


#################################
############## ENTRE PIECE PETITE:PETITE

A[46,14 ] = 1
A[47, 24] = -1
A[48, 24] = 1
A[49, 22] = -1

#################################
############## ENTRE PIECE GRANDE ET PETITES PIECES

A[45, 6] = A[40, 6]= -1
A[45, 14] = A[40, 22]= 1
A[41, 6] = A[36, 6]= -1
A[42, 25] = A[37, 23]=1


A[43,25 ] = A[38, 23]= -1
A[44, 14] = A[39, 22]= 1

##############################################################################
##############################################################################
###################  MATRICE DE CONDUCTANCE ##################################
##############################################################################
##############################################################################

# Conductance matrix
# ==================
G = np.zeros(A.shape[0])

S0 = np.array([L, L/2, L/2])*H # avec L longueur du grand mur


######################################### CONDUCTION OUTDOOR 

# G1 G2 G3 G4 
G[1:5] =  λ1 * S0[0]/w ############################################# ENLEVER LONGUEUR FENETRE

# G13 G14 G14 G16
G[13:17] =  λ1 * S0[1]/w

# G25 G26 G27 G28
G[25:29] =  λ1 * S0[2]/w

######################################## OUTDOOR CONVECTION 
#G0 G6 G7 
G[0] = G[6] = G[7] = ho * S0[0]

#G12 G18 G19 
G[12] = G[18] = G[19] = ho * S0[1]

#G23 G30 G31 
G[23] = G[30] = G[31] = ho * S0[2]

## G10 G22 G34 #############################AIR FLOW 
G[10] = G[22] = G[34] = m_dot * 4184 

## G11 G23 G35 ###########################CONTROLLER 
G[11] = G[23] = G[35] = 0

#########################################INDOOR CONVECTION 
#G36  G39 G41 G44
G[36] = G[39] = G[41] = G[44] = hi * S0[1]

#G49
G[49] = hi * S0[1]

#46 
G[46] = hi * S0[2]

#########################################INDOOR CONDUCTION  
# G37 G38 G42 G43 G47 G48 
G[37] = G[38] =  λ1 * S0[1]/w
G[42] = G[43] =  λ1 * S0[1]/w
G[47] = G[48] =  λ1 * S0[1]/w

####################################### ADVECTANCE GRANDE ---> PETITE
# G40 G45
G[40] = G[45] =  m_dot/2 * 4184


####################################### FENETRES
Surf_window = S0[0] / 5

#CONVECTION G8 G20 G32
G[8] = G[20] = G[32] = ho * Surf_window

#CONDUCTION G9 G21 G33
G[9] = G[21] = G[33] =  λ1 * Surf_window/w

###################################### LONG WAVE RADIATION 

Fact = 1/5 #facteur de forme 

# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass
σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant


Tm = To + 273   # K, mean temp for radiative exchange

GLW1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * S0[0]
GLW12 = 4 * σ * Tm**3 * Fact * S0[0]
GLW13 = 4 * σ * Tm**3 * Fact * S0[1]
GLW2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * Surf_window


#G5 G17 G29
Somme = 1 / (( 1/GLW1) + ( 1/GLW12) + ( 1/GLW2))
Somme1 = 1 / (( 1/GLW1) + ( 1/GLW13) + ( 1/GLW2))
G[5] = Somme 
G[17] = G[29] = Somme1

##############################################################################
##############################################################################
###################  Vector of temperature sources############################
##############################################################################
##############################################################################


b = np.zeros(A.shape[0])

b[[0, 12, 24]] = To         # cyan branches: outdoor temperature for walls
b[[10,22,34]] = To    # green branches: outdoor temperature for ventilation

# Vector of flow-rate sources
# =============================
f = np.zeros(A.shape[1])


# Indexes of outputs
# ==================
indoor_air = [6, 14, 22]   # indoor air temperature nodes
controller = [11, 23, 35]  # controller branches


print(f"Maximum value of conductance: {max(G):.0f} W/K")
b[controller] = 19,19,19 # °C setpoint temperature of the rooms
G[controller] = 1e9             # P-controller gain

θ = np.linalg.inv(A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)

print("Without radiation")
print("θ_int :", θ[indoor_air], "°C")
print("q_ctrl:", q[controller], "W")


