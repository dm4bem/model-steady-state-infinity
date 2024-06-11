# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:19:37 2024

@author: Julie LARRUY
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:36:12 2024

@author: Julie LARRUY
"""

import numpy as np
import pandas as pd 
import dm4bem
import matplotlib.pyplot as plt


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
G[11] = G[23] = G[35] = 10**(6)

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
###################  Vector of temperature sources ###########################
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
Tsp = 19
b[controller] = Tsp,Tsp,Tsp # °C setpoint temperature of the rooms
G[controller] = 0           # P-controller gain

θ = np.linalg.inv(A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)

print("Without radiation")
print("θ_int :", θ[indoor_air], "°C")
print("q_ctrl:", q[controller], "W")

##############################################################################
##############################################################################
############################  Matrice de Capacité ############################
##############################################################################
##############################################################################

#création matrice C pour le step response

#concrete : 
Density_concrete = 2300              # kg/m³
Specific_heat_concrete = 880            # J/(kg⋅K)

#glass : 
Density_glass = 2400               # kg/m³
Specific_heat_glass = 1210            # J/(kg⋅K)

C = np.zeros(nθ)

# Wall 1 

L1 = L - 2*w
Surf_window = S0[0] / 5
Surf_wall_1 = L1 * H - Surf_window

C[1] = C[3] = Density_concrete * Specific_heat_concrete * w * Surf_wall_1

#wall 2 et 3 

L2 = L1 /2
Surf_wall_2 = (L1 * H - 2 * Surf_window)/2
C[9] = C[11] = C[17] = C[19]= Density_concrete  * Specific_heat_concrete * w * Surf_wall_2

#Windows

C[7] = C[15] = C[23] = Density_glass * Specific_heat_glass * w * Surf_window

#Air
l1= l - 2*w
C[14] = C[22] =  ρ * c * L1 *H * l1 /2
C[6] = ρ * c * L1 *H * l1


# indoor big wall 

Surf_wall_3 = (l - 3*w)/2
C[23] = C[25] = Density_concrete  * Specific_heat_concrete * w * Surf_wall_2
C[24] = Density_concrete  * Specific_heat_concrete * w * Surf_wall_3

##############################################################################
##############################################################################
###################  Vector of temperature sources ###########################
##############################################################################
##############################################################################

bss = np.zeros(A.shape[0])

bss[0] = bss[8] = bss[10] = To #première pièce
bss[12] = bss[20] = bss[23] = To #deuxième pièce
bss[24] = bss[32] = bss[34] = To #troisième pièce
bss[11] = bss[23] = bss[35] = Tsp

############################################################## u = pd################
##############################################################################
###################  Vector of heat flow sources #############################
##############################################################################
##############################################################################

fss = np.zeros(nθ)

#Outdoor surface
fss[0] = α_wSW * Surf_wall_1 * E
fss[8] = α_wSW * Surf_wall_2 * E 
fss[16] = α_wSW * Surf_wall_2 * E 

#Windows

fss[7] = α_gSW * Surf_window * E 
fss[15] = α_gSW * Surf_window * E 
fss[23] = α_gSW * Surf_window * E 

#outdoor surface

##############################################################################
##############################################################################
###################  Vector output and step response    ######################
##############################################################################
##############################################################################

y = np.zeros(nθ)

y[0] = y[6] = y[22] = 1 #noeuds controllés 


q_list = [f'q{i}' for i in range (nq)]
θ_list = [f'θ{i}' for i in range (nθ)]

A = pd.DataFrame(A, index=q_list, columns=θ_list)
G = pd.Series(G, index=q_list)
C = pd.Series(C, index=θ_list)
b = pd.Series(bss, index=q_list)
f = pd.Series(fss, index=θ_list)
y = pd.Series(y, index=θ_list)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

A = TC['A']
G = TC['G']
diag_G = pd.DataFrame(np.diag(G), index=G.index, columns=G.index)

θss = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)
print(f'θss = {np.around(θss, 2)} °C')

bT = np.array([5, 5, 5,19,5,5,19,5,5,5,19])     # [To, To, To, Tisp]
fQ = np.array([2160, 912, 780,912,780,912])         # [Φo, Φi, Qa, Φa]
uss = np.hstack([bT, fQ])           # input vector for state space
print(f'uss = {uss}')

inv_As = pd.DataFrame(np.linalg.inv(As),columns=As.index, index=As.index)
yssQ = (-Cs @ inv_As @ Bs + Ds) @ uss

yssQ = float(yssQ.values[0])
print(f'yssQ = {yssQ:.2f} °C')

print(f'Error between DAE and state-space: {abs(θss[6] - yssQ):.2e} °C')

# Eigenvalues analysis
λ = np.linalg.eig(As)[0] 

# time step
Δtmax = 2 * min(-1. / λ)    # max time step for stability of Euler explicit
dm4bem.print_rounded_time('Δtmax', Δtmax)

imposed_time_step = False
Δt = 498    # s, imposed time step

if imposed_time_step:
    dt = Δt
else:
    dt = dm4bem.round_time(Δtmax)
dm4bem.print_rounded_time('dt', dt)

if dt < 10:
    raise ValueError("Time step is too small. Stopping the script.")

# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)

# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                           periods=n, freq=f"{int(dt)}S")

To = 5 * np.ones(n)        # outdoor temperature
Ti_sp = 19 * np.ones(n)     # indoor temperature set point
Φa = 0 * np.ones(n)         # solar radiation absorbed by the glass
Qa = Φo = Φi = Φa           # auxiliary heat sources and solar radiation

data = {'To': To, 'Ti_sp': Ti_sp, 'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa}
input_data_set = pd.DataFrame(data, index=time)

data = {'To': To, 'Ti_sp': Ti_sp, 'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa}
b = pd.Series (['To', 0,0,0,0,0,0,0,'To',0,'To','Ti_sp','To',0,0,0,0,0,0,0,'To',0,0,'Ti_sp','To',0,0,0,0,0,0,0,'To',0,'To','Ti_sp',0,0,0,0,0,0,0,0,0,0,0,0,0,0], index = q_list)
input_data_set = pd.DataFrame(data, index=time)
F = pd.Series (['Φo',0,0,0,0,0,0,'Φa','Φi',0,0,0,0,0,0,'Φa','Φi',0,0,0,0,0,0,'Φa',0,0],index = θ_list )

TC1 = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": F,
      "y": y}

[As1, Bs1, Cs1, Ds1, us1] = dm4bem.tc2ss(TC1)
 
# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us1, input_data_set)

# Initial conditions
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ0 = 0.0                    # initial temperatures
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])
        
# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)

ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {Δtmax:.0f} s')
plt.show()

