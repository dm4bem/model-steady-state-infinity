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
To = 36       # °C

# ventilation rate (air-changes per hour)
ACH = 6             # volume/h

V_dot = L * l * H * ACH / 3600  # volumetric air flow rate
m_dot = ρ * V_dot               # mass air flow rate

nq, nθ = 17, 6  # number of flow-rates branches and of temperaure nodes

# Incidence matrix
# ================
A = np.zeros([nq, nθ])

A[0, 3] = 1
A[2, 5] = 1
A[1, 4] = 1


A[3,0] = 1
A[3,3] = -1

A[4,2] = 1
A[4,4] = -1

A[5,1] = 1
A[5,5] = -1

A[6,0] = 1
A[7,2] = 1
A[8,1] = 1
A[9,0] = 1

A[10,2] = 1
A[11,1] = 1

A[12,0] = -1
A[12,1] = 1

A[13,0] = -1
A[13,2] = 1

A[14,0] = -1
A[14,1] = 1

A[15,0] = -1
A[15,2] = 1

A[16,2] = 1
A[16,1] = -1




# Conductance matrix
# ==================
G = np.zeros(A.shape[0])


#outdoor wall 

    #CONDUCTION 
#L1 = 2 #longueur premier mur
#L2 = 2 #longueur second mur
#So = np.array([L1, L2])*H

#G[0] = 1 / (w / λ1 + 1 / hi) * So[0]
#G[1] = 1 / (w / λ2 + 1 / ho) * So[1]

    #CONVECTION 
    
    #SUR GITUHB A02 THERMAL LOADS

# G0 G1 G2  outdoor convection en vert kaki
S0 = np.array([L, L/2, L/2])*H # avec L longueur du grand mur
G[0:3] = (1/ho)*S0

# G3 G4 G5  bleu cyan, conduction and indoor convection
G[3:6] = 1 /(w / λ1 + 1/hi) * S0

#G6 G7 G8  controller 
G[6:9] = 0 #initialement null

# G9 G10 G11  Advection by ventilation 
G[9:12] = m_dot * 1000 # avec 4184 = capacité de l'air

#G12 G13 Advection by ventilation between the rooms
G[12:14] = m_dot/2 * 1000

# G14 G15 G16 indoor wall 
Si = np.array([L/2,L/2, l/2])*H
G[14:17] = 1 /(w / λ1 + 1/hi) * Si


# Vector of temperature sources
# =============================
b = np.zeros(A.shape[0])

b[0:3] = To         # cyan branches: outdoor temperature for walls
b[9:12] = To    # green branches: outdoor temperature for ventilation

# Vector of flow-rate sources
# =============================
f = np.zeros(A.shape[1])


# Indexes of outputs
# ==================
indoor_air = [0, 1, 2]   # indoor air temperature nodes
controller = range(6,9)  # controller branches


print(f"Maximum value of conductance: {max(G):.0f} W/K")
b[controller] = 19,19,19 # °C setpoint temperature of the rooms
G[controller] = 1e9             # P-controller gain

θ = np.linalg.inv(A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)

print("Without radiation")
print("θ_int :", θ[indoor_air], "°C")
print("q_ctrl:", q[controller], "W")

# Zone 2 & 4 free-running; solar rad; without ventilation
#G[[17, 19]] = 0     # controller gains for room 2 & 4

# # Solar radiation
exterior_wall = [0, 1, 2]
f[exterior_wall] = E * S0

θ = np.linalg.inv(A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)
print("With radiation")
print("θ_int:", θ[indoor_air], "°C")
print("q_ctrl:", q[controller], "W")

