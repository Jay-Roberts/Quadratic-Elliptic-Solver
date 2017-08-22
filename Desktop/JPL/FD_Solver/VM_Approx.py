#------------------------#------------------------#------------------------
#
#       Vainshtein Mechanism Solver for Two Body Problem
#
#------------------------#------------------------#------------------------

import numpy as np
import NeuDir_QuadSolver as sol
import Special_Functions_VM as spf
import matplotlib.pyplot as plt
import time

start_time = time.time()


#--------------------------
#   Set Constants
#--------------------------

d_scale = 384402.0*(10**3) #distance from earth to moon in m
d = 1.0
rc = pow(10,8)*d
rA,rB = 6.371*(10**6)/d_scale,1.737*(10**6)/d_scale

M_scale = 7.35*(10**22) # Mass of moon in kg
MA,MB = 5.972*(10**24)/M_scale,1
#MA,MB = 0,1
G = 6.67408*pow(10,-11)*d_scale**(-3)*M_scale

nodes = 30

L = 200*d

print L



Space = [G,rc]
Radii = [rA,rB]
Mass = [MA,MB]
center = -d

Constants = [Space, Mass, Radii, center ]

alpha = .1

# Where the box is in rho-xi space is different. This is got by solving L = rho + alpha rho**3
Box_size = 15.8


def Lin_term(rho,xi):
    coord = [rho,xi]

    Lin_ans = spf.Lin_rhoxi(coord,alpha,Constants)
    return Lin_ans

def Force_term(rho,xi):
    coord = [rho,xi]

    F = spf.Force_rhoxi(coord,alpha,Constants)
    return F

def Mult_term(rho,xi):
    coord = [rho,xi]

    M = spf.Mult_rhoxi(coord,alpha,Constants)
    return M

U = sol.Non_lin_solv(Lin_term,Mult_term,Force_term,nodes,Box_size)

Sol_grid = U[0]
print Sol_grid.shape
#np.savetxt('/home/jay/Desktop/JPL/Vainshtein/FD_Solver/SolRun1.csv',Sol_grid, delimiter=',')
rho_grid, xi_grid = U[1]


M1,M2 = len(rho_grid), len(xi_grid)

Sol_values = np.copy(Sol_grid)
Sol_values.shape = len(rho_grid)*len(xi_grid)

import pandas as pd

Values = pd.DataFrame(Sol_values)

print Values.describe()


#Values.hist(bins = 10**3)
#plt.show()

Sol_values_uniq = list(set(Sol_values))
Sol_values_ix = np.argsort(Sol_values)
Levels = sorted(Sol_values_uniq)


#print Levels

r_grid, z_grid = spf.back_to_cylin(rho_grid,alpha), spf.back_to_cylin(xi_grid,alpha)




print(time.time() -start_time)

plt.figure(1)
plt.contour(xi_grid,rho_grid,Sol_grid)
plt.colorbar()

plt.figure(2)
plt.title('z-r grid')
plt.contourf(z_grid,r_grid,Sol_grid)
plt.colorbar()
plt.show()
