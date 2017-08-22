import numpy as np
from scipy.sparse import linalg as sla
import NeuDir_QuadSolver as sol
import time



#---------------------------------------------------------------
#
# FUNCTIONS TO EDIT AND CHOOSE FROM
#
#---------------------------------------------------------------

#
# Define a function to test.
# Domain [0,d ] x [-d,d]
# Satisfies ux= 0 on x = 0 and Dirchlet on other sides.
#

def test_function(x,y):
    

    f, g = x**2 -1,  y**2 - 1
    
    lamx, lamy = np.pi/float(2), np.pi
    
    Cx,Sx,Cy,Sy = np.cos(lamx*x),np.sin(lamx*x),np.cos(lamy*y),np.sin(lamy*y)

    u = f*g + Cx*Sy

    return u

#
# Compute the derivatives to be used to test against.
# Must output (ux,uy,uxx,uyy,uxy) at the point
#
def Deriv_test(x,y):

    f, g = x**2 -1,  y**2 - 1
    
    lamx, lamy = np.pi/float(2), np.pi
    
    Cx,Sx,Cy,Sy = np.cos(lamx*x),np.sin(lamx*x),np.cos(lamy*y),np.sin(lamy*y)

    ux,uy = 2*x*g - lamx*Sx*Sy, 2*y*f + lamy*Cx*Cy
    uxx,uyy = 2*g - (lamx**2)*Cx*Sy, 2*f - (lamy**2)*Cx*Sy 
    uxy = 4*y*x - lamy*lamx*Sx*Cy

    D2u = [ux,uy,uxx,uyy,uxy]
    return np.array(D2u)

#
# Define the Linear Operator
# Should return a 5 x 1 array at a point x,y
# Make sure at least one 2nd order term has a non-vanishing coefficient or well posedness may be violated

def Lin_Op(x,y):
    

    L = [1.0,1.0,2-x,2-y,1]

    return np.array(L)



# Define the Muliplier for the Bilinear form
# Take in a point and return 5 x 5 array
#
def Mult_Op(x,y):
    

    M = np.array([
        [1+x**2,  0,  1  ,  0,  0],
        [ 0    , 1 , 0   ,  0,  0],
        [1     , 0 , x+y ,  0,  0],
        [1     , 0 , 1   ,  1,  0],
        [0     , 0 , 1   ,  0,  1]    ])
    
    #M = np.identity(5)
    return M
#-----------------------------------------------------------------------------------------
# 
# Set the parameters
#
#-----------------------------------------------------------------------------------------

d = 1
N_big = 50
skip = 15

# To Test the Linear Solver

def Line_Force_test(x,y):

    L = Lin_Op(x,y)
    D2u = Deriv_test(x,y)
    return np.dot(L,D2u)


# To Test the Full Solver

def Force_Op(x,y):
    D2u = Deriv_test(x,y)
    L = Lin_Op(x,y)
    M = Mult_Op(x,y)

    Lin = np.dot(L,D2u)
    Quad = np.dot(D2u,np.dot(M,D2u))

    F = Lin+Quad
    return F


Lin_error_bucket = []
Quad_error_bucket = []
Solver_error_bucket =[]

nodes_bucket = []
run_times_bucket = []
memory_bucket = []


import pandas as pd


for iLin in range(10,N_big,skip):
#for iLin in range(70,71):

    print('Number of nodes: ', iLin)
    
    start_time = time.time()
    nodes = iLin
    nodes_bucket.append(nodes)

    Lin_Sol_approx, Grid, dims = sol.Lin_Ellip_ND(Lin_Op,Line_Force_test,nodes,d)

    memory = sol.Lin_Ellip_ND(Lin_Op,Force_Op,nodes,d,solve = False)[0].nbytes
    memory_bucket.append(memory)

    Solver_approx = sol.Non_lin_solv(Lin_Op,Mult_Op,Force_Op,nodes,1, test = True, Ana = test_function)[0]


    x_grid,y_grid = Grid
    N,M1,M2 = dims
    
    # Get function value to test solvers

    Ana_Grid = np.zeros((M1,M2), dtype=float)

    
    # Makes grid of test values
    for ix in range(M1):
        for iy in range(M2):
            x,y = x_grid[ix], y_grid[iy]

            Ana_Grid[ix,iy] = test_function(x,y)
    

    Lin_Dif = Lin_Sol_approx - Ana_Grid
    Lin_error = np.max(abs(Lin_Dif))
    Lin_error_bucket.append(Lin_error)

    Sol_dif = Solver_approx - Ana_Grid
    Sol_error = np.max(abs(Sol_dif))
    Solver_error_bucket.append(Sol_error)

    run_time  = time.time()-start_time
    run_times_bucket.append(run_time)



    
    #print(stats.head())
'''
    stats = pd.DataFrame({'Nodes': nodes_bucket, 'Linear Error': Lin_error_bucket, 'Solver Error': Solver_error_bucket, 'Run Time': run_times_bucket})
    stats.to_csv('/home/jay/Desktop/JPL/NeuDirWriteUp/Stats.csv',columns=['Nodes','Linear Error','Solver Error', 'Run Time'] , index=False)
'''
    
import matplotlib.pyplot as plt


#Plot Linear Error and Last Linear Approximation

# A nice way to get the stats. Not necessary for the rest of error computation


Log_nodes = np.log(nodes_bucket)

nodes_spot = np.mean(Log_nodes) - np.std(Log_nodes)
# Deal with linear error
Lin_log_error = np.log(Lin_error_bucket)


Lin_linefit = np.polyfit(Log_nodes,Lin_log_error,1)
Lin_fit = np.poly1d(Lin_linefit)

Lin_rate = Lin_linefit[0]

Lin_spot = np.mean(Lin_log_error) - np.std(Lin_log_error)

# Deal with Solver error
Sol_log_error = np.log(Solver_error_bucket)


Sol_linefit = np.polyfit(Log_nodes,Sol_log_error,1)
Sol_fit = np.poly1d(Sol_linefit)

Sol_rate = Sol_linefit[0]

Sol_spot = np.mean(Sol_log_error) - np.std(Sol_log_error)

# Deal with time

log_run = np.log(run_times_bucket)
run_linefit = np.polyfit(Log_nodes,log_run,1)
run_fit = np.poly1d(run_linefit)

run_rate = run_linefit[0]

run_spot = np.mean(log_run) + np.std(log_run)

plt.figure(1)

plt.subplot(121)
plt.title('Linear Solver Error')

plt.scatter(Log_nodes,Lin_log_error)
plt.plot(Log_nodes,Lin_fit(Log_nodes), '-')
plt.xlabel('Log Steps')
plt.ylabel('Log Absolute Error')

plt.text(nodes_spot,Lin_spot,'Slope = %f' % Lin_rate)

plt.subplot(122)
plt.title('Solver Error')

plt.scatter(Log_nodes,Sol_log_error)
plt.plot(Log_nodes,Sol_fit(Log_nodes), '-')
plt.xlabel('Log Steps')
plt.ylabel('Log Absolute Error')
plt.text(nodes_spot,Sol_spot,'Slope = %f' % Sol_rate)


plt.figure(2)
plt.title('Time v Nodes')
plt.scatter(Log_nodes,log_run)
plt.plot(Log_nodes, run_fit(Log_nodes), '-')
plt.xlabel('Number of Nodes')
plt.ylabel('Time (sec)')

plt.text(nodes_spot,run_spot, 'Slope = %f' % run_rate)
plt.show()










