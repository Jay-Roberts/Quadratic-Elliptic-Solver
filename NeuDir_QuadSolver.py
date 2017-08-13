import numpy as np
from scipy.sparse import linalg as sla

#
#
# This solves quadratic 2D pdes of the form
#
#       L D2u = F + D2uT M D2u      D2uT = (ux,uy,uxx,uyy,uxy)
#
# on a domain [0,d] x [-d,d]. Where F:R2 -> R, L: R2 -> R5, and M: R2 -> R(5 x 5)
# Neumann boundary conditions are imposed on x = 0 and the rest are dirchlet
#
#
#

#
# Defining the linear solver
#


# This is a genearl linear SOLVER for LD2u = F.
# Importantly it explicitly creates the matrix corresponding to the discretization
# which it can return or it can return the solution of this equation.


###Inputs
###
# Linear: A function with two inputs and five outputs. The L from above
# Force: A function  with two inputs and one output. The F from above
# nodes: A number. This is the number of discretization points for the [0,d] axis. A grix of size (2 nodes + 1 )(nodes+1) x (2 nodes + 1 )(nodes+1) is created
# d: a number. This  is the size of the x axis.
###
### Returns
# solve = True, returns the solution on the grid pieces as a Grid. U[i,j] = u(xi,yj)
# solve = False, returns the Discretized A and the grid pieces

def Lin_Ellip_ND(Linear,Force,nodes,d,solve = True):
    N0 = nodes
    M1,M2 = N0+1,2*N0+1
    N = M1*M2
    h = d/float(N0)

    #Make a dictionary for lexicographic ordering back to regular ordering

    lex_order = []
    for iii in range(N):
        lex_order.append(iii)
    lex_order = np.array(lex_order)
    lex_order.shape = (M1,M2)
    
    #print lex_order

    #Make the axis values
    #To easily enforce Neuman we shift the x coordinates

    x_grid,y_grid = np.zeros(M1), np.zeros(M2)
    for ix in range(M1):
        point = (ix+.5)*h
        x_grid[ix] = point

    for iy in range(M2):
        point = (iy - N0)*h
        y_grid[iy] = point
    
    #Make the discretized matrix
    
    Lin = np.zeros((N,N))
    for iA in range(M1):
        
        #Deal with the Neumann x-boundary i = 0  after
        if iA == 0:
            for jA in range(1,M2-1):
                L = Linear(x_grid[0], y_grid[jA])
                ix = lex_order[0,jA]

                row = np.zeros((M1,M2))

                #print rowprint N
                #print lex_order

                P0j = np.array([
                    [0,-1/float(2*h),0,0,1/float(2*h),0],
                    [-1/float(2*h),0,1/float(2*h),0,0,0],
                    [0,-1/float(h**2),0,0,1/float(h**2),0],
                    [1/float(h**2),-2/float(h**2),1/float(h**2),0,0,0],
                    [1/float(4*h**2),0,-1/float(4*h**2),-1/float(4*h**2),0,1/float(4*h**2)]
                ])

                Lij = np.dot(L,P0j)
                Lij.shape = (2,3)

                row[0:2,jA-1:jA+2] = Lij
                
                
                row.shape = (1,N)

                Lin[ix] = row[0]
                #print row[0]

            #Bottom corner i = 0 j = 0
            bcorner = np.zeros((M1,M2))
            bcorner[0,1] = L[1]/float(2*h)+L[3]/float(h**2)-L[4]/float(4*h**2)
            bcorner[1,1] = L[4]/float(4*h**2)

            bcorner.shape = (1,N)
            Lin[lex_order[0,0]] = bcorner[0]

            #Top corner i = 0 j = 2N0 = M2-1
            tcorner = np.zeros((M1,M2))
            tcorner[0,M2-2] = -L[1]/float(2*h)+L[3]/float(h**2)+L[4]/float(4*h**2)
            tcorner[1,M2-2] = -L[4]/float(4*h**2)

            tcorner.shape = (1,N)
            Lin[lex_order[0,M2-1]] = tcorner[0]
        #Deal with the Dirchlet x-boundary i = N0 = M1-1  after
        elif iA == (M1-1):
            #print lex_order
            #print iA
            for jA in range(1,M2-1):
                L = Linear(x_grid[M1-1], y_grid[jA])
                ix = lex_order[M1-1,jA]

                row = np.zeros((M1,M2))
                row[M1-1,jA-1] = L[4]/float(4*h**2)
                row[M1-1,jA] = -L[0]/float(2*h) + L[2]/float(h**2)
                row[M1-1,jA+1] = -L[4]/float(4*h**2)

     
                row.shape = (1,N)
                
                Lin[ix] = row[0]
            
            #Bottom corner iA = N0 = M1-1 and j = 0
            bcorner = np.zeros((M1,M2))
            bcorner[M1-2,1] = -L[4]/float(4*h**2)
            

            bcorner.shape = (1,N)
            Lin[lex_order[M1-1,0]] = bcorner[0]

            #Top corner iA = N0 = M1-1 and j = 2N0 = M2-1
            tcorner = np.zeros((M1,M2))
            tcorner[M1-2,M2-2] = L[4]/float(4*h**2)
            
            #print lex_order
            #print tcorner

            tcorner.shape = (1,N)
            #print tcorner
            Lin[lex_order[M1-1,M2-1]] = tcorner[0]
            #print lex_order[M1-1,M2-1], N
        #Take care of interior rows
        else:
            #Start with the interior of the interior of rows and add the caps j ==0 j ==2N0 = M2 -1 at the end
            for jA in range(1,M2-1):
                L = Linear(x_grid[iA], y_grid[jA])
                ix = lex_order[iA,jA]

                row = np.zeros((M1,M2))

                Pij = np.array([
                    [0,-1/float(2*h),0,0,0,0,0,1/float(2*h),0],
                    [0,0,0,-1/float(2*h),0,1/float(2*h),0,0,0],
                    [0,1/float(h**2),0,0,-2/float(h**2),0,0,1/float(h**2),0],
                    [0,0,0,1/float(h**2),-2/float(h**2),1/float(h**2),0,0,0],
                    [1/float(4*h**2),0,-1/float(4*h**2),0,0,0,-1/float(4*h**2),0,1/float(4*h**2)]
                ])
                Lij = np.dot(L,Pij)
                Lij.shape = (3,3)

                row[iA-1:iA+2,jA-1:jA+2] = Lij

                row.shape = (1,N)

                Lin[ix] = row[0]
            #Do the edges Dirchlet y-boundary condition j = 0 (left edge)

            ledge = np.zeros((M1,M2))
            ledge[iA-1,1] = -L[4]/float(4*h**2)
            ledge[iA,1] = L[1]/float(2*h) + L[3]/float(h**2)
            ledge[iA+1,1] = L[4]/float(4*h**2)

            ledge.shape = (1,N)
            Lin[lex_order[iA,0]] = ledge[0]

            #Do the edges Dirchlet y-boundary condition j = 2N0 = M2-1 (right edge)

            redge = np.zeros((M1,M2))
            redge[iA-1,M2-2] = L[4]/float(4*h**2)
            redge[iA,M2-2] = -L[1]/float(2*h) + L[3]/float(h**2)
            redge[iA+1,M2-2] = -L[4]/float(4*h**2)

            redge.shape = (1,N)
            
            Lin[lex_order[iA,M2-1]] = redge[0]



    #
    #Solve the linear alg System Lin U = F
    #

    #Make the forcing array
    if solve:
        F = np.zeros((M1,M2))
        for iF in range(M1):
            xF = x_grid[iF]
            for jF in range(M2):
                yF = y_grid[jF]
                F[iF,jF] = Force(xF,yF)
        
        F.shape = (1,N)
        F = F[0]


       
        
        lu = sla.splu(Lin)
        U2 = lu.solve(F)
        U2.shape = (M1,M2)
        #U = np.linalg.solve(Lin,F)
        #U.shape = (M1,M2)
        Grid = [x_grid,y_grid]

        dims = [N,M1,M2]
        return [U2,Grid,dims]
    
    else:
        return [Lin,[x_grid,y_grid]]

#
# Quadratic Discretizer
#

#
# This provides a discretization of D2uT M D2u ASSUMING WE HAVE U
# unlike the linear solver it does not compute the discretization of M
#



### Inputs: 
# U_grid: an M1 x M2 array of values of the function being approxiamted
### at the grid points ordered according to lexicographic order by indicy.
### This order is equivilent to np.meshgrid order of 'indicies = 'ij' or matrix ordering.

# Grid: is a tuple with the x_grid and y_grid. This is gotten later as the output of Lin_Ellip_ND
# Mult: The quadratic multiplier M, a function with two inputs and a 5 x 5 matrix output
# test: Used for testing. If False an approximation is computed assuming 0 boundary conditions
### if True must supply the function to test against. 
# Ana: Analytical test function. The approximation does not require any boundary assumptions. However,
### when used in the full solver correct boundary conditions are assumed.


def Quad_disc(U_grid, Grid, Mult,  test = False, Ana = None):
    x0_grid,y0_grid = Grid

    M1,M2 = len(x0_grid),len(y0_grid)
    N = M1*M2

    d,N0 = y0_grid[-1], len(x0_grid)-1
    h = d/float(N0)
    

    U = np.zeros((M1+2,M2+2))
    U[1:M1+1,1:M2+1] = U_grid[:M1,:M2]

    if test:
        x_big,y_big = np.zeros(M1+2), np.zeros(M2+2)

        x_big[1:M1+1],y_big[1:M2+1] = x0_grid,y0_grid

        x_big[0],y_big[0] = x0_grid[0] - h, y0_grid[0] - h
        x_big[-1], y_big[-1] = x0_grid[-1] + h, y0_grid[-1] + h
        
        points = []
        for iii in range(M1):
            xxx = x0_grid[iii]
            for jjj in range(M2):
                yyy = y0_grid[jjj]
                points.append([xxx,yyy])
        
        for ix in range(len(x_big)):
            x = x_big[ix]
            for iy in range(len(y_big)):
                y = y_big[iy]
                U[ix,iy] = Ana(x,y)
        

    else:
        U[0,1:M2+1] = U_grid[0,:M2]
    #This already has the dirchlet condtions so we only need to enforce Neumann


    # To deal with the boundary condtions quickly we pad U with zeros on the dirchlet positions and a reflection on the neuman boundary
 
    #Make the P operator that takes the neighbors of Uij and returns the discrtized D2Uij


    #Make a function that takes in ij and gives back the neighbros subject to boundary conditions
    P = np.array([
         [0,-1/float(2*h),0,0,0,0,0,1/float(2*h),0],
         [0,0,0,-1/float(2*h),0,1/float(2*h),0,0,0],
         [0,1/float(h**2),0,0,-2/float(h**2),0,0,1/float(h**2),0],
         [0,0,0,1/float(h**2),-2/float(h**2),1/float(h**2),0,0,0],
         [1/float(4*h**2),0,-1/float(4*h**2),0,0,0,-1/float(4*h**2),0,1/float(4*h**2)]
                ])
    
    
    #Initialize the quadradic terms on the grid. Can unfold later
    Quad = np.zeros((M1,M2))
    
    
    # The discretization requires the neighbors. Use the boundary condition to get a 9 x 9 grid of neighbors
    





    for ix in range(M1):
        for iy in range(M2):
            xij = np.zeros((3,3))
            ii,jj = ix+1,iy+1

            xx,yy = [x0_grid[ix],y0_grid[iy]]


            M = Mult(xx,yy)
        
            xij[:3,:3] = U[ii-1:ii+2,jj-1:jj+2]

            

            xij.shape = (9,1)
            Pij = np.dot(P,xij)
            PijT = np.transpose(Pij)
            Mij = np.dot(M,Pij)
            nodeij = np.dot(PijT,Mij)
            
            Quad[ix,iy] = nodeij
    #Quad.shape = (1,N)
        
    
    return Quad

#
# Overall Solver
#

# Puts together the above routines through an SOR approxiamtion of 
# the discretized algebraic system.


### Inputs
## Lin: Same as in Lin_Ellip_ND
## Mult: Same as in Quad_disc
## Force: Same as in Lin_Ellip_ND
## d: Same as in Lin_Ellip_ND
## SOR: List [eSOR, omg, tSOR]
### eSOR: the relative tolerance of SOR
### omg : The omega in SOR algorithm
### tSOR: max number of iterations of SOR step


def Non_lin_solv(Lin,Mult,Force,nodes,d, SOR = [ 10**(-8),.5, 1000], test = False, Ana = None):
    
    eSOR, omg, tSOR = SOR
    A, Grid = Lin_Ellip_ND(Lin,Force,nodes,d, solve = False)
    
   # print A
    x_grid,y_grid = Grid
    M1,M2 = len(x_grid), len(y_grid)
    N = M1*M2

    F = np.zeros((M1,M2))
    for ix in range(M1):
        for iy in range(M2):
            gx,gy = x_grid[ix], y_grid[iy]

            F[ix,iy] = Force(gx,gy)
    
    F.shape = N
    

    #We initialize the guess with u = 0

    U = np.zeros(N)

    err = 1
    count = 0


    #SOR stage
    while err>eSOR and count<tSOR:
        U.shape = (M1,M2)
        f_u = Quad_disc(U,[x_grid,y_grid],Mult, test = test, Ana = Ana)
        f_u.shape = N
        f_u = -f_u


        B = f_u + F

        
        lu = sla.splu(A)
        U_star = lu.solve(B)

        U.shape = N
        U = omg*U_star + (1-omg)* U 

        #print 'This is U: ', U
        

        count = count + 1
        #print 'Max Min'
        #print np.max(U), np.min(U)

        err_top = np.linalg.norm(np.dot(A,U)-B)
        err_bot = np.linalg.norm(B)
        err = err_top/err_bot
    
    U.shape = (M1,M2)
    #print err, count
    
    return [U,Grid,[M1,M2,N]]

#