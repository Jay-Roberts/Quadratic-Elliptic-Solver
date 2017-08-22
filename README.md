# Quadratic-Elliptic-Solver
A solver for quadratic nonlinear elliptic PDE's in a 2D rectangle with mixed Neumann/Dirchlet boundary conditions.


NeuDir_QuadSolver.py: 
The actual solver. It uses a central difference scheme and SOR to approximate solutions to certain elliptic PDEs on a symmetric rectangle
with one Neumann boundary and three Dirchlet. Specifics of input can be found the codes comments and a discussion of implementation is
in the write up.

NeuDirTest.py:
Essentially a wrapper for testing the solver and collecting error statistics. 

NeuDirSolver.pdf:
A write up of the implementation of the discretization and performance analysis.
