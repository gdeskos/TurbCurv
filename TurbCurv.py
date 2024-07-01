import numpy as np
import scipy.linalg as la

def D_matrices(N, dz):
    """ Generate finite difference matrix for the second derivative """
    D = np.zeros((N, N))
    D2 = np.zeros((N, N))
    D3 = np.zeros((N, N))
    for i in range(1, N-1):
        D[i,i-1] = -0.5/dz
        D[i,i+1] = 0.5/dz

        D2[i, i-1] = 1 / dz**2
        D2[i, i] = -2 / dz**2
        D2[i, i+1] = 1 / dz**2

    for i in range(2, N-2):
        D3[i, i-2] = -0.5 / dz**3
        D3[i, i-1] = 1/ dz**3
        D3[i,i]=0.
        D3[i, i+1] = 1 / dz**3
        D3[i, i+2] = 0.5 / dz**3

    return D,D2,D3

def apply_boundary_conditions(A, b, z_bc,dz):
    """ Apply boundary conditions to the matrix and RHS vector """

    # Apply dirichlet boundary conditions first
    A[0, :] = 0
    A[0, 0] = 1
    b[0] = z_bc[0]

    A[-1, :] = 0
    A[-1,-1] = 1
    b[-1] = z_bc[2]

    # Apply Neumann boundary conditions next
    A[1, :] = 0
    A[1, 0] =-1/dz
    A[1, 1] = 1/dz
    b[1] = z_bc[1]

    A[-2, :] = 0
    A[-2, -2] = -1/dz
    A[-2, -1] = 1/dz
    b[-2] = z_bc[3]

    return A, b

def orr_sommerfeld(N, zi, nu, nuT, nuT_prime, k, U, c, U_double_prime, f,z_bc):
    z = np.linspace(0, zi, N)
    dz = z[1] - z[0]

    D, D2, D3 = D_matrices(N, dz)
    I = np.eye(N)

    # Left-hand side
    A = -(nu+nuT)/(1j*k)*(D2 - k**2 * I)@(D2 - k**2 * I)+(U-c) * (D2 - k**2 * I) - np.diag(U_double_prime) - 2/(1j*k)* nuT_prime*(D2-k**2*I)@D
 

    # Add the inhomogeneous term f(z) to the right-hand side
    b = f

    # Apply boundary conditions
    A, b = apply_boundary_conditions(A, b, z_bc,dz)
    print(np.shape(A),np.shape(b))
    # Solve the linear system
    w = la.solve(A, b)

    return z, w

