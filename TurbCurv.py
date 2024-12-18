import numpy as np
import scipy.linalg as la

def D_matrices(N, dz):
    """ Generate finite difference matrix for the second derivative """
    D  = np.zeros((N, N))
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

def D_matrices_non_uniform(z):
    """ Generate finite difference matrices for the first, second, and third derivatives on a non-uniform grid """
    N = len(z)
    D = np.zeros((N, N))
    D2 = np.zeros((N, N))
    D3 = np.zeros((N, N))
    
    dz = np.diff(z)

    # First derivative matrix D
    for i in range(1, N-1):
        dz_p = z[i+1]-z[i]
        dz_n = z[i]-z[i-1]
        
        D[i, i-1] = -1 / (dz_p+dz_n)
        D[i, i+1] =  1 / (dz_p+dz_n)

    # Second derivative matrix D2
    for i in range(1, N-1):
        dz_p = z[i+1]-z[i]
        dz_n = z[i]-z[i-1]
        
        D2[i, i-1] =  2 / (dz_n * (dz_n + dz_p))
        D2[i, i]   = -2 / (dz_n * dz_p)
        D2[i, i+1] =  2 / (dz_p * (dz_n + dz_p))

    # Third derivative matrix D3
    for i in range(2, N-2):
        dz_p = z[i+1]-z[i]
        dz_n =   z[i]-z[i-1]
        dz_p_2 = z[i+2]-z[i]
        dz_n_2 =   z[i]-z[i-2]
        
    return D, D2, D3


def apply_boundary_conditions(A, b, z_bc,dz):
    """ Apply boundary conditions to the matrix and RHS vector """
    b = b.astype(complex)
    
    print(A.dtype)
    print(b.dtype)
    
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
    A[-2, -1] =  1/dz
    b[-2] = z_bc[3]

    return A, b

def apply_boundary_conditions_non_uniform(A, b, z_bc, z):
    """ Apply boundary conditions to the matrix and RHS vector for a non-uniform grid """
    b = b.astype(complex)
    
    # Apply Dirichlet boundary conditions (unchanged)
    A[0, :] = 0
    A[0, 0] = 1
    b[0] = z_bc[0]

    A[-1, :] = 0
    A[-1,-1] = 1
    b[-1] = z_bc[2]

    # Apply Neumann boundary conditions with one-sided differences for non-uniform grid
    
    #Bottom boundary
    dz_p=z[1]-z[0]
    A[1, :] = 0
    A[1, 0] =-1/dz_p
    A[1, 1] = 1/dz_p
    b[1] = z_bc[1]
    
    #Top boundary
    dz_n = z[-1]-z[-2]
    A[-2, :] = 0
    A[-2, -2] = -1/dz_n
    A[-2, -1] =  1/dz_n
    b[-2] = z_bc[3]

    return A, b



def orr_sommerfeld(N, z, nu, nuT, nuT_prime, nuT_double_prime, k, U, c, U_double_prime, f,z_bc):
    #z = np.linspace(0, zi, N)
    dz = z[1] - z[0]

    D, D2, D3 = D_matrices(N, dz)
    I = np.eye(N)

    # Left-hand side
    A = -(nu+np.diag(nuT))/(1j*k)*(D2 - k**2 * I)@(D2 - k**2 * I)+(np.diag(U)-c) * (D2 - k**2 * I) - np.diag(U_double_prime) - 2/(1j*k)* np.diag(nuT_prime)*(D2-k**2*I)@D-(np.diag(nuT_double_prime)/(1j*k))*(D2 + k**2 * I)
 

    # Add the inhomogeneous term f(z) to the right-hand side
    b = np.zeros((N,),dtype=complex)
    b = f

    # Apply boundary conditions
    A, b = apply_boundary_conditions(A, b, z_bc,dz)
    #print(np.shape(A),np.shape(b))
    # Solve the linear system
    w = la.solve(A, b)

    return w


def orr_sommerfeld_non_uniform(N, z, nu, nuT, nuT_prime, nuT_double_prime, k, U, c, U_double_prime, f,z_bc):
    #z = np.logspace(log10(1e-6), log10(zi), N)
    #dz = np.diff(z)
    
    D, D2, D3 = D_matrices_non_uniform(z)
    I = np.eye(N)
    nu_matrix = np.diag(nu*np.ones(N))
    
    # Left-hand side
    A = np.diag(U-c) @ (D2 - k**2 * I) - np.diag(U_double_prime) - 2/(1j*k)* np.diag(nuT_prime)@(D2-k**2*I)@D -(nu_matrix+np.diag(nuT))/(1j*k)@(D2 - k**2 * I)@(D2 - k**2 * I)  -(np.diag(nuT_double_prime)/(1j*k))@(D2 + k**2 * I)
     

    # Add the inhomogeneous term f(z) to the right-hand side
    b = np.zeros((N,),dtype=complex)
    b = f

    # Apply boundary conditions
    A, b = apply_boundary_conditions_non_uniform(A, b, z_bc, z)
    #print(np.shape(A),np.shape(b))
    # Solve the linear system
    w = la.solve(A, b)

    return w

