import numpy as np 
from numba import jit, prange

@jit(parallel = True, nopython = True)
def bracket_terms(NX, DX, UN, VN, DT, rho):

    DY = DX
    NY = NX
    output = np.zeros((NX, NX))
    for i in prange(NX):
        for j in prange(NY):
            output[i][j] = (
            (
                rho * (DX**2) * (DY**2) / (2*(DX**2 + DY**2))
            )  
            *
            (
            (
                1/DT
            )
            *
            (
                (UN[i + 1][j] - UN[i - 1][j])/(2*DX) + (VN[i][j + 1] - VN[i][j - 1])/(2*DY)
            )
            -
            (
                ((UN[i + 1][j] - UN[i - 1][j])/(2*DX))**2
            )
            -
            (
                2*((UN[i][j + 1] - UN[i][j - 1])/(2*DY))*((VN[i + 1][j] - VN[i - 1][j])/(2*DX))
            )
            -
            (
                ((VN[i][j + 1] - VN[i][j - 1])/(2*DY))**2
            )
            )
            )
        
    return output

@jit(nopython = True, nogil = True)
def pressure_poisson():
    pass


@jit(nopython = True, nogil = True, parallel = True)
def solve_with_cpu(NX, DT, NT, nu, rho, poisIT):

    x = np.linspace(0, 1, NX - 1)
    y = np.linspace(0, 1, NX - 1)

    VEL_NOW_U = np.zeros((NX, NX))
    VEL_NOW_V= np.zeros((NX, NX))
    PRES_NOW = np.zeros((NX, NX))

    VEL_NEXT_U = np.zeros((NX, NX))
    VEL_NEXT_V = np.zeros((NX, NX))
    PRES_NEXT = np.zeros((NX, NX))

    for i in range(NT):
       P_BRACKET = bracket_terms(NX, (x[1] - x[0]), VEL_NOW_U, VEL_NOW_V, DT, rho)

    #main loop goes here
