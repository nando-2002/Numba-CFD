import numpy as np 
from numba import jit, prange

@jit(parallel = True, nopython = True)
def bracket_terms(P, NX, DX, UN, VN, DT, rho):

    DY = DX
    output = np.zeros(NX, NX)
    for i in prange(NX):
        for j in prange(NX):
            output = (
            (
                rho * (DX**2) * (DY**2) / (2*(DX**2 + DY**2))
            )  
            *
            (
                1/DT
            )
            *
            (
                UN[]
            )
    )


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

    P_BRACKET = bracket_terms(PRES_NOW, NX, (x[1] - x[0]), VEL_NOW_U, VEL_NOW_V, DT, rho)

    #main loop goes here
