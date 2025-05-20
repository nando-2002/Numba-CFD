import numpy as np 
from numba import jit, prange

@jit(nopython = True, nogil = True, parallel = True)
def solve_with_cpu(NX, DT, NT, nu, lid):

    x = np.linspace(0, 1, NX - 1)
    y = np.linspace(0, 1, NX - 1)
    X, Y = np.meshgrid(x, y)

    VEL_NOW = np.zeros((NX, NX))
    PRES_NOW = np.zeros((NX, NX))

    VEL_NEXT = np.zeros((NX, NX))
    PRES_NEXT = np.zeros((NX, NX))

    VEL_TEMP = np.zeros((NX, NX))
    PRES_TEMP = np.zeros((NX, NX))
