import numpy as np 
from numba import jit, prange

@jit(nopython = True, nogil = True, parallel = True)
def solve_with_cpu(NX, DT, NT):

    x = np.linspace(0, 1, NX)
    y = np.linspace(0, 1, NX)

    VEL_NOW = np.zeros((NX, NX))
    PRES_NOW = np.zeros((NX, NX))

    VEL_NEXT = np.zeros((NX, NX))
    PRES_NEXT = np.zeros((NX, NX))

    VEL_TEMP = np.zeros((NX, NX))
    PRES_TEMP = np.zeros((NX, NX))

    for i in prange(NX):
        print(i)