import numpy as np 
from numba import jit, prange

@jit(nopython = True, nogil = True, parallel = False)
def bracket_terms(NX, DX, UN, VN, DT, rho):

    DY = DX
    NY = NX
    output = np.zeros((NX, NX))
    for i in prange(1, NX - 1, 1):
        for j in prange(1, NY - 1, 1):
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

@jit(nopython = True, nogil = True, parallel = False)
def pressure_poisson(P, DX, P_BRACKET, poisIT, NX):

    DY = DX
    NY = NX

    for k in range(poisIT):
        P_COPY = P.copy()
        for i in prange(1, NX - 1, 1):
            for j in prange(1, NY - 1, 1):
                P[i][j] = (
                    (
                    (
                        (P_COPY[i + 1][j] + P_COPY[i - 1][j])*DY**2
                    )
                    +
                    (
                        (P_COPY[i][j + 1] + P_COPY[i][j - 1])*DX**2
                    )
                    )
                    /
                    (
                        2*(DX**2 + DY**2)
                    )
                    -
                    (
                        P_BRACKET[i][j]
                    )
                    )
        
        P[:, -1] = P[:, -2] # dp/dx = 0 at x = 1
        P[0, :] = P[1, :]   # dp/dy = 0 at y = 0
        P[:, 0] = P[:, 1]   # dp/dx = 0 at x = 0
        P[-1, :] = 0       # p = 0 at y = 1
    
    return P


@jit(nopython = True, nogil = True, parallel = False)
def solve(NX, DT, NT, nu, rho, poisIT):

    NY = NX
    x = np.linspace(0, 1, NX)
    y = np.linspace(0, 1, NX)
    DX = x[1] - x[0]
    DY = y[1] - y[0]

    VEL_NOW_U = np.zeros((NX, NX))
    VEL_NOW_V= np.zeros((NX, NX))
    PRES_NOW = np.zeros((NX, NX))

    VEL_NEXT_U = np.zeros((NX, NX))
    VEL_NEXT_V = np.zeros((NX, NX))

    for k in range(NT):
        VEL_NOW_U = VEL_NEXT_U.copy()
        VEL_NOW_V = VEL_NEXT_V.copy()
        
        P_BRACKET = bracket_terms(NX, (x[1] - x[0]), VEL_NOW_U, VEL_NOW_V, DT, rho)
        PRES_NOW = pressure_poisson(PRES_NOW, (x[1] - x[0]), P_BRACKET, poisIT, NX)

        for i in prange(1, NX - 1, 1):
            for j in prange(1, NY - 1, 1):   
               VEL_NEXT_U[i][j] = (
               (
                    VEL_NOW_U[i][j]
               )
               +
               (
                   - VEL_NOW_U[i][j]*(DT/DX)*(VEL_NOW_U[i][j] - VEL_NOW_U[i - 1][j])
               )
               +
               (
                   - VEL_NOW_V[i][j]*(DT/DY)*(VEL_NOW_U[i][j] - VEL_NOW_U[i][j - 1])
               )
               +
               (
                   (-DT/(rho*2*DX))*(PRES_NOW[i + 1][j] - PRES_NOW[i - 1][j]) 
               )
               +
               (
                   nu
                   *
                   (
                       (DT/DX**2)
                       *
                       (
                           VEL_NOW_U[i + 1][j] - 2*VEL_NOW_U[i][j] + VEL_NOW_U[i - 1][j]
                       )
                       +
                       (DT/DX**2)
                       *
                       (
                           VEL_NOW_U[i][j + 1] - 2*VEL_NOW_U[i][j] + VEL_NOW_U[i][j - 1]
                       )
                   )
               )
               )

               VEL_NEXT_V[i][j] = (
               (
                    VEL_NOW_V[i][j]
               )
               +
               (
                   - VEL_NOW_U[i][j]*(DT/DX)*(VEL_NOW_V[i][j] - VEL_NOW_V[i - 1][j])
               )
               +
               (
                   - VEL_NOW_V[i][j]*(DT/DY)*(VEL_NOW_V[i][j] - VEL_NOW_V[i][j - 1])
               )
               +
               (
                   (-DT/(rho*2*DX))*(PRES_NOW[i][j + 1] - PRES_NOW[i][j - 1]) 
               )
               +
               (
                   nu
                   *
                   (
                       (DT/DX**2)
                       *
                       (
                           VEL_NOW_V[i + 1][j] - 2*VEL_NOW_V[i][j] + VEL_NOW_V[i - 1][j]
                       )
                       +
                       (DT/DX**2)
                       *
                       (
                           VEL_NOW_V[i][j + 1] - 2*VEL_NOW_V[i][j] + VEL_NOW_V[i][j - 1]
                       )
                   )
               )
               )
               
        VEL_NEXT_U[0, :]  = 0
        VEL_NEXT_U[:, 0]  = 0
        VEL_NEXT_U[:, -1] = 0
        VEL_NEXT_U[-1, :] = 1   # set velocity on cavity lid equal to 1
        VEL_NEXT_V[0, :]  = 0
        VEL_NEXT_V[-1, :] = 0
        VEL_NEXT_V[:, 0]  = 0
        VEL_NEXT_V[:, -1] = 0

    return PRES_NOW, VEL_NEXT_U, VEL_NEXT_V

               

#solve(41, 0.001, 100, 0.1, 1, 50)