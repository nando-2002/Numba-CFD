import numpy as np 
from numba import jit, prange

@jit(nopython = True, nogil = True, parallel = True)
def bracket_terms(NX, DX, UN, VN, DT, rho):

    DY = DX
    NY = NX
    output = np.zeros((NX, NX))
    for i in prange(1, NX - 1, 1):
        for j in prange(1, NY - 1, 1):
            output[j][i] = (
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
                (UN[j][i + 1] - UN[j][i - 1])/(2*DX) + (VN[j + 1][i] - VN[j - 1][i])/(2*DY)
            )
            -
            (
                ((UN[j][i + 1] - UN[j][i - 1])/(2*DX))**2
            )
            -
            (
                2*((UN[j + 1][i] - UN[j - 1][i])/(2*DY))*((VN[j][i + 1] - VN[j][i - 1])/(2*DX))
            )
            -
            (
                ((VN[j + 1][i] - VN[j - 1][i])/(2*DY))**2
            )
            )
            )
        
    return output

@jit(nopython = True, nogil = True, parallel = True)
def pressure_poisson(P, DX, P_BRACKET, poisIT, NX):

    DY = DX
    NY = NX

    for k in range(poisIT):
        P_COPY = P.copy()
        for i in prange(1, NX - 1, 1):
            for j in prange(1, NY - 1, 1):
                P[j][i] = (
                    (
                    (
                        (P_COPY[j][i + 1] + P_COPY[j][i - 1])*DY**2
                    )
                    +
                    (
                        (P_COPY[j + 1][i] + P_COPY[j - 1][i])*DX**2
                    )
                    )
                    /
                    (
                        2*(DX**2 + DY**2)
                    )
                    -
                    (
                        P_BRACKET[j][i]
                    )
                    )
        
        P[:, -1] = P[:, -2] # dP/dx = 0 at x = 2
        P[0, :] = P[1, :]   # dP/dy = 0 at y = 0
        P[:, 0] = P[:, 1]   # dP/dx = 0 at x = 0
        P[-1, :] = 0        # P = 0 at y = 2
    
    return P


@jit(nopython = True, nogil = True, parallel = True)
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
               VEL_NEXT_U[j][i] = (
               (
                    VEL_NOW_U[j][i]
               )
               +
               (
                   - VEL_NOW_U[j][i]*(DT/DX)*(VEL_NOW_U[j][i] - VEL_NOW_U[j - 1][i])
               )
               +
               (
                   - VEL_NOW_V[j][i]*(DT/DY)*(VEL_NOW_U[j][i] - VEL_NOW_U[j][i - 1])
               )
               +
               (
                   (-DT/(rho*2*DX))*(PRES_NOW[j][i + 1] - PRES_NOW[j][i - 1]) 
               )
               +
               (
                   nu
                   *
                   (
                       (DT/DX**2)
                       *
                       (
                           VEL_NOW_U[j][i + 1] - 2*VEL_NOW_U[j][i] + VEL_NOW_U[j][i - 1]
                       )
                       +
                       (DT/DX**2)
                       *
                       (
                           VEL_NOW_U[j + 1][i] - 2*VEL_NOW_U[j][i] + VEL_NOW_U[j - 1][i]
                       )
                   )
               )
               )

               VEL_NEXT_V[j][i] = (
               (
                    VEL_NOW_V[j][i]
               )
               +
               (
                   - VEL_NOW_U[j][i]*(DT/DX)*(VEL_NOW_V[j][i] - VEL_NOW_V[j - 1][i])
               )
               +
               (
                   - VEL_NOW_V[j][i]*(DT/DY)*(VEL_NOW_V[j][i] - VEL_NOW_V[j][i - 1])
               )
               +
               (
                   (-DT/(rho*2*DX))*(PRES_NOW[j + 1][i] - PRES_NOW[j - 1][i]) 
               )
               +
               (
                   nu
                   *
                   (
                       (DT/DX**2)
                       *
                       (
                           VEL_NOW_V[j][i + 1] - 2*VEL_NOW_V[j][i] + VEL_NOW_V[j][i - 1]
                       )
                       +
                       (DT/DX**2)
                       *
                       (
                           VEL_NOW_V[j + 1][i] - 2*VEL_NOW_V[j][i] + VEL_NOW_V[j - 1][i]
                       )
                   )
               )
               )
               
        VEL_NEXT_U[0, :]  = 0
        VEL_NEXT_U[:, 0]  = 0
        VEL_NEXT_U[:, -1] = 0
        VEL_NEXT_U[-1, :] = 1    # set velocity on cavity lid equal to 1
        VEL_NEXT_V[0, :]  = 0
        VEL_NEXT_V[-1, :] = 0
        VEL_NEXT_V[:, 0]  = 0
        VEL_NEXT_V[:, -1] = 0

    return PRES_NOW, VEL_NEXT_U, VEL_NEXT_V

               

#solve(41, 0.001, 100, 0.1, 1, 50)