import cpu_solver as cpu
import gpu_solver as gpu

import numpy
from matplotlib import pyplot, cm
import time

def main():

    #config begins here

    NX = 41                   #square grid, therefore x and y step counts are always the same
    DT = 0.001                   #seconds
    NT = 100                  #DT * NT = total simulation time
    nu = 0.1                    #kinematic viscosity m^2/s
    hVEL = 1                    #m/s
    rho = 1                     #kg/m^3
    
    poisson_iterations = 500   #internal iterations to calculate pressre

    nvidia_gpu_present = False  #if nvidia gpu present, with cudatoolkits installed, we can run it on the gpu
    
    if nvidia_gpu_present == True:
        gpu.solve(NX, DT, NT, nu, hVEL, rho, poisson_iterations)
    else:
        start = time.time()
        p, u, v = cpu.solve(NX, DT, NT, nu, rho, poisson_iterations)
        end = time.time()
        print(end - start)
            
    x = numpy.linspace(0, 1, NX)
    y = numpy.linspace(0, 1, NX)
    X, Y = numpy.meshgrid(x, y)
    fig = pyplot.figure(figsize=(9, 7), dpi=100)
    # plotting the pressure field as a contour
    pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
    pyplot.colorbar()
    # plotting the pressure field outlines
    pyplot.contour(X, Y, p, cmap=cm.viridis)  
    # plotting velocity field
    pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.show()

    fig = pyplot.figure(figsize=(9, 7), dpi=100)
    pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    pyplot.colorbar()
    pyplot.contour(X, Y, p, cmap=cm.viridis)
    pyplot.streamplot(X, Y, u, v)
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.show()
    

if __name__ == "__main__":
    main()