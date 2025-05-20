import cpu_solver as cpu
import gpu_solver as gpu

def main():

    #config begins here

    number_of_grid_steps = 41               #square grid, therefore x and y step counts are always the same
    time_step_size = 0.01                   #seconds
    time_step_count = 500                   #time_step_size * time_step_count = total simulation time
    kinematic_viscosity = 0.1               #kinematic viscosity m^2/s
    horiz_velocity = 1                      #m/s
    density = 1                             #kg/m^3
    
    poisson_iterations = 200                #internal iterations to calculate pressre

    nvidia_gpu_present = False              #if nvidia gpu present, with cudatoolkits installed, we can run it on the gpu
    
    if nvidia_gpu_present == True:
        gpu.solve_with_gpu(number_of_grid_steps, time_step_size, time_step_count, kinematic_viscosity, horiz_velocity, density, poisson_iterations)
    else:
        cpu.solve_with_cpu(number_of_grid_steps, time_step_size, time_step_count, kinematic_viscosity, horiz_velocity, density, poisson_iterations)

    pass

if __name__ == "__main__":
    main()