# Vulkan N-Body Gravity Simulator

A high-performance, GPU-driven N-Body gravity simulation built from scratch using C++, Vulkan 1.3, and GLSL Compute Shaders. This project models the orbital mechanics of 10,000+ interacting particles in real-time by leveraging highly parallel GPU architecture and shared memory optimization.

## Overview
This project was developed to understand the need for parallel GPU architecture to reduce the computational complexity from $O(N^2)$. By distributing the work to 256 threads simultaneously, the workforce can be heavily optimized and the number of global memory loading operations can be reduced.

### Key Features
* **Real-Time Interactivity:** Users can use their mouse to pan the camera and WASD to move around the galaxy to observe the real-time unfolding of the cosmic structure.
* **GPU Compute Shader:** The math is entirely offloaded to the GPU using a custom GLSL compute shader.
* **Ping-Pong Storage Buffers:** Seamlessly swaps the Nth and (N+1)th state arrays in memory without expensive CPU-GPU data transfers.
* **Dynamic Particle Rendering:** Utilizes Vulkan 1.3 Dynamic Rendering to draw thousands of independent point primitives, using the fragment shader to map particle velocity to a dynamic color gradient.

## The Mathematics
The simulation calculates the gravitational force between every pair of particles. To prevent forces from shooting to infinity when particles get too close to one another, a softening factor ($\epsilon$) is added to Newton's law of Universal Gravitation to calculate the total acceleration ($\mathbf{a}_i$) for each particle:

$$\mathbf{a}_i=\sum_{j \neq i}G\frac{m_j(\mathbf{r}_j-\mathbf{r}_i)}{(||\mathbf{r}_j-\mathbf{r}_i||^2+\epsilon^2)^{3/2}}$$

This is discretized using **Semi-Implicit Euler Integration** in the compute shader to calculate the velocity and position of each particle for the next time step. Unlike standard explicit Euler, this method uses the new velocity to calculate the new position, which conserves orbital kinetic energy and keeps the galaxy stable:

$$\mathbf{v}_i(t+\Delta t)=\mathbf{v}_i(t)+\mathbf{a}_i\Delta t$$
$$\mathbf{r}_i(t+\Delta t)=\mathbf{r}_i(t)+\mathbf{v}_i(t+\Delta t)\Delta t$$

## Real-Time Interactive Demo

<video src="media/grav_sim.mp4" controls="controls" muted="muted" autoplay="autoplay" width="100%"></video>

*Leveraging highly parallel GPU architecture and shared memory optimization allows for the real-time simulation of 10,000+ particles at 60 FPS.*

## Acknowledgments
A massive thank you to [Sascha Willems](https://github.com/SaschaWillems/Vulkan) for his incredible open-source Vulkan C++ examples. His repositories were an invaluable resource for understanding Vulkan's synchronization, pipeline setup, and memory management architecture.
