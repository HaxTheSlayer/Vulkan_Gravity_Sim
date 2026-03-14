#version 450

layout(location = 0) in vec4 inPositionMass; 
layout(location = 1) in vec4 inVelocity;     

layout(location = 0) out vec3 fragVelocity;

layout(push_constant) uniform PushConstants {
    mat4 mvp; // Model-View-Projection matrix for the 3D camera
} pcs;

void main() {
    // Transform the 3D particle position into screen space
    gl_Position = pcs.mvp * vec4(inPositionMass.xyz, 1.0);
    
    // Required when drawing points: sets the size of the pixel square
    gl_PointSize = 2.0; 
    
    // Pass velocity to the fragment shader for coloring
    fragVelocity = inVelocity.xyz; 
}