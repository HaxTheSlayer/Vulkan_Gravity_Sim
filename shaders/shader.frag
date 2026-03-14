#version 450

layout(location = 0) in vec3 fragVelocity;
layout(location = 0) out vec4 outColor;

void main() {
    // Optional Magic Trick: Turn the square gl_PointSize into a perfect circle
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) {
        discard; // Kill the pixel if it's outside the circle radius
    }

    // Calculate speed (magnitude of the velocity vector)
    float speed = length(fragVelocity);
    
    // Normalize speed 
    float normalizedSpeed = clamp(speed / 200.0, 0.0, 1.0); 

    vec3 coldColor = vec3(0.0, 0.5, 1.0); // Deep Blue
    vec3 hotColor  = vec3(1.0, 0.9, 0.1); // Bright Yellow/Orange
    
    vec3 finalColor = mix(coldColor, hotColor, normalizedSpeed);
    
    // Add a slight glow effect based on speed
    outColor = vec4(finalColor, 1.0);
}