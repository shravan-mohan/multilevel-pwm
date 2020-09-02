# Designing THD Optimal Multilevel PWMs 

This code generates a THD (Total Harmonic Distortion) optimal multilevel PWM (Pulse Width Modulation) waveform with a given finite set of harmonic constraints using 
linear programming.

# Use
1) The level set is given by S (for example, S = [-2,-1,0,1,2])
2) The harmonic numbers are described by their sine and socine components. Each of the sine and cosine components is described by 
harmonic numbers and values. For example, the harmonic numbers [1,3,5,7] and values [1,0,0,0] for the sine compoenents implies that the 
first sine harmonic value must be 1, and the 3rd, 5th and the 7th sine harmonic values must be 0. 
3) The number N is the time discretization number, which must be chosen much larger (and a power of 2) than the highest of the sine and cosine harmonic numbers. 
For example, if the highest harmonic number is less than 50, 2048 would be a good choice. 

# Package Requirements
1) Numpy 
2) Scipy
3) Matplotlib
4) CVXPY
