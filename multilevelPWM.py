
# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import scipy as sp

def multilevelPWM(S=[-2,-1,0,1,2], sineharmonicnums=[1,3,5,7], sinevals=[1.5,0,0,0], cosineharmonicnums=[1,3,5,7], cosinevals=[0,0,0,0], N=2048, DCval=0, solver='ECOS', plots='OFF'):
    """
    :param S: List consisting of possible voltage levels (floats).
    :param sineharmonicnums: List of sine harmonic numbers (integers).
    :param sinevals: List of sine harmonic values (floats).
    :param cosineharmonicnums: List of cosine harmonic numbers (integers).
    :param cosinevals: List of cosine harmonic values (floats).
    :param N: Time discretization (integer, default is 2048). Must be much larger than max(sineharmonicnums, sosineharmonicnums).
    :param DCval: Desired DC value (floats).
    :param solver: One of the CVXPY solvers, (default: ECOS).
    :param plots: Set it to either 'ON' or 'OFF',  (default: 'OFF).
    :return: The waveform as a list, the switching angles as a list (degrees) and the norm of the difference of
    actual and desired as a percentage of the absolute sum of desired harmonics
    """
    Fs = np.zeros([len(sineharmonicnums), N])
    Fc = np.zeros([len(cosineharmonicnums), N])

    for k in range(len(sineharmonicnums)):
        Fs[k, :] = np.sin(2 * np.pi * (np.linspace(0, N - 1, N) / N) * sineharmonicnums[k])
    for k in range(len(cosineharmonicnums)):
        Fc[k, :] = np.cos(2 * np.pi * (np.linspace(0, N - 1, N) / N) * cosineharmonicnums[k])

    Z = cvx.Variable([N, len(S)])
    s = np.array(S)

    constraints = [np.ones(N) @ (Z @ s) / N == DCval,
                   2*Fc @ (Z @ s) / N == cosinevals,
                   2*Fs @ (Z @ s) / N == sinevals,
                   Z >= 0,
                   Z * np.ones([5, 1]) == 1]

    prob = cvx.Problem(cvx.Minimize(np.ones([1, N]) * (Z @ (s ** 2)) / N), constraints)
    prob.solve(solver=solver, verbose=False)

    x = np.matmul(Z.value, s)
    xc = np.zeros(len(x))

    for k in range(len(x)):
        idx = np.where(np.abs(np.array(s)-x[k])==np.min(np.abs(np.array(s)-x[k])))[0][0]
        xc[k] = s[idx]

    if prob.status=='infeasible':
        print('A multilevel PWM with the given constraints does not exist!')
        return -1
    elif(prob.status=='optimal_inaccurate'):
        print('Solution is inaccurate. Try another solver!')
    else:
        print('An optimal solution was found!')

    angles = 360 * (np.where(np.abs(np.diff(xc)) > 0)[0] + 1) / N
    y = np.matmul(sp.linalg.dft(N), xc) / N

    if(plots=='ON'):
        plt.figure()
        plt.title('The Multievel PWM Waveform', fontsize=30)
        plt.plot(np.linspace(0, N - 1, N), xc)

        plt.figure()
        plt.title('The Fourier Spectrum', fontsize=30)
        plt.stem(np.linspace(0,N-1,N), np.abs(y))

    diff_norm_percentage = 100*(np.linalg.norm(2*Fc@xc/N-cosinevals) + np.linalg.norm(2*Fs@xc/N-sinevals))/(np.sum(np.abs(cosinevals)) + np.sum(np.abs(sinevals)))

    return xc, angles, diff_norm_percentage
