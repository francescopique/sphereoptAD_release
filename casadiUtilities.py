'''
Utilities for manipulation of casadi variables
'''

import casadi as ca

def derive(q,dt):
    '''
    Returns the derivatives of an array of casadi variables
    to be interpreted over time, using centered finite
    difference approximation,
    q: casadi MX array (must be uniformly sampled)
    dt: interval between samples

    '''
    temp = ca.MX.zeros(q.shape)
    samples = q.shape[0]
    for i in range(samples):
        if i == 0: temp[i,:] = (q[1,:]-q[0,:])/dt
        elif i == (samples-1): temp[i,:] = (q[i,:]-q[i-1,:])/dt
        else: temp[i,:] = (q[i+1,:]-q[i-1,:])/(2*dt)
    return temp

def Lq(q,dt):
    '''
    Returns Lq, as a bulding block for the 2nd difference,
    discrete acceleration operator:

    Lq_i = q_(i-1) + 2q_i + q_(i+1),
    i = 2, ... , N - 1
    '''
    temp = ca.MX.zeros(q.shape)
    samples = q.shape[0]
    for i in range(1,samples-1):
        # if i == 0: temp[i,:] = q[i+2,:] - 2*q[i+1,:] + q[i,:] 
        # elif i == (samples-1): temp[i,:] = q[i,:] - 2*q[i-1,:] + q[i-2,:]
        # else:  temp[i,:] = q[i+1,:] - 2*q[i,:] + q[i-1,:]
        temp[i,:] = q[i+1,:] - 2*q[i,:] + q[i-1,:]

    return temp/(dt**2)
