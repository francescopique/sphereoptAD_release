'''
Reimplementation of numpy's filtfilt with casadi variables.
Here, only a 2nd order butterworth filter is implemented.
Francesco may 2025
'''

from scipy import signal
import matplotlib.pyplot as plt
import casadi as ca

def pad_odd_extension(arr, pad_width):

    left_pad = ca.MX(pad_width,1)
    right_pad = ca.MX(pad_width,1)

    # Odd extension on the left
    for i in range(pad_width):
        val = 2*arr[0] - arr[pad_width - i]   
        left_pad[i] = val

    # Odd extension on the right
    for i in range(pad_width):
        val = 2*arr[-1] - arr[-2 - i]   
        right_pad[i] = val

    return ca.vertcat(left_pad , arr , right_pad)

def remove_pad(x,padlen): return x[padlen:-padlen]


def butt_2(x,a,b):
    #      a[0]*y[n] = b[0] * x[n]               + d[0][n-1]
    #        d[0][n] = b[1] * x[n] - a[1] * y[n] + d[1][n-1]
    #        d[1][n] = b[2] * x[n] - a[2] * y[n] + d[2][n-1]
    #      ...
    #      d[N-2][n] = b[N-1]*x[n] - a[N-1]*y[n] + d[N-1][n-1]
    #      d[N-1][n] = b[N] * x[n] - a[N] * y[n]

    d = signal.lfilter_zi(b, a)*x[0]
    y = ca.MX(x.shape[0],1)

    for i in range(x.shape[0]):
        y[i] = (1/a[0]) * (b[0]*x[i] + d[0])
        d[0] = b[1]*x[i] - a[1]*y[i] + d[1]
        d[1] = b[2]*x[i] - a[2]*y[i]

    return y

def reverse(x):
    y = ca.MX(x.shape[0],1)
    for i in range(x.shape[0]):
        y[i] = x[-1-i]
    return y

def myfiltfilt(b,a,x):
    '''
    b: denominator coefficients
    a: numeraotr coefficients
    x: array to be filtered

    returns array filtered twice (zero phase) using second order 
    butterworth filter, using coefficients given by:

    b, a = scipy.signal.butter(n, fc/(fs/2))

    where:

    fc: cut off frequency
    fs: sampling frequency

    Padding is with an odd extension on both
    ends with deafult length 3*len(a)


    '''
    padlen = 3*len(a)
    x = pad_odd_extension(x,padlen)
    x = butt_2(x,a,b)
    x = reverse(x)
    x = butt_2(x,a,b)
    x = reverse(x)
    x = remove_pad(x,padlen)
    return x

def cma(y):
    '''
    Centered moving average filter with window = 1,
    on the left and right (effectively 3)
    '''
    yy = ca.MX(y.shape[0],1)
    k = 1
    yy[0] = (y[0] + y[1])/2
    yy[1] = (y[0] + y[1] + y[2])/3
    for i in range(2,y.shape[0]-1):
        yy[i] = yy[i-1] + (y[i+k]-y[i-k-1])/(2*k+1)
    yy[-1] = (y[-1] + y[-2])/2
    return yy

def cma_k2(y):
    '''
    Centered moving average filter with window = 2,
    on the left and right (effectively 5)
    '''
    yy = ca.MX(y.shape[0],1)
    k = 2
    yy[0] = (y[0] + y[1] + y[2])/3
    yy[1] = (y[0] + y[1] + y[2] + y[3])/4
    yy[2] = (y[0] + y[1] + y[2] + y[3]+ y[4])/5
    for i in range(3,y.shape[0]-2):
        yy[i] = yy[i-1] + (y[i+k]-y[i-k-1])/(2*k+1)
    yy[-2] = (y[-4] + y[-3] + y[-2] + y[-1])/4
    yy[-1] = (y[-3] + y[-2] + y[-1])/3
    return yy

def cma_k2edge(y):
    '''
    Centered moving average filter with window = 2,
    on the left and right (effectively 5).
    Discrads 2 edge values.
    '''
    yy = ca.MX(y.shape[0],1)
    k = 2
    yy[0] = (y[0] + y[1] + y[2])/3
    yy[1] = (y[0] + y[1] + y[2] + y[3])/4
    yy[2] = (y[0] + y[1] + y[2] + y[3]+ y[4])/5
    for i in range(3,y.shape[0]-2):
        yy[i] = yy[i-1] + (y[i+k]-y[i-k-1])/(2*k+1)
    yy[-2] = y[-2]
    yy[-1] = y[-1]
    return yy
