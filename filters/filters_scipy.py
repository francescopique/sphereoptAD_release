'''
reimplementazione di filtfilt
'''

import numpy as np
from scipy import signal

def pad_odd_extension(arr, pad_width):

    left_pad = []
    right_pad = []

    # Odd extension on the left
    for i in range(pad_width):
        val = 2*arr[0] - arr[pad_width - i]   
        left_pad.append(val)

    # Odd extension on the right
    for i in range(pad_width):
        val = 2*arr[-1] - arr[-2 - i]   
        right_pad.append(val)

    return np.concatenate((left_pad , arr , right_pad))

def remove_pad(x,padlen): return x[padlen:-padlen]


def butt_2(x,a,b):
    #      a[0]*y[n] = b[0] * x[n]               + d[0][n-1]
    #        d[0][n] = b[1] * x[n] - a[1] * y[n] + d[1][n-1]
    #        d[1][n] = b[2] * x[n] - a[2] * y[n] + d[2][n-1]
    #      ...
    #      d[N-2][n] = b[N-1]*x[n] - a[N-1]*y[n] + d[N-1][n-1]
    #      d[N-1][n] = b[N] * x[n] - a[N] * y[n]

    d = signal.lfilter_zi(b, a)*x[0]
    y = np.zeros(len(x))

    for i in range(len(x)):
        y[i] = (1/a[0]) * (b[0]*x[i] + d[0])
        d[0] = b[1]*x[i] - a[1]*y[i] + d[1]
        d[1] = b[2]*x[i] - a[2]*y[i]

    return y

def reverse(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = x[-1-i]
    return y

def myfiltfilt(b,a,x):
    padlen = 3*len(a)
    x = pad_odd_extension(x,padlen)
    x = butt_2(x,a,b)
    x = reverse(x)
    x = butt_2(x,a,b)
    x = reverse(x)
    x = remove_pad(x,padlen)
    return x