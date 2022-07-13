import numpy as np
import pandas as pd

def fetch_params_to_fit(kwargs_status):
    if type(kwargs_status) is pd.DataFrame:
        kwargs_status = kwargs_status.to_dict('list') # if kwargs_status[x] is list it stays as list

    if all([type(kwargs_status[x]) is list for x in kwargs_status]):
        params_to_fit = [x for x in kwargs_status if (kwargs_status[x][0] == 'True') ^ (kwargs_status[x][0] == 'Share')]
        return params_to_fit
    else:
        params_to_fit = [x for x in kwargs_status if (kwargs_status[x] == 'True') ^ (kwargs_status[x] == 'Share')]
        return params_to_fit


def Converter(kwargs):
    if type(kwargs) is pd.DataFrame:
        kwargss = kwargs.to_dict('records')
        if len(kwargss)>1:
            return kwargss
        else:
            return kwargss[0]
    else:
        return kwargs


def eis_df(df):
    Z = df['zr']+1j*df['zi']
    C = 1/(df['f']*2*np.pi*Z*1j)
    df_dict = {'f':list(df['f']),
           'zreal':list(df['zr']),
           'zimag':list(df['zi']),
           'zphase':list(180/np.pi*np.angle(Z)),
           'zabs': list(np.abs(Z)),
           'creal':list(np.real(C)),
           'cimag': list(np.imag(C))}
    return pd.DataFrame(df_dict)


def N_range(nr=60, logNmin=-2, logNmax=2.5):
    logN = np.linspace(logNmin, logNmax, nr)
    return logN

def regularization_matx(N=60):
    L=-np.identity(N)
    np.fill_diagonal(L[0:,1:],1)
    L[0][0]=2
    L[-1][-1]=2
    return L

def gaussian(x,mean,std=0.4):
    G=1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std**2))
    return G/np.sum(G)


def cole(f,Z):
    return 1/(2*np.pi*f*Z*1j)
