import numpy as np 
import pandas as pd
from typing import List, tuple, Optional


def count_other_cluster(Z : List,n : Optional[int] = None):
    if n:
        Z = Z[:n]+Z[n+1:]
    return Z.nunique()

def generate_mu_psi(): 
    mu = np.random.normal(0,5)
    psi = np.random.normal(560,476)
    return (mu,psi)

def InferDPnSSM(
    Y : pd.DataFrame,
    alpha : float, 
    m : int, 
    I : int,
    Z : List,
    Theta : List[tuple]
    ):
    for i in range(1,I):
        Z_i,Theta_i = Z,Theta
        for n in range(len(Y)):
            K_bis = count_other_cluster(Z_i,n)
            for k in range(K_bis+m):
                if k <= K_bis:
                    P_y = cSMC(Y[n],Theta_i[k])
                else : 
                    Theta_i.append(generate_mu_psi())
                    P_y = cSMC(Y[n],Theta_i[k])
                
