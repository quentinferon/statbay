import numpy as np 
import pandas as pd
import math
from typing import List, tuple, Optional


def count_distinct_cluster(Z : List,n : Optional[int] = None):
    """_summary_

    Args:
        Z (List): _description_
        n (Optional[int], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if n:
        Z = Z[:n]+Z[n+1:]
    return Z.nunique()

def generate_mu_psi(): 
    """_summary_

    Returns:
        _type_: _description_
    """
    mu = np.random.normal(0,5)
    psi = np.random.normal(560,476)
    return (mu,psi)

def sample_z(P_y,Z,n,K_bis,N,alpha,m):
    """_summary_

    Args:
        P_y (_type_): _description_
        Z (_type_): _description_
        n (_type_): _description_
        K_bis (_type_): _description_
        N (_type_): _description_
        alpha (_type_): _description_
        m (_type_): _description_
    """
    Z_bis = Z[:n]+Z[n+1:]
    for k in range(len(P_y)):
        if k <= K_bis:
            P_y[k]*= Z_bis.count(k)/(alpha + N - 1)
        else :
            P_y[k] *= (alpha/m)/(alpha + N - 1)

def update_theta(prev_theta):
    """_summary_

    Args:
        prev_theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.random.normal(prev_theta[0],0.25),np.random.normal(prev_theta[1],0.25)

def density_funct_norm(theta,mu,sigma):
    """_summary_

    Args:
        theta (_type_): _description_
        mu (_type_): _description_
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1/np.sqrt(2*math.pi)*math.exp(-1/2*((theta-mu)/sigma)**2)

def calculate_prob_ratio(theta_bis,theta,p_y_bis,p_y):
    """_summary_

    Args:
        theta_bis (_type_): _description_
        theta (_type_): _description_
        p_y_bis (_type_): _description_
        p_y (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_1 = density_funct_norm(theta_bis[0],0,5)
    x_2 = density_funct_norm(theta_bis[1],560,476)
    y_1 = density_funct_norm(theta[0],theta_bis[0],0.25)
    y_2 = density_funct_norm(theta[1],theta_bis[1],0.25)
    w_1 = density_funct_norm(theta[0],0,5)
    w_2 = density_funct_norm(theta[1],560,476)
    z_1 = density_funct_norm(theta_bis[0],theta[0],0.25)
    z_2 = density_funct_norm(theta_bis[1],theta[1],0.25)
 
    likelihood_bis = 1
    likelihood = 1
    for i in range(len(p_y_bis)):
        likelihood *= p_y[i]
        likelihood_bis *= p_y_bis[i]

    return (x_1*x_2*likelihood_bis*y_1*y_2)/(w_1*w_2*likelihood*z_1*z_2)

#def cSMC(y, theta) :

def InferDPnSSM(
    Y : pd.DataFrame,
    alpha : float, 
    m : int, 
    I : int,
    Z : List,
    Theta : List[tuple]
    ):
    """_summary_

    Args:
        Y (pd.DataFrame): _description_
        alpha (float): _description_
        m (int): _description_
        I (int): _description_
        Z (List): _description_
        Theta (List[tuple]): _description_
    """
    N = len(Y)
    Z_list = [Z]
    Theta_list = [Theta]

    for i in range(1,I):
        Z_i,Theta_i = Z_list[i-1],Theta_list[i-1]
        for n in range(N):
            K_bis = count_distinct_cluster(Z_i,n)
            P_y = []
            for k in range(K_bis+m):
                if k <= K_bis:
                    P_y.append(cSMC(Y[n],Theta_i[k]))
                else : 
                    Theta_i.append(generate_mu_psi())
                    P_y.append(cSMC(Y[n],Theta_i[k]))
            Z_i[n] = sample_z(P_y,Z_i,n,K_bis,N,alpha,m)
        for k in range(Z_i.unique()):
            Theta_bis = update_theta(Theta_i[k])
            P_y_bis = []
            P_y_reduced = []
            for n in range(N):
                if Z_i[n] == k :
                    P_y_bis.append(cSMC(Y[n],Theta_bis))
                    P_y_reduced.append(P_y[n])

            a = calculate_prob_ratio(
                theta_bis=Theta_bis,
                theta=Theta[k], 
                p_y_bis=P_y_bis, 
                p_y=P_y_reduced)
            
            u = np.random.random()
            if u < a :
                Theta_i[k] = Theta_bis
            
        Z_list.append(Z_i)
        Theta_list.append(Theta_i)
    
    return Z_list, Theta_list

