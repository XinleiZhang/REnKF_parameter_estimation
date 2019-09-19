# -*- coding: utf-8 -*-
"""
main program for parameter estimation case
"""

import numpy as np
import numpy.linalg as la
import os
import matplotlib.pyplot as plt
import shutil
import penalty_fun

# EnKF
def EnKF(X, HX, R, Nen):
    coeff = 1.0 / (Nen - 1.0)
    xp = X - np.mean(X)
    hxp = HX - np.mean(HX)
    pht = coeff * np.dot(xp, hxp)
    hpht = coeff * hxp.dot(hxp.T)
    inv = 1. / (hpht + R)
    kalman_gain_matrix = pht * inv
    return kalman_gain_matrix.reshape(2, 1)

# REnKF
def REnKF(X, HX, R, Nen):
    coeff = 1.0 / (Nen - 1.0)
    xp = X - np.mean(X)
    hxp = HX - np.mean(HX)
    pht = coeff * np.dot(xp, hxp)
    hpht = coeff * hxp.dot(hxp.T)
    inv = 1. / (hpht + R)
    kalman_gain_matrix = pht * inv
    hxx = coeff * np.dot(hxp, xp.T)
    p = coeff * np.dot(xp, xp.T)
    k2_gain_matrix = kalman_gain_matrix.dot(hxx) - p
    return kalman_gain_matrix.reshape(2, 1), k2_gain_matrix

# state function to propagate ensemble realizations
def state_model(theta):
    X = np.zeros((2, Nen))
    for i in range(Nen):
        X[0,i] = np.exp(-(theta[0, i] + 1)**2 - (theta[1, i] + 1)**2)
        X[1,i] = np.exp(-(theta[0, i] - 1)**2 - (theta[1, i] - 1)**2)
    return X

# state function to propagate ensemble mean
def state_mean_model(theta):
    X = np.zeros(2)
    X[0] = np.exp(-(theta[0] + 1)**2 - (theta[1] + 1)**2)
    X[1] = np.exp(-(theta[0] - 1)**2 - (theta[1] - 1)**2)
    return X

# observation_model
def obs_model(X):
    H = np.array([-1.5, -1.0])
    HX = H.dot(X)
    return HX

# to determine: prior, penalty, and filter
case_name = 'prior-2&-2'   # 'prior-2&-2' or 'prior0&0' or 'prior2&2'
penalty_type = 0         # 0: equality; 1: inequality 2: multiple
filter_name = 'EnKF'    # 'EnKF' or 'REnKF'

# show plots or not
plot_flag = 1

# model error covariance
P = np.diag([0.01, 0.01])

# observation error variance
sigmad = 0.01
R =  sigmad * sigmad
# observation
obs = -1.0005

# weight of constraint
W = 1
#hyper-parameters in penalty function
lamb = 0.1
S = 5
d = 2

#determined parameters
Nen = 1000                  # ensemble size
max_iter = 100             # maximum iteration number
cri = 1.e-2**3              # converge criteria

# file operation
if filter_name == 'REnKF':
    folder_name = 'penalty' + str(penalty_type) + '_' + \
                filter_name + '_' + case_name + '_la' + str(lamb) \
                + '_S' + str(S) + '_d' + str(d)
else:
    folder_name = filter_name + '_' + case_name

dir = './postprocessing/' + folder_name+'/'
if os.path.exists(dir):
    shutil.rmtree('./postprocessing/' + folder_name)
    os.makedirs(dir + 'samples/theta')
    os.mkdir(dir + 'samples/HX')
else:
    os.makedirs(dir + 'samples/theta')
    os.mkdir(dir + 'samples/HX')

# prior / first guess
if case_name == 'prior2&2':
    theta_mean = np.array([2, 2])
if case_name == 'prior0&0':
    theta_mean = np.array([0, 0])
if case_name == 'prior-2&-2':
    theta_mean = np.array([-2, -2])

theta_ensemble = np.random.multivariate_normal(theta_mean, P, Nen).T

# initial variables for saving
converge_flag = 'False'
theta_all = theta_mean
HX_all = []
cost1_all = []
cost2_all = []
cost3_all = []
dx1_all = []
dx2_all = []
dx_all = []
penalty_all = []

for iter in range(max_iter):    
    X = theta_ensemble
 
    state = state_model(X)
    HX = obs_model(state)
    
    # misfit check
    HX_mean = obs_model(state_mean_model(theta_mean))
    HX_all.append(HX_mean)
    misfit = (la.norm(obs - HX_mean))

    # converge check
    if misfit < cri: converge_flag = 'True'
    if misfit >=1: 
        break
        print ('filter diverge')
    # output
    print ('itertaion', iter, 'misfit =',misfit)
    with open(dir + 'results.dat', 'a') as f:
        f.write('\n'+str(iter)+' '+str(misfit)+' '+str(HX_mean) + ' ' + str(np.linalg.norm(P)))
        f.close()
    np.savetxt(dir + 'samples/theta/' + 'iter' + str(iter) + '_theta.dat', theta_ensemble)
    np.savetxt(dir + 'samples/HX/' + 'iter' + str(iter) + '_HX.dat', HX)
    if filter_name == 'EnKF':
        # ensemble observation
        obs_ensemble = np.random.normal(obs, sigmad, Nen).reshape(1,Nen) 
        # get Kalman gain matrix
        analysis_matrix = EnKF(X, HX, R, Nen)
        # delta X
        dx = analysis_matrix * (obs_ensemble - HX)
        # update augmented X
        X = X + dx
        # obtain the theta mean and save in theta_all
        theta_ensemble = X
        theta_mean = theta_ensemble.mean(axis = 1)
        theta_all = np.vstack((theta_all, theta_mean))
        dx = np.mean(dx, axis=1)
    else:
        # get regularization parameter
        Anomaly = X - np.tile(X.mean(axis = 1), (Nen, 1)).T
        P = Anomaly.dot(Anomaly.T) / Nen
        lamb_m = lamb / np.linalg.norm(P)
        lamda = 0.5 * lamb_m * (np.tanh( 1.0 / d * (iter - S)) + 1)
        # ensemble observation
        obs_ensemble = np.random.normal(obs, sigmad, Nen).reshape(1, Nen)
        # get Kalman gain matrix
        kalman_gain_matrix, k2_gain_matrix = REnKF(X, HX, R, Nen)

        # penalty function
        penalty_list = penalty_fun.penalties(penalty_type)
        penalty_mat = np.zeros((2, Nen))
        penalty_term = np.zeros((2, Nen))
        for i in range(len(penalty_list)):
            penalty, grad_penalty = penalty_list[i](theta_ensemble, Nen)
            penalty_mat += lamda * W * (grad_penalty * penalty)
            penalty_term += penalty 
        # delta X
        dx1 = kalman_gain_matrix * (obs_ensemble - HX)
        dx2 = k2_gain_matrix.dot(penalty_mat)
        if iter < 10 and (la.norm(dx2) > 5*la.norm(dx1)): 
            print('Caution: Overcorrection')
            dx2 = dx2/5
        # update X
        X = X + dx1 + dx2
        # obtain theta mean and save in theta_all
        theta_ensemble = X
        theta_mean = theta_ensemble.mean(axis = 1)
        theta_all = np.vstack((theta_all, theta_mean))
        
        dx = np.mean(dx1 + dx2, axis=1)
        dx1_all.append(la.norm(dx1))
        dx2_all.append(la.norm(dx2))
        # calculate the proportion of constraint in the cost function
        cost_3 = 0.5 * lamda * np.mean(penalty_term, axis=1).dot(W * np.mean(penalty_term,axis=1))
        cost3_all.append(cost_3)
        penalty_all.append(np.mean(penalty_term))
        
    # calculate cost function value of prior and model output
    coeff = 1.0 / (Nen - 1.0)
    xp = X - np.mean(X)
    p = coeff * xp.dot(xp.T)  
    cost_1 = 0.5 * np.dot(dx.T.dot(la.inv(p)), dx)
    
    HXa = obs_model(state_model(X))
    dy = np.mean(obs_ensemble - HXa)
    cost_2 = 0.5 * dy * dy / R
    
    cost1_all.append(cost_1)
    cost2_all.append(cost_2)
    
    # save the total update
    dx_all.append(la.norm(dx))
    
    if converge_flag == 'True':
        print ('reach convergence condition at iteration', iter )
        break
    if iter == (max_iter - 1): 
        print ('reach max iteration')
        iterations = np.arange(iter+1)

        if plot_flag:
            fig1 = plt.figure(1)
            plt.plot(iterations, theta_all[:max_iter, 0], label='theta1')
            plt.plot(iterations, theta_all[:max_iter, 1], label='theta2')
            plt.legend()
            plt.show()

            fig2 =plt.figure(2)
            plt.plot(iterations, cost1_all, label='cost1')
            plt.plot(iterations, cost2_all, label='cost2')

            if filter_name == 'REnKF':
                plt.plot(iterations, cost3_all, label='cost3')
                plt.legend()
                plt.show()

                fig3 =plt.figure(3)
                plt.plot(iterations, dx1_all, label='dx1')
                plt.plot(iterations, dx2_all, label='dx2')
                plt.legend()
                plt.show()
    
                fig4 = plt.figure(4)
                plt.plot(iterations, penalty_all[:max_iter], label='penalty')
                plt.legend()
                plt.show()   

# save
np.savetxt(dir + 'dx1_all.dat', dx1_all)
np.savetxt(dir + 'dx2_all.dat', dx2_all)
np.savetxt(dir + 'dx_all.dat', dx_all)
np.savetxt(dir + 'cost1_all.dat', cost1_all)
np.savetxt(dir + 'cost2_all.dat', cost2_all)
np.savetxt(dir + 'cost3_all.dat', cost3_all)
np.savetxt(dir + 'penalty_all.dat', penalty_all)
np.savetxt(dir + 'HX_all.dat', HX_all)
np.savetxt(dir + 'theta_all.dat',theta_all)

