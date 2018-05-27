import numpy as np
import statsmodels.api as sm

def generalized_KF_update(prev_S, prev_P, y, Z, sys_params):
    
    '''
    y: obervation values (r_t)
    Z: obervation model ([1, F_t] = [1, r_{1, t}, ..., r{k, t}])
    '''
    
    SST = sys_params['SST'] # I_{1+k}
    Q = sys_params['Q'] # diag matrix
    H = sys_params['H'] # variance of r - (a + b*rm)
    
    # === state prediction ===
    pred_S = SST.dot(prev_S)
    
    # === prediction dispersion ===
    pred_P = SST.dot(prev_P).dot(SST.T) + Q
    
    # === observation prediction ===
    pred_y = Z.dot(pred_S)
    
    # === prediction error ===
    pred_err = y - pred_y
    
    # === error dispersion ===
    F = Z.dot(pred_P).dot(Z.T) + H # assume 1 by 1
    
    # === kalman gain ===
    G = pred_P.dot(Z.T) / F
    
    # === state estimate ===
    est_S = pred_S + G.dot(pred_err)
    
    # === dispersion estimate ===
    est_P = pred_P - G.dot(Z).dot(pred_P)
    
    return est_S, est_P

def generalized_KF(obs_val, obs_model, init_state, init_dispersion, sys_params):
    
    assert(obs_val.shape[0] == obs_model.shape[0])
    N = obs_val.shape[0]
    k = len(init_state)
    
    S = np.array(init_state).reshape(k, 1)
    P = np.array(init_dispersion).reshape(k, k)
    
    kf = np.empty([N, k])
    
    o_val = obs_val.values
    o_model = obs_model.values
    
    for i in range(N):
        y = o_val[i]
        Z = o_model[i].reshape(1, k)
        
        S, P = generalized_KF_update(S, P, y, Z, sys_params)
        kf[i, :] = S.T
        
    return kf

def get_init_state_from_OLS(R, Rm):
    
    k = Rm.shape[1] + 1 if len(Rm.shape) > 1 else 2
    res = sm.OLS(R, sm.add_constant(Rm)).fit(cov_type='HC0')
    
    S = res.params
    P = np.zeros((k, k))
    
    sys_params = dict()
    sys_params['H'] = res.mse_model,
    sys_params['Q'] = np.diag(res.HC0_se.values)
    sys_params['SST'] = np.eye(k)

    return S, P, sys_params