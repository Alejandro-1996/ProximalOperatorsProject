# Utils for the project
import numpy as np
import cv2 as cv

# class prox_utils:
nonneg = lambda x: np.maximum(np.zeros_like(x), x)

def evaluate(reg, reg_nonneg, n = 100, shape = (100,), mean = 0, sigma = 1, lamb = 0.5):
    # Define conditions
    reg_nonneg_bool = True
    nonneg_reg_bool = True
    # Define nonneg projection
#         nonneg = lambda x: np.maximum(np.zeros_like(x), x)

    # Start making tests
    for i in range(n):
        # Generate test vector
        v = np.random.normal(loc=mean, scale=sigma, size=shape)
        # Calculate prox_{reg+nonneg}()), prox_reg(prox_nonneg) and prox_nonneg(prox_ref) in that order
        # Try will do the job for functions that perform element wise comparisons 
        try:
            reg_plus_nonneg_vect = reg_nonneg(np.copy(v), np.ones_like(v)*lamb)
            reg_o_nonneg_vect = reg(nonneg(np.copy(v)), np.ones_like(v)*lamb)
            nonneg_o_reg_vect = nonneg(reg(np.copy(v), np.ones_like(v)*lamb))
        # Except will do the job for CVXPy
        except:
            reg_plus_nonneg_vect = reg_nonneg(np.copy(v), lamb)
            reg_o_nonneg_vect = reg(nonneg(np.copy(v)), lamb)
            nonneg_o_reg_vect = nonneg(reg(np.copy(v), lamb))
        if not(np.allclose(reg_plus_nonneg_vect, reg_o_nonneg_vect)) and reg_nonneg_bool:
            reg_nonneg_bool = False
        if not(np.allclose(reg_plus_nonneg_vect, nonneg_o_reg_vect)) and nonneg_reg_bool:
            nonneg_reg_bool = False

    if nonneg_reg_bool:
        print(f'For the given regularizer, prox_nonneg(prox_reg(v)) SEEMS equal to prox(reg + nonneg)')
    else:
        print(f'For the given regularizer, prox_nonneg(prox_reg(v)) IS NOT equal to prox(reg + nonneg)')

    if reg_nonneg_bool:
        print(f'For the given regularizer, prox_reg(prox_nonneg(v)) SEEMS equal to prox(reg + nonneg)')
    else:
        print(f'For the given regularizer, prox_reg(prox_nonneg(v)) IS NOT equal to prox(reg + nonneg)')
    return reg_nonneg_bool, nonneg_reg_bool