# Utils for the project
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

# class prox_utils:
nonneg = lambda x: np.maximum(np.zeros_like(x), x)

# def solver(inp, params, function = nonnegativity):
#     function = params.function
#     lamb = params.lambda
#     if function = 'nonnegativity':
#         return 
def evaluate(reg, reg_nonneg, n=100, size=(100,), mean=0, sigma=1, lamb=0.5, rtol=1e-5, atol=1e-8, plot=False, 
             distribution='normal', low=-1, high=1):
    # Define conditions
    reg_nonneg_bool = True
    nonneg_reg_bool = True
    # Define nonneg projection
#         nonneg = lambda x: np.maximum(np.zeros_like(x), x)

    # Start making tests
    for i in range(n):
        # Generate test vector
        if distribution == 'normal':
            v = np.random.normal(loc=mean, scale=sigma, size=size)
        if distribution == 'uniform':
            v = np.random.uniform(low=low, high=high, size=size)
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
        if not(np.allclose(reg_plus_nonneg_vect, reg_o_nonneg_vect, rtol=rtol, atol=atol)) and reg_nonneg_bool:
            reg_nonneg_bool = False
        if not(np.allclose(reg_plus_nonneg_vect, nonneg_o_reg_vect, rtol=rtol, atol=atol)) and nonneg_reg_bool:
            nonneg_reg_bool = False

    if nonneg_reg_bool:
        print(r'For the given regularizer, $\mathrm{prox}_{\mathcal{R}}(\mathrm{prox}_{\delta})$ SEEMS equal to prox(reg + nonneg)')
    else:
        print(r'For the given regularizer, $\mathrm{prox}_{\delta}(\mathrm{prox}_{\mathcal{R}})$ IS NOT equal to prox(reg + nonneg)')

    if reg_nonneg_bool:
        print(r'For the given regularizer, $\mathrm{prox}_{\mathcal{R}}(\mathrm{prox}_{\delta})$ SEEMS equal to prox(reg + nonneg)')
    else:
        print(r'For the given regularizer, $\mathrm{prox}_{\delta}(\mathrm{prox}_{\mathcal{R}})$ IS NOT equal to prox(reg + nonneg)')
    
    if plot:
        print('Plotting example')
        plt.figure(figsize = (12, 9))
        plt.plot(v, label = 'Original')
        plt.plot(reg_plus_nonneg_vect, 'k-', label = '$\mathrm{prox}_{\mathcal{R} + \delta}$ (Target)')
        plt.plot(nonneg_o_reg_vect, 'rs-.', label = '$\mathrm{prox}_{\delta}(\mathrm{prox}_{\mathcal{R}})$', alpha = 0.5)
        plt.plot(reg_o_nonneg_vect, 'g1--', label = '$\mathrm{prox}_{\mathcal{R}}(\mathrm{prox}_{\delta})$', alpha = 0.5)
        plt.plot(reg(np.copy(v),  lamb), 'm.-', label = '$\mathrm{prox}_{\mathcal{R}}$ (Middle Step)', alpha = 0.5)
        plt.grid()
        plt.legend()
        plt.show()
    return reg_nonneg_bool, nonneg_reg_bool