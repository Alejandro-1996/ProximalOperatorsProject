# Utils for the project
import numpy as np
import cv2 as cv
import cvxpy as cp
import matplotlib.pyplot as plt 
from iplabs import IPLabViewer as viewer
import sys

def nonneg(v):
    x = cp.Variable(v.shape, nonneg=True)
    obj = cp.sum_squares(x - v)

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    return x.value

def snr_db(orig, img):
    return 10*np.log10((np.sum(orig**2))/(np.sum((orig-img)**2)))

l1_prox = lambda x, lamb: np.sign(x)*np.maximum(np.abs(x) - lamb, 0)
l2_prox = lambda x, lamb: (1 - np.minimum(lamb/np.linalg.norm(x, 2), np.ones_like(x)))*x
l2_prox_param = lambda x, lamb: (1 - np.minimum(lamb, np.ones_like(x)))*x

def evaluate(reg, reg_nonneg, n=100, size=(100, ), mean=0, sigma=1, lamb=0.5, rtol=1e-5, atol=1e-8, plot=False, 
             distribution='normal', low=-1, high=1, err_thr = 1e-5, additional_params = []):
    '''
    Evaluates wheter $prox_{reg+\delta}(v) == prox_{reg}(prox_{\delta}(v))$. (Or the other way around). 

    PARAMETERS
    ---------
    reg : function
        Function that performs one proximal step of the given regularizer. It can be a closed form solution 
        (via a lambda function for example) or a function solving a problem via CVXPy. It is called with the 
        parameters v (vector created here), lambda (or lambda * np.ones_like(v)), and the additional parameters
        given by the parameter `additional_parameters`. 

    reg_nonneg : function
        Function that performs one proximal step of the given regularizer + nonnegativity constraints. It is called 
        in the same way as the function `reg`

    n : int 
        The number of different elements to try

    size : tuple of ints 
        The shape of the elements to test

    distribution : string
        Choose the type of distribution to use (currently supporting gaussian and uniform distributions)

    additional_parameters : iterable
        Any additional parameters to include in the calling of reg, reg_nonneg (e.g. p of a p-norm, size of a group, etc.)
        Defaults to an empty list (so no additional parameter is added)[]

    plot : boolean
        Whether to plot the results (only supports 1D and 2D sizes)

    err_threshold : float
        The tolerance to decide whether one of the conditions is ture or false. ANY element of ANY of the n vectors to 
        test that is outside this range, will count for a False

    RETURNS
    ---------

    (reg_nonneg_bool : boolean, avrg_reg_nonneg_error : float)
        The findings on whether $prox_{reg+\delta}(v) == prox_{reg}(prox_{\delta}(v))$ in the form of a boolean, 
        Including the average error (absolute error)

    (nonneg_reg_bool : boolean, avrg_nonneg_reg_error : float)
        The findings on whether $prox_{reg+\delta}(v) == prox_{\delta}(prox_{reg}(v))$ in the form of a boolean, 
        Including the average error (absolute error)    

    '''
    
    # Common mistake when callling the function
    try:
        len(size)
    except:
        print('WARNING:\nYou might want to add a coma at the end of the tuple in the parameter size. ')
    
    # Define conditions
    reg_nonneg_bool = True
    nonneg_reg_bool = True
    # Define nonneg projection
#         nonneg = lambda x: np.maximum(np.zeros_like(x), x)


    # Variables to keep track of largest and average errors
    largest_nonneg_reg_error = 0
    largest_reg_nonneg_error = 0
    avrg_nonneg_reg_error = 0
    avrg_reg_nonneg_error = 0

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
            reg_plus_nonneg_vect = reg_nonneg(np.copy(v), np.ones_like(v)*lamb, *additional_params)
            reg_o_nonneg_vect = reg(nonneg(np.copy(v)), np.ones_like(v)*lamb, *additional_params)
            nonneg_o_reg_vect = nonneg(reg(np.copy(v), np.ones_like(v)*lamb, *additional_params))
        # Except will do the job for CVXPy
        except:
            reg_plus_nonneg_vect = reg_nonneg(np.copy(v), lamb, *additional_params)
            reg_o_nonneg_vect = reg(nonneg(np.copy(v)), lamb, *additional_params)
            nonneg_o_reg_vect = nonneg(reg(np.copy(v), lamb, *additional_params))

        # Calculate absolute and relative errors for prox_reg(prox_nonneg(v))
        abs_error_reg_o_nonneg = np.abs(reg_plus_nonneg_vect - reg_o_nonneg_vect)
#         rel_error_reg_o_nonneg = (abs_error_reg_o_nonneg + sys.float_info.min) / (np.abs(reg_plus_nonneg_vect) + sys.float_info.min)
        rel_error_reg_o_nonneg = abs_error_reg_o_nonneg
        # Check if error is below the threshold, and
        if rel_error_reg_o_nonneg.max() > err_thr:
            reg_nonneg_bool = False
        # Keep track of largest and average errors. 
        if rel_error_reg_o_nonneg.max() > largest_reg_nonneg_error:
            largest_reg_nonneg_error = rel_error_reg_o_nonneg.max()
        avrg_reg_nonneg_error = (i*avrg_reg_nonneg_error + rel_error_reg_o_nonneg.mean())/(i+1)

        # Repeat for prox_nonneg(prox_reg(v))
        abs_error_nonneg_o_reg = np.abs(reg_plus_nonneg_vect - nonneg_o_reg_vect)
#         rel_error_nonneg_o_reg = (abs_error_nonneg_o_reg + sys.float_info.min) / (np.abs(reg_plus_nonneg_vect) + sys.float_info.min)
        rel_error_nonneg_o_reg = abs_error_nonneg_o_reg
        # Check if error is below the threshold, and
        if rel_error_nonneg_o_reg.max() > err_thr:
            nonneg_reg_bool = False
        # Keep track of largest and average errors. 
        if rel_error_nonneg_o_reg.max() > largest_nonneg_reg_error:
            largest_nonneg_reg_error = rel_error_nonneg_o_reg.max()
        avrg_nonneg_reg_error = (i*avrg_nonneg_reg_error + rel_error_nonneg_o_reg.mean())/(i+1)

        # Obsolete block using numpy.allclose - deprecated because of no information on error
#         if not(np.allclose(reg_plus_nonneg_vect, reg_o_nonneg_vect, rtol=rtol, atol=atol)) and reg_nonneg_bool:
#             reg_nonneg_bool = False
#         if not(np.allclose(reg_plus_nonneg_vect, nonneg_o_reg_vect, rtol=rtol, atol=atol)) and nonneg_reg_bool:
#             nonneg_reg_bool = False

    # Display information to the user for prox_nonneg(prox_reg(v))
    if nonneg_reg_bool:
        print(f'prox_nonneg(prox_reg(v)) SEEMS equal to prox(reg + nonneg).')
    else:
        print(f'prox_nonneg(prox_reg(v)) IS NOT equal to prox(reg + nonneg)')
    print(f'Max absolute error: {largest_nonneg_reg_error:.3e}\nAverage absolute error: {avrg_nonneg_reg_error:.3e}\n')

    # Display information to the user for prox_reg(prox_nonneg(v))
    if reg_nonneg_bool:
        print(f'prox_reg(prox_nonneg(v)) SEEMS equal to prox(reg + nonneg)')
    else:
        print(f'prox_reg(prox_nonneg(v)) IS NOT equal to prox(reg + nonneg)')
    print(f'Max absolute error: {largest_reg_nonneg_error:.3e}\nAverage absolute error: {avrg_reg_nonneg_error:.3e}\n')

    if plot:
        if len(size) == 1:
            print('Plotting example')
            plt.figure(figsize = (8, 6))
            plt.plot(v, 'o', label = '$\mathbf{v}$', markersize = 6)
            plt.plot(reg_plus_nonneg_vect, 's', label = '$\mathrm{prox}_{\mathcal{R} + \delta}(\mathbf{v})$', alpha = 1 ,markersize = 4)
            plt.plot(nonneg_o_reg_vect, 'x', label = '$\mathrm{prox}_{\delta}(\mathrm{prox}_{\mathcal{R}}(\mathbf{v}))$', alpha = 1, markersize = 9)
            plt.plot(reg_o_nonneg_vect, '+', label = '$\mathrm{prox}_{\mathcal{R}}(\mathrm{prox}_{\delta}(\mathbf{v}))$', alpha = 1, markersize = 11)
            plt.xlabel('Index', fontname='Calibri', fontsize=14)
            plt.ylabel('Value', fontname='Calibri', fontsize=14)
            plt.grid()
            plt.legend()
            plt.show()
        if len(size) == 2:
            print('Plotting example as image.')
            viewer([v, reg_plus_nonneg_vect, nonneg_o_reg_vect, reg_o_nonneg_vect, reg_plus_nonneg_vect-nonneg_o_reg_vect, 
                    reg_plus_nonneg_vect - reg_o_nonneg_vect], title = ['\mathbf{v}', '$\mathrm{prox}_{\mathcal{R} + \delta}(\mathbf{v})$', '$\mathrm{prox}_{\delta}(\mathrm{prox}_{\mathcal{R}})$', '$\mathrm{prox}_{\mathcal{R}}(\mathrm{prox}_{\delta}(\mathbf{v}))$', '$\mathrm{prox}_{\mathcal{R} + \delta}(\mathbf{v}) - \mathrm{prox}_{\delta}(\mathrm{prox}_{\mathcal{R}})$', 
                   '$\mathrm{prox}_{\mathcal{R} + \delta}(\mathbf{v}) - \mathrm{prox}_{\mathcal{R}}(\mathrm{prox}_{\delta})$'], cmap = 'viridis')

    return (reg_nonneg_bool, avrg_reg_nonneg_error), (nonneg_reg_bool, avrg_nonneg_reg_error)


################# Lp Proximals ###############################################################
def Lp_prox(v, lamb, p = 1, exp_p = False):
    
    if len(v.shape) == 1:
        nx = len(v)
        x = cp.Variable((nx))
    elif len(v.shape) == 2:
        nx, ny = v.shape
        x = cp.Variable((nx, ny))
    else:
        raise Exception('Provide an input of only 1 or 2 dimensions')

    if exp_p:
        obj = cp.power(cp.norm(x, p), p) + cp.sum_squares(x - v)/(2*lamb)
    else:
        obj = cp.pnorm(x, p) + cp.sum_squares(x - v)/(2*lamb)

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    return x.value

def Lp_plus_nonneg_prox(v, lamb, p = 1, exp_p = False):
    if len(v.shape) == 1:
        nx = len(v)
        x = cp.Variable((nx),nonneg = True)
    elif len(v.shape) == 2:
        nx, ny = v.shape
        x = cp.Variable((nx, ny),nonneg = True)
    else:
        raise Exception('Provide an input of only 1 or 2 dimensions')

    if exp_p:
        obj = cp.power(cp.norm(x, p), p) + cp.sum_squares(x - v)/(2*lamb)
    else:
        obj = cp.pnorm(x, p) + cp.sum_squares(x - v)/(2*lamb)

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)

    return x.value

########################################### TV Prox 1D #############
def TV_1D_Lp_prox(v, lamb, p = 1, exp_p = False):
    
    if len(v.shape) > 1:
        print('WARNING:\nThis function is designed for 1D signals. Stopping')
        return
    # Get dimension of v
    n = len(v)
    
    vector = np.zeros((n,))
    vector[0] = 1
    vector[1] = -1
    idx = np.arange(n)
    Q = np.matlib.repmat(vector, n, 1)
    for i, row in enumerate(Q):
        Q[i] = np.roll(row, i)
    
    # Define variable
    x = cp.Variable(n) 
    # Define de the cost function. cp.tv(x) = x_{i+} - x{i}. Results (of interest for the project) are equivalent
#     obj = cp.norm(((Q@x)[:-1]), p) + cp.sum_squares(x - v)/(2*lamb)
    if exp_p:
        obj = cp.power(cp.norm((x[1:] - x[:-1]), p), p) + cp.sum_squares(x - v)/(2*lamb)
    else:
        obj = cp.norm((x[1:] - x[:-1]), p) + cp.sum_squares(x - v)/(2*lamb)
#     obj = cp.tv(x) + cp.sum_squares(x - v)/(2*lamb)

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)    
    return x.value


def TV_1D_Lp_plus_nonneg_prox(v, lamb, p = 1, exp_p = False):
    
    if len(v.shape) > 1:
        print('WARNING:\nThis function is designed for 1D signals. Stopping')
        return
    # Get dimension of v
    n = len(v)
    
    vector = np.zeros((n,))
    vector[0] = 1
    vector[1] = -1
    idx = np.arange(n)
    Q = np.matlib.repmat(vector, n, 1)
    for i, row in enumerate(Q):
        Q[i] = np.roll(row, i)
    
    # Define variable
    x = cp.Variable(n, nonneg = True) 
    # Define de the cost function. cp.tv(x) = x_{i+} - x{i}. Results (of interest for the project) are equivalent
#     obj = cp.norm(((Q@x)[:-1]), p) + cp.sum_squares(x - v)/(2*lamb)
    if exp_p:
        obj = cp.power(cp.norm((x[1:] - x[:-1]), p), p) + cp.sum_squares(x - v)/(2*lamb)
    else:
        obj = cp.norm((x[1:] - x[:-1]), p) + cp.sum_squares(x - v)/(2*lamb)
#     obj = cp.tv(x) + cp.sum_squares(x - v)/(2*lamb)

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    
    return x.value

################################################3 Mixed norm p, q applied to the TV in 2D 
def TV_prox(v, lamb, p=1, q=1, exp = False):
    
    if len(v.shape) != 2:
        print('WARNING\n:This function is  designed for 2D operators. Terminating function.')
        return
    
    nx, ny = v.shape
    
    x = cp.Variable((nx, ny))
    
    # Defining D1 and D2x We loose one element in the *other* dimension to account for the lost element and have elements of equal size
#     D1 = cp.abs((x[:, 1:] - x[:, :-1])[:-1, :])
#     D2 = cp.abs((x[1:, :] - x[:-1, :])[:, :-1])
    D1 = (x[:, 1:] - x[:, :-1])[:-1, :]
    D2 = (x[1:, :] - x[:-1, :])[:, :-1]
    
                
#     if exp:
#         lp = cp.power(D1, p) + cp.power(D2, p)
#         obj = cp.sum(cp.power(lp, q)) + cp.sum_squares(x - v)/(2*lamb)
#     else:
#         lp = cp.power(cp.power(D1, p) + cp.power(D2, p), 1/p)
#         obj = cp.power(cp.sum(cp.power(lp, q)), 1/q) + cp.sum_squares(x - v)/(2*lamb)
#     prob.solve(solver=cp.ECOS)
    D1 = cp.reshape(D1, ((nx-1)*(ny-1)))
    D2 = cp.reshape(D2, ((nx-1)*(ny-1)))
    L = cp.vstack([D1, D2])
    if exp:
        lp = cp.power(cp.pnorm(L, p, axis=0), p)
        lq = cp.power(cp.pnorm(lp, q), q)
    else:
        lp = cp.pnorm(L, p, axis=0)
        lq = cp.pnorm(lp, q)
    obj = lq + cp.sum_squares(x - v)/(2*lamb)
    
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    
    return x.value

def TV_plus_nonneg_prox(v, lamb, p=1, q=1, exp = False):
    
    if len(v.shape) != 2:
        print('WARNING\n:This function is  designed for 2D operators. Terminating function.')
        return
    
    nx, ny = v.shape
    
    x = cp.Variable((nx, ny), nonneg = True)
    
    # Defining D1 and D2x We loose one element in the *other* dimension to account for the lost element and have elements of equal size
    D1 = (x[:, 1:] - x[:, :-1])[:-1, :]
    D2 = (x[1:, :] - x[:-1, :])[:, :-1]
#     D1 = cp.abs(x[:, 1:] - x[:, :-1])
#     D2 = cp.abs(x[1:, :] - x[:-1, :])
                
    ################### Isometric TV ################    
#     obj = cp.pnorm(D1_1 + D2_1, 2) + cp.pnorm(D1_2 +  D2_2, 2) + cp.sum_squares(x - v)/(2*lamb)
    
    ################### Non-Isometric TV ################
    # This objective represents the mixed norm l 1, 1
#     obj = cp.pnorm(D1 + D2, 1) + cp.sum_squares(x - v)/(2*lamb)
#     obj = cp.pnorm(D1, 1) + cp.pnorm(D2, 1) +cp.sum_squares(x - v)/(2*lamb)
#     obj = cp.sum(D1) + cp.sum(D2) +cp.sum_squares(x - v)/(2*lamb)
    
    
    # This objective represents the mixed norm l 2, 1
#     obj = cp.sum(cp.sqrt(cp.square(D1) + cp.square(D2))) + cp.sum_squares(x - v)/(2*lamb)
    
    # This objective represents the general mixed norm L p, q
#     lp = cp.power(D1, p) + cp.power(D2, p)
#     obj = cp.sum(cp.power(lp, q)) + cp.sum_squares(x - v)/(2*lamb)
#     obj = cp.tv(x) + cp.sum_squares(x - v)/(2*lamb)

    # This objective represents the general mixed norm L p, q with reshaping ang stacking 
    D1 = cp.reshape(D1, ((nx-1)*(ny-1)))
    D2 = cp.reshape(D2, ((nx-1)*(ny-1)))
    L = cp.vstack([D1, D2])
    if exp:
        lp = cp.power(cp.pnorm(L, p, axis=0), p)
        lq = cp.power(cp.pnorm(lp, q), q)
    else:
        lp = cp.pnorm(L, p, axis=0)
        lq = cp.pnorm(lp, q)
    obj = lq + cp.sum_squares(x - v)/(2*lamb)
    
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    
    return x.value


##################################################3 Group Sparsity 1D with p, q mixed norm ###########3
def GS_1D_prox(v, lamb, g=1, p=1, q=1, exp=False):
    if len(v.shape) > 1:
        print('WARNING:\nThis function is designed for 1D signals. Stopping')
        return
    
    n = len(v)
    # Check wheter all groups will be the same size, if not throw warning
    len_n_g = n//g
    if n/g != len_n_g:
        print('WARNING:\n`v` cannot be divided in the exact number ')
    
    x = cp.Variable(n)
    # Initialize the list of groups
    group_norms = []
    for i in np.arange(0, n, len_n_g):
        # Partition X and get norm in one step
        if exp:
            group_norms.append(cp.power(cp.norm(x[i:i+len_n_g], p), p))
        else:
            group_norms.append(cp.norm(x[i:i+len_n_g], p))
    
    if exp:
        lq = cp.power(cp.pnorm(cp.hstack(group_norms), q),q)
    else:
        lq = cp.pnorm(cp.hstack(group_norms), q)
        
        
    obj = lq + cp.sum_squares(x - v)/(2*lamb)
    
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    
    return x.value
    
def GS_1D_plus_nonneg_prox(v, lamb, g=1, p=1, q=1, exp=False):
    if len(v.shape) > 1:
        print('WARNING:\nThis function is designed for 1D signals. Stopping')
        return
    
    n = len(v)
    # Check wheter all groups will be the same size, if not throw warning
    len_n_g = n//g
    if n/g != len_n_g:
        print('WARNING:\n`v` cannot be divided in the exact number ')
    
    x = cp.Variable(n, nonneg = True)
    # Initialize the list of groups
    n_g_v = []
    n_g_x = []
    group_norms = []
    for i in np.arange(0, n, len_n_g):
        # Partition X and get norm in one step
        if exp:
            group_norms.append(cp.power(cp.pnorm(x[i:i+len_n_g], p), p))
        else:
            group_norms.append(cp.pnorm(x[i:i+len_n_g], p))
    
    if exp:
        lq = cp.power(cp.pnorm(cp.hstack(group_norms), q), q)
    else:
        lq = cp.pnorm(cp.hstack(group_norms), q)
        
    obj = lq + cp.sum_squares(x - v)/(2*lamb)
    
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    
    return x.value

######################################3 2D Group Sparsity Norm

def GS_prox(v, lamb, g=[1, 1], p=1, q=1, exp=False):
    if len(v.shape) != 2:
        print('WARNING:\nThis function is designed for 2D signals. Stopping')
        return
    
    nx, ny = v.shape
    # Check wheter all groups will be the same size, if not throw warning
    len_n_gx = nx//g[0]
    len_n_gy = ny//g[1]
    
    if nx/g[0] != len_n_gx or ny/g[1] != len_n_gy:
        print('WARNING:\n`v` cannot be exactly divided in the number requested.')
    
    x = cp.Variable((nx, ny))
    # Initialize the list of groups
    group_norms = []
    
    # Create groups and append norm directly 
    for i in np.arange(0, g[0], len_n_gx):
        for j in np.arange(0, g[0], len_n_gy):
            if exp:
                group_norms.append(cp.power(cp.pnorm(x[i:i+len_n_gx, j:j+len_n_gy], p), p))
            else:
                group_norms.append(cp.pnorm(x[i:i+len_n_gx, j:j+len_n_gy], p))
    
    # Get norm of norms 
    if exp:
        lq = cp.power(cp.pnorm(cp.hstack(group_norms), q), q)
    else:
        lq = cp.pnorm(cp.hstack(group_norms), q)
        
    obj = lq + cp.sum_squares(x - v)/(2*lamb)
    
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    
    return x.value
    
def GS_plus_nonneg_prox(v, lamb, g=[1, 1], p=1, q=1, exp=False):
    if len(v.shape) != 2:
        print('WARNING:\nThis function is designed for 2D signals. Stopping')
        return
    
    nx, ny = v.shape
    # Check wheter all groups will be the same size, if not throw warning
    len_n_gx = nx//g[0]
    len_n_gy = ny//g[1]
    
    if nx/g[0] != len_n_gx or ny/g[1] != len_n_gy:
        print('WARNING:\n`v` cannot be exactly divided in the number requested.')
    
    x = cp.Variable((nx, ny), nonneg=True)
    # Initialize the list of groups
    group_norms = []
    
    # Create groups and append norm directly 
    for i in np.arange(0, g[0], len_n_gx):
        for j in np.arange(0, g[0], len_n_gy):
            if exp:
                group_norms.append(cp.power(cp.pnorm(x[i:i+len_n_gx, j:j+len_n_gy], p), p))
            else:
                group_norms.append(cp.pnorm(x[i:i+len_n_gx, j:j+len_n_gy], p))
    
    # Get norm of norms 
    if exp:
        lq = cp.power(cp.pnorm(cp.hstack(group_norms), q), q)
    else:
        lq = cp.pnorm(cp.hstack(group_norms), q)
        
    obj = lq + cp.sum_squares(x - v)/(2*lamb)
    
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    
    return x.value


######################################3 Schatten Norms
def schatten_prox(v, lamb, schatten_norm = 'nuc'):
    
    # Initial tests on input
    if len(v.shape) != 2:
        print('WARNING:\nThis function is designed for 2D signals. Stopping...')
        return
    
    ny, nx = v.shape
    if nx != ny:
        print('WARNING:\nThis function is designed squares matrices. Stopping...')
        return
    
    x = cp.Variable((nx, ny))

    obj = cp.norm(x, schatten_norm) + cp.sum_squares(x - v)/(2*lamb)
    
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
#     prob.solve(cp.MOSEK)
    return x.value
    
def schatten_plus_nonneg_prox(v, lamb, schatten_norm = 'nuc'):
    # Initial tests on input
    if len(v.shape) != 2:
        print('WARNING:\nThis function is designed for 2D signals. Stopping...')
        return
    ny, nx = v.shape
    if nx != ny:
        print('WARNING:\nThis function is designed squares matrices. Stopping...')
        return
    
    x = cp.Variable((nx, ny), nonneg=True)
    
    obj = cp.norm(x, schatten_norm) + cp.sum_squares(x - v)/(2*lamb)
    
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    return x.value


###################################################### HESSIAN SCHATTEN ###########
def Hessian_Schatten_prox(v, lamb, p='nuc'):
    
    if len(v.shape) != 2:
        print('WARNING\n:This function is  designed for 2D operators. Terminating function.')
        return
    
    nx, ny = v.shape
    
    x = cp.Variable((nx, ny))
    
    # Defining D1 and D2x We loose one element in the *other* dimension to account for the lost element and have elements of equal size
    D1 = (x[:, 1:] - x[:, :-1])[:-1, :]
    D2 = (x[1:, :] - x[:-1, :])[:, :-1]
    D11 = (D1[:, 1:] - D1[:, :-1])[:-1, :]
    D22 = (D2[1:, :] - D2[:-1, :])[:, :-1]
    D21 = (D2[:, 1:] - D2[:, :-1])[:-1, :]
    D12 = (D1[1:, :] - D1[:-1, :])[:, :-1]
    
    hs_obj = []
#     hs_obj = 0
    for i in range(D11.shape[0]):
        for j in range(D22.shape[1]):
            top_row = cp.hstack([D11[i, j], D12[i, j]])
            bottom_row = cp.hstack([D21[i, j], D22[i, j]])
            hs_obj.append(cp.abs(cp.norm(cp.vstack([top_row, bottom_row]), p)))
        
    obj = cp.norm(cp.vstack(hs_obj), 1) + cp.sum_squares(x - v)/(2*lamb)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    
    return x.value

def Hessian_Schatten_plus_nonneg_prox(v, lamb, p='nuc'):
    
    if len(v.shape) != 2:
        print('WARNING\n:This function is  designed for 2D operators. Terminating function.')
        return
    
    nx, ny = v.shape
    
    x = cp.Variable((nx, ny), nonneg = True)
    
    # Defining D1 and D2x We loose one element in the *other* dimension to account for the lost element and have elements of equal size
    D1 = (x[:, 1:] - x[:, :-1])[:-1, :]
    D2 = (x[1:, :] - x[:-1, :])[:, :-1]
    D11 = (D1[:, 1:] - D1[:, :-1])[:-1, :]
    D22 = (D2[1:, :] - D2[:-1, :])[:, :-1]
    D21 = (D2[:, 1:] - D2[:, :-1])[:-1, :]
    D12 = (D1[1:, :] - D1[:-1, :])[:, :-1]
    
    hs_obj = []
    for i in range(D11.shape[0]):
        for j in range(D22.shape[1]):
            # Construct array over which we will take l* norm
            top_row = cp.hstack([D11[i, j], D12[i, j]])
            bottom_row = cp.hstack([D21[i, j], D22[i, j]])
            
            
            hs_obj.append(cp.abs(cp.norm(cp.vstack([top_row, bottom_row]), p)))
        
    obj = cp.norm(cp.vstack(hs_obj), 1) + cp.sum_squares(x - v)/(2*lamb)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(cp.MOSEK)
    
    return x.value

