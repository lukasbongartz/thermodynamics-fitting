import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, root
import os
from matplotlib import gridspec
from cycler import cycler
import lmfit
from scipy.integrate import cumulative_trapezoid as ctr
import scipy.constants as const
from scipy.integrate import simps
from scipy.optimize import minimize
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

# Load data
data = pd.read_csv('Data/Example.csv', header=[0], sep=',').to_numpy()
Vgs = data[:,2] # in V
Id = data[:,1] # in A

# Temperature
T = 293

# Data preparation

def data_preparation(Vgs, Id):
    # Normalize the drain current
    Id_norm = (abs(Id)-np.min(abs(Id)))/(np.max(abs(Id))-np.min(abs(Id)))

    # Split the data into forward and backward sweeps
    peak_index = np.argmax(Vgs)
    Vgs_forward = Vgs[:peak_index+1]
    Id_forward = Id_norm[:peak_index+1]
    Vgs_backward = np.flip(Vgs[1+peak_index:])
    Id_backward = np.flip(Id_norm[1+peak_index:])

    # Sigmoid fitting 
    def generalized_sigmoid(x, a, b, c, d, v):
        a = 1
        d = 0
        return a / ((1 + np.exp(-b * (x - c))) ** v) + d

    initial_guess_forward = [1.0, 0., 0, 0, 0]
    initial_guess_backward = [1.0, 0., 0, 0, 0]

    params_forward, _ = curve_fit(generalized_sigmoid, Vgs_forward, Id_forward, 
                                  p0=initial_guess_forward, maxfev=10000)
    params_backward, _ = curve_fit(generalized_sigmoid, Vgs_backward, Id_backward, 
                                   p0=initial_guess_backward, maxfev=10000)

    # Find Vgs at psi ~ 0.5
    def find_half_Id(Vgs, params):
        def slope_func(V):
            return np.gradient(generalized_sigmoid(V, *params), V)
        
        Vgs_range = np.linspace(Vgs.min(), Vgs.max(), 1000)
        slopes = slope_func(Vgs_range)
        max_slope_index = np.argmax(np.abs(slopes))
        Vgs_at_max_slope = Vgs_range[max_slope_index]
        
        def half_Id_func(V):
            return generalized_sigmoid(V, *params) - 0.5
        
        return root(half_Id_func, Vgs_at_max_slope).x[0]

    # Analytic continuation for symmetry
    def extend_data_range(Vgs, Id, params_forward, params_backward, find_half_Id_func):

        Vgs_at_half_Id = [find_half_Id_func(Vgs, params) for params in (params_forward, params_backward)]
        
        middle_Vgs = np.mean(Vgs_at_half_Id)
        diff_to_add = abs(Vgs.max() - middle_Vgs) - abs(Vgs.min() - middle_Vgs)
        
        step_size = np.mean(np.diff(np.unique(Vgs)))
        num_points_to_add = int(np.round(diff_to_add / step_size))

        new_Vgs_below = np.arange(Vgs.min() - step_size * num_points_to_add, Vgs.min(), step_size)[:-1]
        new_Id_below = [generalized_sigmoid(new_Vgs_below, *params) for params in (params_forward, params_backward)]
        
        # Extend Vgs and Id arrays
        Vgs_extended = np.concatenate((new_Vgs_below, Vgs, np.flip(new_Vgs_below)))
        Id_extended = np.concatenate((new_Id_below[0], Id, np.flip(new_Id_below[1])))
        
        return Vgs_extended, Id_extended

    Vgs_raw_extended, Id_raw_extended = extend_data_range(Vgs, Id_norm, params_forward, params_backward, find_half_Id)

    return Id_norm, Vgs_forward, Id_forward, Vgs_backward, Id_backward, params_forward, params_backward, Vgs_raw_extended, Id_raw_extended

def plot_preparation(Vgs, Id, Id_norm, Vgs_forward, Id_forward, Vgs_backward, Id_backward, Vgs_raw_extended, Id_raw_extended):
    colormap = plt.cm.tab20b
    plt.rcParams['axes.prop_cycle'] = cycler(color=colormap(np.linspace(0, 1, 10)))
    fig = plt.figure(figsize=(12,3))

    gs0 = gridspec.GridSpec(1, 3, figure=fig, wspace = 0.4, bottom = 0.15)
    ax1 = fig.add_subplot(gs0[:,0])
    ax2 = fig.add_subplot(gs0[:,1])
    ax3 = fig.add_subplot(gs0[:,2])

    bw = 0.5
    tick_length = 2
    alpha = 1
    lw = 2 
    ms = 4

    ax1.plot(Vgs, Id, lw = lw, ms = ms, marker = 'o')
    ax1.set_ylabel(r'$I_\mathrm{D}$ (A)')
    ax1.set_xlabel(r'$V_\mathrm{GS}$ (V)')
    ax1.set_title('Transfer curve (original)')
        
    ax2.plot(Vgs, Id_norm, lw = lw, ms = ms, marker = 'o')
    ax2.set_ylabel(r'$I_\mathrm{D}$ (norm.)')
    ax2.set_xlabel(r'$V_\mathrm{GS}$ (V)')
    ax2.set_title('Transfer curve (normalized)')

    ax3.plot(Vgs_forward, Id_forward, ms = ms, label='Data', ls = '', marker = 'o', color = colormap(0))
    ax3.plot(Vgs_backward, Id_backward, ms = ms, ls = '', marker = 'o', color = colormap(0))

    mask = ~np.isin(Vgs_raw_extended, np.concatenate((Vgs_forward, Vgs_backward)))
    ax3.plot(Vgs_raw_extended[mask], Id_raw_extended[mask], color=colormap(2), ms=ms, marker='o', ls='', label='Extension')

    ax3.plot(Vgs_forward, Id_forward, label='Fit', color = plt.cm.tab20c(4))
    ax3.plot(Vgs_backward, Id_backward, color = plt.cm.tab20c(4))

    ax3.set_ylabel(r'$I_\mathrm{D}$ (norm.)')
    ax3.set_xlabel(r'$V_\mathrm{GS}$ (V)')
    ax3.set_title('Sigmoid fit')
    ax3.legend(loc = 0)

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis = 'both', which = 'major', width = bw, length = tick_length, grid_linewidth = bw, direction = 'in')
        ax.set_axisbelow(True)
        ax.grid(True,'major',alpha=0.2, zorder = 0)  
        ax.tick_params(axis = 'both', which = "minor", width = bw, length = tick_length/2, grid_linewidth = bw, direction = 'in')

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(bw)

    plt.tight_layout()
    #plt.show()

Id_norm, Vgs_forward, Id_forward, Vgs_backward, Id_backward, params_forward, params_backward, Vgs_raw_extended, Id_raw_extended = data_preparation(Vgs, Id)
plot_preparation(Vgs, Id, Id_norm, Vgs_forward, Id_forward, Vgs_backward, Id_backward, Vgs_raw_extended, Id_raw_extended)

# Fitting

def G_func(psi, a, p, q, r,T):
    return 1/a*(const.k*(T)/const.eV*1000*(psi*np.log(psi)+(1-psi)*np.log(1-psi))+p*psi**2+q*psi+r)

def get_hdu(g0,g1,G05,T):
    return -(g0+g1)+ 4*(G05+const.k*T/const.eV*1000*np.log(2))

def g_fit(data, x, T, weights):
    params = lmfit.Parameters()

    params.add('a', value=0.01, vary=1, min=0, max=1)
    params.add('p', value=-60, vary=1, min=-80, max=-40)
    params.add('q', value=60, vary=1, min=30.0, max=90.0)
    params.add('r', value=1.5, vary=1, min=0.1, max=2)

    return lmfit.minimize(g_residual, params, args=(data, x, T, weights), 
                          ftol=1e-10, xtol=1e-10, gtol=1e-10)

def pad_array(arr, target_shape, fill_value=np.ma.masked):
    result = np.full(target_shape, fill_value)
    result[:arr.shape[0]] = arr
    return result

def g_residual(params, data, x, T, weights):
    a = params['a'].value
    r = params['r'].value
    p = params['p'].value
    q = params['q'].value
    err = np.zeros_like(data[0])
    for i in range(1):
        model = G_func(x[i], a, p, q, r, T)
        error_array = ((data[i]-model)*weights[i])**2
        if error_array.shape[0] < err.shape[0]:
            error_array = pad_array(error_array, err.shape)
        err += error_array
    return err

def mu_func(psi,a,p,q, T):
    return 1/a*(np.log( psi /(1-psi ))*const.k*(T)/const.eV*1000 + 2*p*psi + q)

class OptimizationConvergedException(Exception):
    pass

def calc_shift(psi_raw, mu_raw, pt=1001):
    sep = np.argmin(mu_raw)
    psi_new = np.linspace(0,1,pt)
    mu_new1 = np.interp(psi_new,np.flip(psi_raw[0:sep]),np.flip(mu_raw[0:sep]))
    mu_new2 = np.interp(psi_new, psi_raw[sep:],mu_raw[sep:])

    final_shift = [0]
    def integral_diff(shift):
        mu_shifted1 = mu_new1 + shift
        mu_shifted2 = mu_new2 + shift

        integral_pos1 = simps(mu_shifted1[mu_shifted1 > 0], psi_new[mu_shifted1 > 0])
        integral_neg1 = simps(mu_shifted1[mu_shifted1 < 0], psi_new[mu_shifted1 < 0])
        integral_pos2 = simps(mu_shifted2[mu_shifted2 > 0], psi_new[mu_shifted2 > 0])
        integral_neg2 = simps(mu_shifted2[mu_shifted2 < 0], psi_new[mu_shifted2 < 0])
        integral_diff = (abs(integral_pos1) + abs(integral_pos2)) - (abs(integral_neg1) + abs(integral_neg2))
        if abs(integral_diff) < 1e-8:
            final_shift[0] = shift
            raise OptimizationConvergedException()
        return abs(integral_diff)
    try:
        minimize(integral_diff, 0, tol=1e-10, options={'maxiter': 1000, 'disp': True})
    except OptimizationConvergedException:
        print("Optimization converged!")
    shift = final_shift[0]
    deriv1 = np.gradient(psi_raw[0:sep+1],mu_raw[0:sep+1])
    deriv2 = np.gradient(psi_raw[sep+1:],np.flip(mu_raw[sep+1:]))
    deriv1 = np.nan_to_num(deriv1)
    deriv2 = np.nan_to_num(deriv2)


    weights = np.abs(np.concatenate((deriv1, deriv2)))**2
    weights /= np.max(weights)

    return shift, weights

def create_mask(mu_raw, psi_raw, branches=[True, True, True, True], 
                separator=0.5, cutoff_low=0.0001, cutoff_high=0.999, spacing=0):
    
    mask_topleft = np.zeros(mu_raw.shape, dtype=(bool))   
    mask_topright = np.zeros(mu_raw.shape, dtype=(bool))  
    mask_bottomleft = np.zeros(mu_raw.shape, dtype=(bool))  
    mask_bottomright = np.zeros(mu_raw.shape, dtype=(bool))

    mask_topleft[:mu_raw.argmin()]=True  
    mask_topleft[psi_raw<separator+spacing] = True   

    mask_topright[mu_raw.argmin():]=True  
    mask_topright[psi_raw<separator+spacing] = True  

    mask_bottomleft[:mu_raw.argmin()]=True  
    mask_bottomleft[psi_raw>separator-spacing] = True  

    mask_bottomright[mu_raw.argmin():]=True  
    mask_bottomright[psi_raw>separator-spacing] = True

    mask_bounds = np.zeros(psi_raw.shape, dtype=bool)
    mask_bounds[psi_raw<cutoff_low] = True
    mask_bounds[psi_raw>cutoff_high] = True
    mask_branches = np.ones(psi_raw.shape, dtype=bool)

    if branches[0]: mask_branches *= mask_topleft 
    if branches[1]: mask_branches *= mask_topright 
    if branches[2]: mask_branches *= mask_bottomleft 
    if branches[3]: mask_branches *= mask_bottomright 
    if np.sum(branches)==0: print("No branch activated!")

    mask_final = np.logical_not(np.logical_not(mask_branches)*np.logical_not(mask_bounds))
    return mask_final

def lambda_value(h1, h2, h3, psi_value, T):
    kB = const.k
    e = const.e
    lambda_value = (np.round(h1,3) + np.round(h2,3) -2*np.round(h3,3))/(kB*T*1000/e) * psi_value*(psi_value-1)
    return np.round(lambda_value,3)

def main():
    branches=[0,1,1,0] #top left, top right, bottom left, bottom right
    exp_all_G=[]
    exp_all_psi_int=[]
    exp_all_weights=[]
    exp_all_mu_raw=[]
    exp_all_psi_raw=[]
    exp_all_mu=[]
    exp_all_psi=[]
    exp_all_shift=[]

    psi_raw = Id_raw_extended
    mu_raw = -Vgs_raw_extended*1e3

    mask = create_mask(mu_raw, psi_raw, branches=branches)

    mu = np.ma.array(mu_raw, mask=mask)
    psi = np.ma.array(psi_raw, mask=mask)
    shift, weights1 = calc_shift(psi_raw, mu_raw)

    weights = (weights1[1:]+ weights1[:-1])/2
    sort_index=np.argsort(psi)
    psi_int=psi[sort_index[:-1]]+np.diff(psi[sort_index])
    G = ctr(mu[sort_index]+shift, psi[sort_index])

    distances = np.abs(np.diff(psi[sort_index], n=1))
    weights = distances
    weights /= np.sum(weights)
    exp_all_G.append(G)
    exp_all_psi_int.append(psi_int)
    exp_all_weights.append(weights)
    exp_all_mu_raw.append(mu_raw)
    exp_all_psi_raw.append(psi_raw)
    exp_all_mu.append(mu)
    exp_all_psi.append(psi)
    exp_all_shift.append(shift)
    print("Start Fitting...")

    result_g = g_fit(exp_all_G, exp_all_psi_int, T, exp_all_weights)

    r = result_g.params['r'].value
    p = result_g.params['p'].value
    q = result_g.params['q'].value
    a = result_g.params['a'].value

    G = exp_all_G[0]
    psi_int = exp_all_psi_int[0]
    weights_ = exp_all_weights[0]
    mu_raw = exp_all_mu_raw[0]
    psi_raw = exp_all_psi_raw[0]
    mu = exp_all_mu[0]
    psi = exp_all_psi[0]
    shift = exp_all_shift[0]

    mu_fit = mu_func(psi_int, a, p, q, T)
    x = np.linspace(0, 1, 100000)[1:-1]
    G_0 = G_func(x, 1, p, q, r, T)

    hdd = 2 * G_0[-1]
    huu = 2 * G_0[0]
    hdu = get_hdu(G_0[-1], G_0[0], G_0[int(len(x)/2)], T)

    fit_results = {
        r'$h_\mathrm{dd}$': np.round(hdd, 3),
        r'$h_\mathrm{uu}$': np.round(huu, 3),
        r'$h_\mathrm{du}$': np.round(hdu, 3),
        r'$\lambda$': lambda_value(hdd, huu, hdu, 0.5, T)
    }

    print('Fit Results:')
    print(fit_results)

    fig = plt.figure(figsize=(12,3))

    gs0 = gridspec.GridSpec(1, 3, figure=fig, wspace = 0.4, bottom = 0.15)
    ax1 = fig.add_subplot(gs0[:,0])
    ax2 = fig.add_subplot(gs0[:,1])
    ax3 = fig.add_subplot(gs0[:,2])

    bw = 0.5
    tick_length = 2
    alpha = 1
    lw = 2 
    ms = 4

    ax1.plot(psi_raw, mu_raw+shift, ls='', marker='o', ms = ms, label = r'Exp.')
    ax1.plot(psi_int, mu_fit, ls='-', marker='', ms = ms, label = r'Fit', lw = lw, color = plt.cm.tab20c(4))

    ax1.set_xlabel(r'$\psi$')
    ax1.set_ylabel(r'$\mu(\psi)$ (meV)')
    ax1.set_ylim((mu_raw.min()+shift, mu_raw.max()+shift))
    ax1.legend(loc=0)
    
    ax2.plot(psi_int, a*G, marker='o', label='Exp.', ls='', ms = ms)
    ax2.plot(x, G_0, marker='', label='Fit', ls = '-', lw = lw, color = plt.cm.tab20c(4))
    ax2.set_xlabel(r'$\psi$')
    ax2.set_ylabel(r'$G(\psi)$ (meV)')
    ax2.legend(loc=0)


    # Prepare data for bar plot
    h_params = [r'$h_\mathrm{dd}$', r'$h_\mathrm{uu}$', r'$h_\mathrm{du}$']
    h_values = [fit_results[param] for param in h_params]

    x_h = np.arange(len(h_params))  # the label locations
    width_h = 0.5  # width of the bars
    bars = ax3.bar(x_h, h_values, width_h, color=plt.cm.tab20b(0))
    ax3.set_ylabel('Energy (meV)')
    ax3.set_xticks(x_h, h_params)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0))
    ax3.set_xlabel('')

    for ax in [ax1, ax2]:
        ax.tick_params(axis = 'both', which = 'major', width = bw, length = tick_length, grid_linewidth = bw, direction = 'in')
        ax.set_axisbelow(True)
        ax.grid(True,'major',alpha=0.2, zorder = 0)  
        ax.tick_params(axis = 'both', which = "minor", width = bw, length = tick_length/2, grid_linewidth = bw, direction = 'in')

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(bw)

    for ax in [ax3]:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(bw)
        ax.set_axisbelow(True)
        ax.grid(True, 'major', axis = 'y', alpha=0.2)
        ax.tick_params(axis = 'y', width = bw, length = tick_length, grid_linewidth = bw, direction = 'in')
        ax.tick_params(axis = 'x', length = 0)

    plt.tight_layout()
    plt.savefig('Fit_Results/Fit_Plot.pdf', dpi=300)
    plt.show()



    # Save data
    data = np.column_stack((psi_int.compressed(), a*G[~psi_int.mask]))
    np.savetxt("Fit_Results/G_Exp.txt", data, delimiter=",")

    data = np.column_stack((x, G_0))
    np.savetxt("Fit_Results/G_Fit.txt", data, delimiter=",")

    fitted_data = np.column_stack((psi_int.compressed(), mu_fit[~psi_int.mask]))
    np.savetxt("Fit_Results/mu_fit.txt", fitted_data, delimiter=",")

    with open("Fit_Results/Fit_Params.txt", "w") as file:
        file.write("alpha: " + str(a) + "\n")
        file.write("h_dd: " + str(hdd) + "\n")
        file.write("h_uu: " + str(huu) + "\n")
        file.write("h_du: " + str(hdu) + "\n")

if __name__ == "__main__":
    main()


