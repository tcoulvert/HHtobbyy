import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

N_BINS = 80
RANGE = (100, 180)

files = [
    'resampled_mass_at0p0_all.npy',    'resampled_mass_at0p9955_all.npy',
    'resampled_mass_at0p0_fold0.npy',  'resampled_mass_at0p9955_fold0.npy',
    'resampled_mass_at0p0_fold1.npy',  'resampled_mass_at0p9955_fold1.npy',
    'resampled_mass_at0p0_fold2.npy',  'resampled_mass_at0p9955_fold2.npy',
    'resampled_mass_at0p0_fold3.npy',  'resampled_mass_at0p9955_fold3.npy',
    'resampled_mass_at0p0_fold4.npy',  'resampled_mass_at0p9955_fold4.npy',
    'resampled_mass_at0p7_all.npy',  'resampled_mass_at0p99_all.npy',
    'resampled_mass_at0p7_fold0.npy',  'resampled_mass_at0p99_fold0.npy',
    'resampled_mass_at0p7_fold1.npy',  'resampled_mass_at0p99_fold1.npy',
    'resampled_mass_at0p7_fold2.npy',  'resampled_mass_at0p99_fold2.npy',
    'resampled_mass_at0p7_fold3.npy',  'resampled_mass_at0p99_fold3.npy',
    'resampled_mass_at0p7_fold4.npy',  'resampled_mass_at0p99_fold4.npy',
]

def exp_plus_gauss(x, A, tau, B, sigma, C):
    exp = A * np.exp(-x * tau)
    gauss = B * np.exp(-0.5 * ((x - 125) / sigma)**2)
    return exp + gauss + C

for file in files:
    if 'all' not in file: continue

    with open(file, "rb") as f:
        np_arr = np.load(f, allow_pickle=True)

    np_hist, bin_edges = np.histogram(np_arr, bins=N_BINS, range=RANGE, density=True)
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(np_hist))]

    popt, pcov = scp.optimize.curve_fit(exp_plus_gauss, bin_centers, np_hist, p0=[10, 1/70, 1, 2, 0])
    perr = np.sqrt(np.diag(pcov))

    print('-'*60)
    print(file)
    for i, param in enumerate(['Exp Amp', 'Exp tau', 'Gauss Amp', 'Gauss sigma', 'Y-intercept']):
        print(f"{param} = {popt[i]:.4f}±{perr[i]:.4f}")

    x_trial = np.linspace(100, 180, 1000)
    y_fit = exp_plus_gauss(x_trial, *popt)
    plt.hist(np_arr, bins=N_BINS, range=RANGE, density=True, histtype='step', color='red', label='Resampled nonResonant MC')
    #plt.errorbar(bin_centers, np_hist, yerr=np.sqrt(np_hist), color='red', marker='')
    plt.plot(x_trial, y_fit, color='blue', label='Fit - Exponential + Gaussian@125GeV')
    plt.legend()
    plt.savefig(file[:-3]+'png')
    plt.close()
