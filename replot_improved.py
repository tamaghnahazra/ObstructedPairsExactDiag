import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

plt.rcParams.update({'font.size': 24})

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def format_sci(num):
    if num == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(num))))
    mant = num / 10**exp
    return rf"{mant:.2f} \times 10^{{{exp}}}"

def format_equation(E0, Ds):
    # If Ds is incredibly small (numerical noise), just format it nicely as 0
    if abs(Ds) < 1e-7:
        return r"$E(\phi) = E_0$"
    else:
        return r"$E(\phi) = E_0 + \frac{1}{2} D_s \phi^2$" + "\n" + rf"$D_s = {format_sci(Ds)}$"

def plot_E_vs_phi(csv_file, N_cells, out_file):
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
        
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    phi_vals = data[:, 0]
    energies = data[:, 1] / N_cells  # Divide by number of unit cells
    
    # Fit
    popt, _ = curve_fit(parabola, phi_vals, energies)
    a, b, c = popt
    Ds = 2 * a
    E0 = c
    
    # Remove constant factor for plotting
    energies_plot = energies - E0
    
    plt.figure(figsize=(8, 6))
    
    phi_smooth = np.linspace(min(phi_vals), max(phi_vals), 100)
    
    if abs(Ds) < 1e-7:
        # Force it to be a perfect zero flat line to hide numerical noise
        fit_curve = np.zeros_like(phi_smooth)
        energies_plot = np.zeros_like(energies_plot)
    else:
        fit_curve = parabola(phi_smooth, *popt) - E0
        
    plt.plot(phi_vals, energies_plot, 'o', markersize=10, label='ED Data')
    plt.plot(phi_smooth, fit_curve, '-', linewidth=3, label='Fit')
    
    if abs(Ds) < 1e-7:
        # Prevent zooming in to e-15 numerical noise
        plt.ylim([-1e-6, 1e-6])
        
    eq_text = format_equation(E0, Ds)
    if abs(Ds) < 1e-7:
        y_pos = 0.5
    else:
        y_pos = 0.8
        
    plt.text(0.5, y_pos, eq_text, transform=plt.gca().transAxes, 
             fontsize=24, horizontalalignment='center', verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.xlabel(r'Twist angle $\phi$')
    plt.ylabel(r'$E(\phi) - E_0$')
    
    # Enforce x 10^x notation for axes
    formatter = plt.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

def plot_stiffness_vs_Wb(csv_file, N_cells, out_file):
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
        
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    W_b_vals = data[:, 0]
    stiffness_vals = data[:, 1] / N_cells  # Divide by number of unit cells
    
    plt.figure(figsize=(8, 6))
    plt.plot(W_b_vals, stiffness_vals, 's-', markersize=10, linewidth=3, color='purple')
    
    plt.xlabel(r'$W_b / t_b$')
    plt.ylabel(r'$D_s$ per unit cell')
    
    formatter = plt.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Task 1: 5x4 = 20 unit cells
    plot_E_vs_phi('Task1_Wb_0_5x4_4b.csv', 20, 'Task1_Wb_0_5x4_4b_improved.png')
    
    # Task 2: 4x4 = 16 unit cells
    plot_E_vs_phi('Task2_Wb_0_4x4_8b.csv', 16, 'Task2_Wb_0_4x4_8b_improved.png')
    
    # Task 3: 5x4 = 20 unit cells
    plot_E_vs_phi('Task3_Wb_0.1_5x4_4b.csv', 20, 'Task3_Wb_0.1_5x4_4b_improved.png')
    
    # Task 4: 5x4 = 20 unit cells
    plot_stiffness_vs_Wb('Task4_Stiffness_vs_Wb_5x4_4b.csv', 20, 'Task4_Stiffness_vs_Wb_improved.png')
    
    print("Plots regenerated successfully.")
