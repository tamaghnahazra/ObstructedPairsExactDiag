import numpy as np
import matplotlib.pyplot as plt
from quspin.basis import boson_basis_general
from quspin.operators import hamiltonian
from scipy.optimize import curve_fit
import os
import time

def get_site(x, y, sub, Lx, Ly):
    return 2 * ((x % Lx) + (y % Ly) * Lx) + sub

def compute_E_vs_phi(Lx, Ly, N_bosons, W_b_ratio, phi_vals):
    N_sites = 2 * Lx * Ly
    
    # y-translation operator for kyblock
    y_t = np.zeros(N_sites, int)
    for x in range(Lx):
        for y in range(Ly):
            for s in range(2):
                y_t[get_site(x, y, s, Lx, Ly)] = get_site(x, y+1, s, Lx, Ly)
                
    basis = boson_basis_general(N_sites, sps=2, Nb=N_bosons, kyblock=(y_t, 0))
    print(f"Lx={Lx}, Ly={Ly}, Nb={N_bosons}, W_b={W_b_ratio}t_b | Basis size: {basis.Ns}")

    t_up = 1.0
    t_dn = -1.0
    U = 500.0
    t_b = -2.0 * t_up * t_dn / U    
    V_b = 2.0 * (t_up**2 + t_dn**2) / U  
    
    W_b = W_b_ratio * t_b
    t_b_intra = t_b - W_b / 4.0
    t_b_inter = t_b

    pot_list = [[-3.0 * V_b, i] for i in range(N_sites)]
    energies = []

    for phi in phi_vals:
        hop_list = []
        int_list = []
        unique_bonds = set()
        
        def add_directed_bond(x1, y1, sub1, x2, y2, sub2, is_intra):
            wx1, wy1 = x1 % Lx, y1 % Ly
            wx2, wy2 = x2 % Lx, y2 % Ly
            s1 = get_site(wx1, wy1, sub1, Lx, Ly)
            s2 = get_site(wx2, wy2, sub2, Lx, Ly)
            if s1 == s2: return
            
            b_id = tuple(sorted((s1, s2)))
            if b_id in unique_bonds: return
            unique_bonds.add(b_id)
            
            phase = 1.0
            # Apply twist ONLY to x-boundary bonds
            if wx1 == Lx - 1 and wx2 == 0 and (x1 - x2 == -1):
                phase = np.exp(-1j * phi)
            elif wx1 == 0 and wx2 == Lx - 1 and (x1 - x2 == 1):
                phase = np.exp(1j * phi)
                
            amp = t_b_intra if is_intra else t_b_inter
            hop_list.append([amp * phase, s1, s2])
            hop_list.append([amp * np.conj(phase), s2, s1])
            int_list.append([V_b, s1, s2])

        for x in range(Lx):
            for y in range(Ly):
                add_directed_bond(x, y, 0, x+1, y, 0, True)
                add_directed_bond(x, y, 0, x-1, y, 0, True)
                add_directed_bond(x, y, 1, x, y+1, 1, True)
                add_directed_bond(x, y, 1, x, y-1, 1, True)
                add_directed_bond(x, y, 0, x, y, 1, False)
                add_directed_bond(x, y, 0, x+1, y, 1, False)
                add_directed_bond(x, y, 0, x, y-1, 1, False)
                add_directed_bond(x, y, 0, x+1, y-1, 1, False)

        static_terms = [["+-", hop_list], ["n", pot_list], ["nn", int_list]]
        H = hamiltonian(static_terms, [], basis=basis, dtype=np.complex128, check_herm=False)
        E_ground = H.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
        shift = len(unique_bonds) * V_b / 4.0
        energies.append(E_ground + shift)
        
    return np.array(energies)

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def run_task_1():
    print("--- Task 1: E vs phi at W_b=0, 5x4 lattice 4 bosons ---")
    phi_vals = np.linspace(-0.2, 0.2, 11)
    energies = compute_E_vs_phi(5, 4, 4, 0.0, phi_vals)
    popt, _ = curve_fit(parabola, phi_vals, energies)
    Ds = 2 * popt[0]
    
    plt.figure(figsize=(6,5))
    plt.plot(phi_vals, energies, 'o', label='ED Data')
    phi_smooth = np.linspace(-0.2, 0.2, 100)
    plt.plot(phi_smooth, parabola(phi_smooth, *popt), '-', label=f'Fit $D_s={Ds:.2e}$')
    plt.xlabel('Twist angle $\phi$')
    plt.ylabel('Energy')
    plt.title('5x4, 4 bosons, $W_b=0$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.savefig('Task1_Wb_0_5x4_4b.png', dpi=300)
    plt.close()
    
    data = np.column_stack((phi_vals, energies))
    np.savetxt('Task1_Wb_0_5x4_4b.csv', data, delimiter=',', header='phi,E', comments='')

def run_task_2():
    print("--- Task 2: E vs phi at W_b=0, 4x4 lattice 8 bosons ---")
    phi_vals = np.linspace(-0.2, 0.2, 11)
    energies = compute_E_vs_phi(4, 4, 8, 0.0, phi_vals)
    popt, _ = curve_fit(parabola, phi_vals, energies)
    Ds = 2 * popt[0]
    
    plt.figure(figsize=(6,5))
    plt.plot(phi_vals, energies, 'o', label='ED Data')
    phi_smooth = np.linspace(-0.2, 0.2, 100)
    plt.plot(phi_smooth, parabola(phi_smooth, *popt), '-', label=f'Fit $D_s={Ds:.2e}$')
    plt.xlabel('Twist angle $\phi$')
    plt.ylabel('Energy')
    plt.title('4x4, 8 bosons, $W_b=0$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.savefig('Task2_Wb_0_4x4_8b.png', dpi=300)
    plt.close()
    
    data = np.column_stack((phi_vals, energies))
    np.savetxt('Task2_Wb_0_4x4_8b.csv', data, delimiter=',', header='phi,E', comments='')

def run_task_3():
    print("--- Task 3: E vs phi at W_b=0.1 t_b, 5x4 lattice 4 bosons ---")
    phi_vals = np.linspace(-0.2, 0.2, 11)
    energies = compute_E_vs_phi(5, 4, 4, 0.1, phi_vals)
    popt, _ = curve_fit(parabola, phi_vals, energies)
    Ds = 2 * popt[0]
    
    plt.figure(figsize=(6,5))
    plt.plot(phi_vals, energies, 'o', label='ED Data')
    phi_smooth = np.linspace(-0.2, 0.2, 100)
    plt.plot(phi_smooth, parabola(phi_smooth, *popt), '-', label=f'Fit $D_s={Ds:.2e}$')
    plt.xlabel('Twist angle $\phi$')
    plt.ylabel('Energy')
    plt.title('5x4, 4 bosons, $W_b=0.1 t_b$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.savefig('Task3_Wb_0.1_5x4_4b.png', dpi=300)
    plt.close()
    
    data = np.column_stack((phi_vals, energies))
    np.savetxt('Task3_Wb_0.1_5x4_4b.csv', data, delimiter=',', header='phi,E', comments='')

def run_task_4():
    print("--- Task 4: Stiffness vs W_b for 0.0 to 0.5 step 0.1, 5x4 lattice 4 bosons ---")
    W_b_vals = np.arange(0.0, 0.6, 0.1)
    phi_vals = np.linspace(-0.2, 0.2, 11)
    stiffness_vals = []
    
    for W_b in W_b_vals:
        energies = compute_E_vs_phi(5, 4, 4, W_b, phi_vals)
        popt, _ = curve_fit(parabola, phi_vals, energies)
        Ds = 2 * popt[0]
        stiffness_vals.append(Ds)
        print(f"  W_b={W_b:.1f}t_b -> Ds={Ds:.3e}")
        
    plt.figure(figsize=(6,5))
    plt.plot(W_b_vals, stiffness_vals, 's-', color='purple')
    plt.xlabel('$W_b / t_b$')
    plt.ylabel('Stiffness $D_s$')
    plt.title('Stiffness vs Bandwidth (5x4, 4 bosons)')
    plt.grid(True, alpha=0.3)
    plt.savefig('Task4_Stiffness_vs_Wb.png', dpi=300)
    plt.close()
    
    data = np.column_stack((W_b_vals, stiffness_vals))
    np.savetxt('Task4_Stiffness_vs_Wb_5x4_4b.csv', data, delimiter=',', header='W_b,Ds', comments='')

if __name__ == "__main__":
    t0 = time.time()
    run_task_1()
    run_task_2()
    run_task_3()
    run_task_4()
    print(f"All tasks completed in {time.time()-t0:.1f} seconds.")
