import numpy as np
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox

# Polynomial evaluation using Horner's scheme: returns P, P', P''
def horner_all(coeffs, x):
    P = coeffs[0]
    P1 = 0.0
    P2 = 0.0
    for a in coeffs[1:]:
        P2 = P2 * x + 2 * P1
        P1 = P1 * x + P
        P = P * x + a
    return P, P1, P2

# Halley method for polynomial roots
def halley_poly(coeffs, epsilon, kmax, x0):
    x = x0
    for _ in range(kmax):
        P, P1, P2 = horner_all(coeffs, x)
        denom = 2 * P1 * P1 - P * P2
        if abs(denom) < epsilon:
            return None
        delta = 2 * P * P1 / denom
        x_new = x - delta
        if abs(delta) < epsilon:
            return x_new
        x = x_new
    return None

# Newton-type fourth-order (N4) using paper's formula
def newton_type_N4_poly(coeffs, epsilon, kmax, x0):
    x = x0
    for _ in range(kmax):
        P, P1, _ = horner_all(coeffs, x)
        if abs(P1) < epsilon:
            return None
        # step 1
        y = x - P / P1
        P_y, _, _ = horner_all(coeffs, y)
        num = P * P + P * P_y
        den = P1 * (P - P_y)
        if abs(den) < epsilon:
            return None
        x_new = x - num / den
        if abs(x_new - x) < epsilon:
            return x_new
        x = x_new
    return None

# Newton-type fifth-order (N5) using paper's formula
def newton_type_N5_poly(coeffs, epsilon, kmax, x0):
    x = x0
    for _ in range(kmax):
        P, P1, _ = horner_all(coeffs, x)
        if abs(P1) < epsilon:
            return None
        # step 1
        y = x - P / P1
        P_y, _, _ = horner_all(coeffs, y)
        num = P * P + P * P_y
        den = P1 * (P - P_y)
        if abs(den) < epsilon:
            return None
        # step 2
        z = x - num / (2 * den)
        P_z, _, _ = horner_all(coeffs, z)
        # final update
        x_new = z - P_z / P1
        if abs(x_new - x) < epsilon:
            return x_new
        x = x_new
    return None

# Generic root-finding across methods
def find_real_roots(coeffs, method_func, p, kmax, samples=100):
    a0 = coeffs[0]
    A = max(abs(a) for a in coeffs[1:])
    R = (abs(a0) + A) / abs(a0)
    epsilon = 10**(-p)
    roots = []
    for x0 in np.linspace(-R, R, samples):
        root = method_func(coeffs, epsilon, kmax, x0)
        if root is not None and -R <= root <= R:
            if not any(abs(root - r) < epsilon for r in roots):
                roots.append(root)
    return R, sorted(roots)

# GUI Application
def run_app():
    root = tk.Tk()
    root.title("Polynomial Root Finder")

    # Inputs
    tk.Label(root, text="Coefficients (a0,a1,...,an):").grid(row=0, column=0, sticky='w')
    coeff_entry = tk.Entry(root, width=50)
    coeff_entry.grid(row=0, column=1)

    tk.Label(root, text="Precision p (ε=10^-p):").grid(row=1, column=0, sticky='w')
    p_entry = tk.Entry(root)
    p_entry.grid(row=1, column=1, sticky='w')

    tk.Label(root, text="Max iterations kmax:").grid(row=2, column=0, sticky='w')
    kmax_entry = tk.Entry(root)
    kmax_entry.grid(row=2, column=1, sticky='w')

    output = scrolledtext.ScrolledText(root, width=60, height=20)
    output.grid(row=4, column=0, columnspan=3)

    # Compute handlers
    def compute(method_name):
        try:
            coeffs = list(map(float, coeff_entry.get().split(',')))
            p = int(p_entry.get())
            kmax = int(kmax_entry.get())
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return
        func = {'Halley': halley_poly,
                'N4': newton_type_N4_poly,
                'N5': newton_type_N5_poly}[method_name]
        R, roots = find_real_roots(coeffs, func, p, kmax)
        output.delete('1.0', tk.END)
        output.insert(tk.END, f"Method: {method_name}\nInterval [-R,R]: [{-R:.6f}, {R:.6f}]\n")
        output.insert(tk.END, "Roots found:\n")
        for r in roots:
            output.insert(tk.END, f"{r:.{p}f}\n")
        if messagebox.askyesno("Save", "Save roots to file? (Yes/No)"):
            path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text files','*.txt')])
            if path:
                with open(path, 'w') as f:
                    for r in roots:
                        f.write(f"{r:.{p}f}\n")
                messagebox.showinfo("Saved", f"Roots saved to {path}")

    # Buttons
    tk.Button(root, text="Compute (Halley)", command=lambda: compute('Halley')).grid(row=3, column=0)
    tk.Button(root, text="Compute (N4)", command=lambda: compute('N4')).grid(row=3, column=1)
    tk.Button(root, text="Compute (N5)", command=lambda: compute('N5')).grid(row=3, column=2)

    root.mainloop()

if __name__ == '__main__':
    run_app()
