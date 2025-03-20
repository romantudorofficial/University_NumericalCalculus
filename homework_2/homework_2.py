'''
    Homework 2
'''
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
from tkinter import simpledialog

# ------------------ LU Decomposition Functions ------------------

def lu_decomposition_inplace(A, dU, eps):
    n = A.shape[0]
    for p in range(n):
        # Compute L[p, j] for j=0,...,p-1
        for j in range(p):
            sum_LU = 0.0
            for k in range(j):
                sum_LU += A[p, k] * A[k, j]
            if abs(dU[j]) < eps:
                raise ValueError(f"Zero pivot encountered in U at index {j}")
            A[p, j] = (A[p, j] - sum_LU) / dU[j]
        # Compute L[p, p] diagonal:
        sum_LU = 0.0
        for k in range(p):
            sum_LU += A[p, k] * A[k, p]
        A[p, p] = A[p, p] - sum_LU
        if abs(A[p, p]) < eps:
            raise ValueError(f"Zero pivot encountered in L at index {p}")
        # Compute U[p, j] for j = p+1,..., n-1:
        for j in range(p+1, n):
            sum_LU = 0.0
            for k in range(p):
                sum_LU += A[p, k] * A[k, j]
            A[p, j] = (A[p, j] - sum_LU) / A[p, p]
            
def forward_substitution(A, b):
    n = A.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += A[i, j] * y[j]
        y[i] = (b[i] - s) / A[i, i]
    return y

def backward_substitution(A, dU, y):
    n = A.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        s = 0.0
        for j in range(i+1, n):
            s += A[i, j] * x[j]
        if abs(dU[i]) < 1e-15:
            raise ValueError(f"Zero pivot encountered in U at index {i}")
        x[i] = (y[i] - s) / dU[i]
    return x

def compute_determinant(A, dU):
    n = A.shape[0]
    detL = 1.0
    for i in range(n):
        detL *= A[i, i]
    detU = np.prod(dU)
    return detL * detU

def run_inplace_version(A_init, b, dU, eps):
    n = A_init.shape[0]
    A = A_init.copy()  # copy to preserve original
    lu_decomposition_inplace(A, dU, eps)
    detA = compute_determinant(A, dU)
    y = forward_substitution(A, b)
    x_LU = backward_substitution(A, dU, y)
    res_norm = np.linalg.norm(A_init @ x_LU - b, 2)
    x_lib = np.linalg.solve(A_init, b)
    A_inv_lib = np.linalg.inv(A_init)
    diff1 = np.linalg.norm(x_LU - x_lib, 2)
    diff2 = np.linalg.norm(x_LU - A_inv_lib @ b, 2)
    result = (f"--- In-Place LU Decomposition ---\n"
              f"Determinant of A = {detA:.6f}\n"
              f"Residual norm ||A*x - b|| = {res_norm:.2e}\n"
              f"||x_LU - x_lib|| = {diff1:.2e}\n"
              f"||x_LU - A_inv_lib*b|| = {diff2:.2e}\n"
              f"Solution x (in-place):\n{x_LU}\n")
    return result

# ----- Bonus (Memory-Restricted) Functions -----

def idx_lower(i, j):
    return i*(i+1)//2 + j

def idx_upper(i, j, n):
    return (i * n - (i*(i-1))//2) + (j - i)

def lu_decomposition_bonus(A, dU, eps):
    n = A.shape[0]
    size = n*(n+1)//2
    L_vec = np.zeros(size)
    U_vec = np.zeros(size)
    for p in range(n):
        for j in range(p):
            sum_LU = 0.0
            for k in range(j):
                sum_LU += L_vec[idx_lower(p, k)] * U_vec[idx_upper(k, j, n)]
            if abs(dU[j]) < eps:
                raise ValueError(f"Zero pivot encountered in U at index {j}")
            L_val = (A[p, j] - sum_LU) / dU[j]
            L_vec[idx_lower(p, j)] = L_val
        sum_LU = 0.0
        for k in range(p):
            sum_LU += L_vec[idx_lower(p, k)] * U_vec[idx_upper(k, p, n)]
        L_diag = A[p, p] - sum_LU
        if abs(L_diag) < eps:
            raise ValueError(f"Zero pivot encountered in L at index {p}")
        L_vec[idx_lower(p, p)] = L_diag
        for j in range(p+1, n):
            sum_LU = 0.0
            for k in range(p):
                sum_LU += L_vec[idx_lower(p, k)] * U_vec[idx_upper(k, j, n)]
            U_val = (A[p, j] - sum_LU) / L_diag
            U_vec[idx_upper(p, j, n)] = U_val
        U_vec[idx_upper(p, p, n)] = dU[p]
    return L_vec, U_vec

def forward_substitution_bonus(L_vec, b, n):
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L_vec[idx_lower(i, j)] * y[j]
        y[i] = (b[i] - s) / L_vec[idx_lower(i, i)]
    return y

def backward_substitution_bonus(U_vec, dU, y, n):
    x = np.zeros(n)
    for i in reversed(range(n)):
        s = 0.0
        for j in range(i+1, n):
            s += U_vec[idx_upper(i, j, n)] * x[j]
        if abs(dU[i]) < 1e-15:
            raise ValueError(f"Zero pivot encountered in U at index {i}")
        x[i] = (y[i] - s) / dU[i]
    return x

def reconstruct_LU(L_vec, U_vec, dU, n):
    L_full = np.zeros((n, n))
    U_full = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            L_full[i, j] = L_vec[idx_lower(i, j)]
        for j in range(i, n):
            U_full[i, j] = dU[i] if i == j else U_vec[idx_upper(i, j, n)]
    LU = L_full @ U_full
    return LU

def run_bonus_version(A, b, dU, eps):
    n = A.shape[0]
    L_vec, U_vec = lu_decomposition_bonus(A, dU, eps)
    y = forward_substitution_bonus(L_vec, b, n)
    x_LU_bonus = backward_substitution_bonus(U_vec, dU, y, n)
    LU_prod = reconstruct_LU(L_vec, U_vec, dU, n)
    diff_norm = np.linalg.norm(A - LU_prod, 2)
    result = (f"--- Bonus LU (Memory-Restricted) ---\n"
              f"Solution x (bonus version):\n{x_LU_bonus}\n"
              f"Reconstructed LU product (should approximate A):\n{LU_prod}\n"
              f"||A - L*U|| = {diff_norm:.2e}\n")
    return result

# ------------------ Tkinter GUI ------------------

class LUApp:
    def __init__(self, master):
        self.master = master
        master.title("LU Decomposition and Linear System Solver")
        
        # Create a frame for options
        options_frame = ttk.LabelFrame(master, text="Options", padding=10)
        options_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # System type: Default, Manual, Random
        self.system_type = tk.StringVar(value="default")
        ttk.Label(options_frame, text="System Type:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(options_frame, text="Default", variable=self.system_type, value="default", command=self.toggle_system_input).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(options_frame, text="Manual", variable=self.system_type, value="manual", command=self.toggle_system_input).grid(row=0, column=2, sticky="w")
        ttk.Radiobutton(options_frame, text="Random", variable=self.system_type, value="random", command=self.toggle_system_input).grid(row=0, column=3, sticky="w")
        
        # Dimension n and precision eps:
        ttk.Label(options_frame, text="Dimension n:").grid(row=1, column=0, sticky="w")
        self.n_entry = ttk.Entry(options_frame, width=10)
        self.n_entry.grid(row=1, column=1, sticky="w")
        self.n_entry.insert(0, "3")
        
        ttk.Label(options_frame, text="Precision eps:").grid(row=1, column=2, sticky="w")
        self.eps_entry = ttk.Entry(options_frame, width=10)
        self.eps_entry.grid(row=1, column=3, sticky="w")
        self.eps_entry.insert(0, "1e-8")
        
        # LU method selection:
        ttk.Label(options_frame, text="LU Method:").grid(row=2, column=0, sticky="w")
        self.method = tk.StringVar(value="inplace")
        method_menu = ttk.OptionMenu(options_frame, self.method, "inplace", "inplace", "bonus")
        method_menu.grid(row=2, column=1, sticky="w")
        
        # Frame for matrix and vector input (only for manual)
        self.input_frame = ttk.LabelFrame(master, text="Input Matrix A and Vector b", padding=10)
        self.input_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        ttk.Label(self.input_frame, text="Matrix A (rows separated by newlines, entries by spaces):").grid(row=0, column=0, sticky="w")
        self.A_text = scrolledtext.ScrolledText(self.input_frame, width=50, height=5)
        self.A_text.grid(row=1, column=0, pady=5)
        
        ttk.Label(self.input_frame, text="Vector b (entries separated by spaces):").grid(row=2, column=0, sticky="w")
        self.b_entry = ttk.Entry(self.input_frame, width=50)
        self.b_entry.grid(row=3, column=0, pady=5)
        
        # Button to solve the system:
        self.solve_button = ttk.Button(master, text="Solve", command=self.solve)
        self.solve_button.grid(row=2, column=0, padx=10, pady=10)
        
        # Text area for displaying results:
        self.result_text = scrolledtext.ScrolledText(master, width=80, height=20)
        self.result_text.grid(row=3, column=0, padx=10, pady=10)
        
        self.toggle_system_input()  # initialize input state
        
    def toggle_system_input(self):
        """Show or hide the manual input fields based on system type."""
        sys_type = self.system_type.get()
        if sys_type == "manual":
            self.input_frame.grid()
        else:
            self.input_frame.grid_remove()
    
    def parse_manual_input(self, n):
        try:
            # Parse matrix A from text box
            A_str = self.A_text.get("1.0", tk.END).strip()
            rows = A_str.splitlines()
            if len(rows) != n:
                raise ValueError(f"Expected {n} rows in A, got {len(rows)}.")
            A = []
            for row in rows:
                entries = row.split()
                if len(entries) != n:
                    raise ValueError(f"Each row must have {n} entries.")
                A.append([float(x) for x in entries])
            A = np.array(A)
            # Parse vector b:
            b_str = self.b_entry.get().strip()
            b_entries = b_str.split()
            if len(b_entries) != n:
                raise ValueError(f"Vector b must have {n} entries.")
            b = np.array([float(x) for x in b_entries])
            return A, b
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return None, None

    def solve(self):
        try:
            n = int(self.n_entry.get())
            eps = float(self.eps_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for n and eps.")
            return
        
        sys_type = self.system_type.get()
        if sys_type == "default":
            # Default example from assignment
            A_init = np.array([[4.0, 2.0, 3.0],
                               [2.0, 7.0, 5.5],
                               [6.0, 3.0, 12.5]])
            b = np.array([21.6, 33.6, 51.6])
            dU = np.array([2.0, 3.0, 4.0])
        elif sys_type == "manual":
            A_init, b = self.parse_manual_input(n)
            if A_init is None:
                return
            # For manual input, let’s ask the user to input U's diagonal (dU) too:
            dU_str = tk.simpledialog.askstring("Input", f"Enter U's diagonal dU (space separated {n} values):")
            if not dU_str:
                messagebox.showerror("Input Error", "No dU provided.")
                return
            dU_entries = dU_str.split()
            if len(dU_entries) != n:
                messagebox.showerror("Input Error", f"Expected {n} values for dU.")
                return
            dU = np.array([float(x) for x in dU_entries])
        else:  # random
            A_init = np.random.rand(n, n) * 10
            b = np.random.rand(n) * 10
            dU = np.random.rand(n) * 10 + 1.0  # ensure nonzero
            self.result_text.insert(tk.END, "Randomly generated A:\n" + str(A_init) + "\n")
            self.result_text.insert(tk.END, "Randomly generated b:\n" + str(b) + "\n")
        
        method = self.method.get()
        self.result_text.delete("1.0", tk.END)
        try:
            if method == "inplace":
                result = run_inplace_version(A_init, b, dU, eps)
            else:
                result = run_bonus_version(A_init, b, dU, eps)
            self.result_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error during computation", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = LUApp(root)
    root.mainloop()
