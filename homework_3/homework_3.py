import math
import tkinter as tk
from tkinter import messagebox, scrolledtext

# --------------------------------------------
# Sparse Matrix Storage - Method 1
# Economic storage scheme as described:
#  - diag: a list for the diagonal elements
#  - rows: a list (for each row) of dictionaries mapping column indices (j) to nonzero values (for off-diagonals)
# --------------------------------------------
class SparseMatrix1:
    def __init__(self, n):
        self.n = n
        self.diag = [0.0] * n
        self.rows = [dict() for _ in range(n)]
    
    def add_entry(self, i, j, value):
        if i == j:
            self.diag[i] += value
        else:
            self.rows[i][j] = self.rows[i].get(j, 0.0) + value

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        n = int(lines[0].strip())
        mat = cls(n)
        for line in lines[1:]:
            if line.strip() == "":
                continue
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                value = float(parts[0].strip())
                i = int(parts[1].strip())
                j = int(parts[2].strip())
            except Exception as e:
                raise ValueError(f"Error parsing line '{line.strip()}': {e}")
            mat.add_entry(i, j, value)
        return mat

    def multiply_vector(self, x):
        n = self.n
        result = [0.0] * n
        for i in range(n):
            s = self.diag[i] * x[i]
            for j, aij in self.rows[i].items():
                s += aij * x[j]
            result[i] = s
        return result

    def add(self, other):
        if self.n != other.n:
            raise ValueError("Matrix sizes do not match")
        result = SparseMatrix1(self.n)
        for i in range(self.n):
            result.diag[i] = self.diag[i] + other.diag[i]
            keys = set(self.rows[i].keys()).union(other.rows[i].keys())
            for j in keys:
                result.rows[i][j] = self.rows[i].get(j, 0.0) + other.rows[i].get(j, 0.0)
        return result

    def equals(self, other, epsilon):
        if self.n != other.n:
            return False
        for i in range(self.n):
            if abs(self.diag[i] - other.diag[i]) >= epsilon:
                return False
            keys = set(self.rows[i].keys()).union(other.rows[i].keys())
            for j in keys:
                if abs(self.rows[i].get(j, 0.0) - other.rows[i].get(j, 0.0)) >= epsilon:
                    return False
        return True

# --------------------------------------------
# Sparse Matrix Storage - Method 2
# An alternative representation that simply stores nonzero entries
# in a dictionary with key (i, j).
# --------------------------------------------
class SparseMatrix2:
    def __init__(self, n):
        self.n = n
        self.entries = {}  # key: (i,j)

    def add_entry(self, i, j, value):
        self.entries[(i, j)] = self.entries.get((i, j), 0.0) + value

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        n = int(lines[0].strip())
        mat = cls(n)
        for line in lines[1:]:
            if line.strip() == "":
                continue
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                value = float(parts[0].strip())
                i = int(parts[1].strip())
                j = int(parts[2].strip())
            except Exception as e:
                raise ValueError(f"Error parsing line '{line.strip()}': {e}")
            mat.add_entry(i, j, value)
        return mat

    def multiply_vector(self, x):
        n = self.n
        result = [0.0] * n
        for (i, j), value in self.entries.items():
            result[i] += value * x[j]
        return result

    def add(self, other):
        if self.n != other.n:
            raise ValueError("Matrix sizes do not match")
        result = SparseMatrix2(self.n)
        for key, value in self.entries.items():
            result.entries[key] = value
        for key, value in other.entries.items():
            result.entries[key] = result.entries.get(key, 0.0) + value
        return result

    def equals(self, other, epsilon):
        if self.n != other.n:
            return False
        keys = set(self.entries.keys()).union(other.entries.keys())
        for key in keys:
            if abs(self.entries.get(key, 0.0) - other.entries.get(key, 0.0)) >= epsilon:
                return False
        return True

# --------------------------------------------
# Helper function to read vector file (for b_i.txt)
# Format: first line is n, then n lines with one number each.
# --------------------------------------------
def read_vector_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    vec = []
    for line in lines[1:]:
        if line.strip() == "":
            continue
        try:
            vec.append(float(line.strip()))
        except Exception as e:
            raise ValueError(f"Error parsing vector line '{line.strip()}': {e}")
    if len(vec) != n:
        vec = vec[:n]  # truncate if more lines than expected
    return n, vec

# --------------------------------------------
# Gauss-Seidel Iterative Solver
# Uses the sparse matrix stored in SparseMatrix1.
# Uses one vector x that is updated in place.
# --------------------------------------------
def gauss_seidel(sparse_matrix, b, epsilon, kmax=10000):
    n = sparse_matrix.n
    x = [0.0] * n
    iter_count = 0
    while iter_count < kmax:
        x_old = x.copy()
        for i in range(n):
            sum_val = 0.0
            # Sum over non-diagonal elements of row i
            for j, aij in sparse_matrix.rows[i].items():
                sum_val += aij * x[j]
            # Update using the Gauss-Seidel formula
            x[i] = (b[i] - sum_val) / sparse_matrix.diag[i]
        # Compute the infinity norm of (x - x_old)
        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        iter_count += 1
        if diff < epsilon:
            return x, iter_count
    return x, iter_count  # if kmax reached

# Compute the residual norm ||Ax - b||∞
def residual_norm(sparse_matrix, x, b):
    Ax = sparse_matrix.multiply_vector(x)
    res = [abs(Ax[i] - b[i]) for i in range(len(b))]
    return max(res)

# --------------------------------------------
# Function to solve a linear system from the given files a_i.txt and b_i.txt.
# Checks that the diagonal elements are nonzero.
# --------------------------------------------
def solve_system(system_index, epsilon_power):
    # File names: a_1.txt, b_1.txt, etc.
    file_a = f"a_{system_index}.txt"
    file_b = f"b_{system_index}.txt"
    try:
        A = SparseMatrix1.from_file(file_a)
        n_b, b = read_vector_from_file(file_b)
        if A.n != n_b:
            return f"Error: Dimension mismatch between matrix and vector in system {system_index}"
        # Check that diagonal elements are nonzero using tolerance ε = 10^(-p)
        eps = 10 ** (-epsilon_power)
        for i in range(A.n):
            if abs(A.diag[i]) < eps:
                return f"Error: Zero or near-zero diagonal element at row {i}"
        sol, iterations = gauss_seidel(A, b, eps)
        res_norm = residual_norm(A, sol, b)
        result_text = f"System {system_index} solved in {iterations} iterations.\n"
        result_text += "Solution x:\n" + "\n".join(f"  x[{i}] = {sol[i]}" for i in range(len(sol))) + "\n"
        result_text += f"Residual norm: {res_norm}"
        return result_text
    except Exception as e:
        return f"Error solving system {system_index}: {str(e)}"

# --------------------------------------------
# Bonus: Matrix Addition Verification
# Reads files a.txt, b.txt and aplusb.txt, computes A+B using both methods,
# and checks that the result is equal (within tolerance ε).
# --------------------------------------------
def bonus_matrix_addition(epsilon_power):
    eps = 10 ** (-epsilon_power)
    try:
        # Method 1:
        A1 = SparseMatrix1.from_file("a.txt")
        B1 = SparseMatrix1.from_file("b.txt")
        Sum1 = A1.add(B1)
        C1 = SparseMatrix1.from_file("aplusb.txt")
        eq1 = Sum1.equals(C1, eps)
        # Method 2:
        A2 = SparseMatrix2.from_file("a.txt")
        B2 = SparseMatrix2.from_file("b.txt")
        Sum2 = A2.add(B2)
        C2 = SparseMatrix2.from_file("aplusb.txt")
        eq2 = Sum2.equals(C2, eps)
        result_text = "Bonus Matrix Addition Verification:\n"
        result_text += "  Method 1: " + ("Passed" if eq1 else "Failed") + "\n"
        result_text += "  Method 2: " + ("Passed" if eq2 else "Failed") + "\n"
        return result_text
    except Exception as e:
        return f"Error in bonus matrix addition: {str(e)}"

# --------------------------------------------
# Tkinter GUI Setup
# The GUI lets the user choose a system number (1-5) and set the p value for ε = 10^-p.
# It has two buttons: one for solving the selected system and one for running the bonus verification.
# --------------------------------------------
def run_gui():
    window = tk.Tk()
    window.title("Numerical Calculus Homework 3")
    window.geometry("800x600")
    
    instruction = tk.Label(window, text="Select system number (1-5) and enter p (for ε = 10⁻ᵖ):")
    instruction.pack(pady=10)
    
    frame = tk.Frame(window)
    frame.pack(pady=5)
    
    tk.Label(frame, text="System number:").grid(row=0, column=0, padx=5, pady=5)
    system_entry = tk.Entry(frame, width=10)
    system_entry.grid(row=0, column=1, padx=5, pady=5)
    
    tk.Label(frame, text="p (for ε = 10⁻ᵖ):").grid(row=1, column=0, padx=5, pady=5)
    epsilon_entry = tk.Entry(frame, width=10)
    epsilon_entry.grid(row=1, column=1, padx=5, pady=5)
    
    result_area = scrolledtext.ScrolledText(window, width=90, height=25)
    result_area.pack(pady=10)
    
    def solve_button_action():
        try:
            sys_num = int(system_entry.get())
            p_val = int(epsilon_entry.get())
            result = solve_system(sys_num, p_val)
            result_area.delete(1.0, tk.END)
            result_area.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def bonus_button_action():
        try:
            p_val = int(epsilon_entry.get())
            result = bonus_matrix_addition(p_val)
            result_area.delete(1.0, tk.END)
            result_area.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    solve_button = tk.Button(window, text="Solve System", command=solve_button_action, width=20)
    solve_button.pack(pady=5)
    
    bonus_button = tk.Button(window, text="Bonus: Matrix Addition", command=bonus_button_action, width=20)
    bonus_button.pack(pady=5)
    
    window.mainloop()

# --------------------------------------------
# Main execution: run the GUI.
# --------------------------------------------
if __name__ == "__main__":
    run_gui()