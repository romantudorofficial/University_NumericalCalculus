'''
    Homework 5
    Name: Roman Tudor
    Student ID: 310910401ESL201031
    Email Address: romantudor.contact@gmail.com
    Discord Username: romantudorofficial
    Bibliography: ChatGPT, Lecture Notes
    LLM Percentile: 40%
'''

import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from scipy import linalg



class SparseMatrix:

    '''
        Class for storing a rare matrix in a sparse format.
    '''

    def __init__ (self, n):

        '''
            Initialize the sparse matrix with n rows and columns.
            Input:
                - n: size of the matrix (number of rows and columns)
            Output:
                - None
        '''

        # Initialize the sparse matrix with n rows and columns.
        self.n = n

        # Initialize the diagonal and row dictionaries.
        self.d = np.zeros(n)
        self.rows = [dict() for _ in range(n)]


    def add (self, i, j, v):

        '''
            Add a value v to the matrix at position (i, j).
            Input:
                - i: row index
                - j: column index
                - v: value to be added
            Output:
                - None
        '''

        if i == j:
            self.d[i] = v
        else:
            self.rows[i][j] = v
            self.rows[j][i] = v


    def matvec (self, x):

        '''
            Multiply the sparse matrix by a vector x.
            Input:
                - x: vector to be multiplied
            Output:
                - y: result of the multiplication
        '''

        y = self.d * x

        for i,row in enumerate(self.rows):
            for j,v in row.items():
                y[i] += v * x[j]

        return y


    def to_dense (self):

        '''
            Convert the sparse matrix to a dense format.
            Input:
                - None
            Output:
                - A: dense matrix (numpy array)
        '''

        A = np.diag(self.d)
        
        for i,row in enumerate(self.rows):
            for j,v in row.items():
                A[i,j] = v
        
        return A



def read_sparse (filename):

    '''
        Read a sparse matrix from a file.
        Input:
            - filename: name of the file containing the sparse matrix
        Output:
            - M: the sparse matrix (SparseMatrix object)
    '''

    lines = [L.strip() for L in open(filename) if L.strip()]
    n = int(lines[0])
    M = SparseMatrix(n)
    
    for L in lines[1:]:
        parts = [p.strip() for p in L.split(',')]
        if len(parts)==3:
            v,i,j = float(parts[0]), int(parts[1]), int(parts[2])
            M.add(i,j,v)
    
    return M



def is_symmetric (M, tol = 1e-12):

    '''
        Check if a sparse matrix M is symmetric.
        Input:
            - M: matrix (SparseMatrix object)
            - tol: tolerance for symmetry (default = 1e-12)
        Output:
            - True if M is symmetric, False otherwise
    '''

    for i,row in enumerate(M.rows):
        for j,v in row.items():
            if abs(v - M.rows[j].get(i,0.0)) > tol:
                return False
    
    return True



def power_method (M, eps, kmax = 10**6):

    '''
        Power method to find the largest eigenvalue of a matrix M.
        Input:
            - M: matrix (SparseMatrix object)
            - eps: tolerance for convergence
            - kmax: maximum number of iterations (default = 10^6)
        Output:
            - lam: largest eigenvalue of M
            - res: residual of the power method
    '''

    n = M.n
    x = np.random.randn(n); x /= np.linalg.norm(x)
    w = M.matvec(x)
    lam = x.dot(w)
    
    for _ in range(kmax):
        x = w/np.linalg.norm(w)
        w = M.matvec(x)
        lam_new = x.dot(w)
        if np.linalg.norm(w - lam*x) <= n*eps:
            lam = lam_new
            break
        lam = lam_new
    else:
        raise RuntimeError("Power method did not converge")
    
    res = np.linalg.norm(M.matvec(x) - lam*x)
    
    return lam, res



def generate_random_sparse (n, density = 0.01):

    '''
        Generate a random sparse matrix of size n x n with a given density.
        Input:
            - n: size of the matrix
            - density: density of the matrix (default = 0.01)
        Output:
            - M: the generated sparse matrix
    '''

    M = SparseMatrix(n)
    
    for i in range(n):
        M.add(i, i, np.random.rand())
    
    k = max(1, int(density * n))
    
    for i in range(n):
        js = np.random.choice([j for j in range(n) if j != i], k, replace = False)
        for j in js:
            M.add(i, j, np.random.rand())
    
    return M



def svd_analysis (A, b, eps):

    '''
        Perform SVD analysis on the matrix A and vector b.
        Input:
            - A: matrix (numpy array)
            - b: vector (numpy array)
            - eps: tolerance for singular values
        Output:
            - s: singular values of A
            - rank: rank of A
            - cond: condition number of A
            - xI: least-squares solution of Ax = b
            - res2: residual of the least-squares solution
    '''
    
    # Get the SVD of A.
    U, s, VT = linalg.svd(A, full_matrices = False)

    # Get the rank of A.
    rank = np.sum(s > eps)

    # Get the condition number of A.
    cond = s[0] / s[rank-1] if rank > 0 else np.inf

    # Get the pseudo-inverse of A.
    Splus = np.diag([1 / si if si > eps else 0.0 for si in s])
    Aplus = VT.T @ Splus @ U.T

    # Get the least-squares solution xI.
    xI = Aplus @ b
    res2 = np.linalg.norm(b - A @ xI)

    # Return the singular values, rank, condition number, xI, and residual.    
    return s, rank, cond, xI, res2



class App:

    '''
        Class for the GUI application.
    '''

    def __init__ (self, root):

        '''
            Initializes the GUI application.
        '''

        # Set the title.
        root.title("Homework 5")

        # Set the main frame.
        frame = tk.Frame(root)
        frame.pack(padx = 5, pady = 5, anchor = "w")

        # Input entries
        for idx, (lbl, w) in enumerate ([("p", 6), ("n", 6), ("ε", 8)]):

            tk.Label(frame, text = lbl + ":").grid(row = 0, column = 2 * idx)
            ent = tk.Entry(frame, width = w)
            ent.grid(row = 0, column = 2 * idx + 1)
            setattr(self, f"ent_{lbl}", ent)

        # Set the load file button.
        tk.Button(frame, text = "Load File", command = self.load_file)\
          .grid(row = 1, column = 0, columnspan = 2, pady = 4)
        
        # Set the run button.
        tk.Button(frame, text = "Run", command = self.run_all, bg = "#aaffaa")\
          .grid(row = 1, column = 2, columnspan = 4, padx = 10, pady = 4)

        # Set the output text area.
        self.txt = scrolledtext.ScrolledText(root, width = 90, height = 25)
        self.txt.pack(padx = 5, pady = 5)

        self.M_file = None
        self.A_rand = None


    def load_file (self):

        '''
            Load a sparse matrix from a file.
        '''

        f = filedialog.askopenfilename(title="Select sparse A_file (.txt)")

        if not f:
            return
        
        try:
            M = read_sparse(f)
            self.M_file = M
            self.txt.insert(tk.END, f"\nLoaded File (n={M.n}) from {f}\n\n")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))


    def run_all (self):

        '''
            Run all the calculations and display the results.
        '''

        # Clear the text area.
        self.txt.delete('1.0', tk.END)

        # Get the input values.
        try:
            p = int(self.ent_p.get())
            n = int(self.ent_n.get())
            eps = float(self.ent_ε.get())
        except:
            messagebox.showerror("Input Error", "Enter valid p, n, ε.")
            return

        self.txt.insert(tk.END, "\n=== Run Results ===\n\n")

        # Get the results for the first requirement.
        self.txt.insert(tk.END, "Requirement 1\n\n")
        self.txt.insert(tk.END, f"p = {p},  n = {n},  ε = {eps}\n\n")
        A_r = generate_random_sparse(p)
        self.A_rand = A_r
        self.txt.insert(tk.END, "Generated A_rand (dense):\n")
        self.txt.insert(tk.END, np.array2string(A_r.to_dense(), precision=4) + "\n\n")
        if self.M_file:
            Mf = self.M_file
            self.txt.insert(tk.END, "Sparse storage of A_file:\n")
            self.txt.insert(tk.END, f"  d = {np.array2string(Mf.d,precision=4)}\n")
            for i,row in enumerate(Mf.rows):
                if row:
                    self.txt.insert(tk.END, f"  row {i}: {row}\n")
            self.txt.insert(tk.END, "\n")
        else:
            self.txt.insert(tk.END, "A_file not loaded → file storage N/A\n\n")

        # Get the results for the second requirement.
        self.txt.insert(tk.END, "Requirement 2\n\n")
        lam_r, res_r = power_method(A_r, eps)
        self.txt.insert(tk.END, f"A_rand →    λ_max = {lam_r:.6e},   residual = {res_r:.6e}\n")
        if self.M_file:
            Mf = self.M_file
            sym = is_symmetric(Mf)
            self.txt.insert(tk.END, f"A_file symmetric? {'Yes' if sym else 'No'}\n")
            if sym:
                lam_f, res_f = power_method(Mf, eps)
                self.txt.insert(tk.END, f"A_file →    λ_max = {lam_f:.6e},   residual = {res_f:.6e}\n\n")
            else:
                self.txt.insert(tk.END, "Power Method on A_file N/A (non-symmetric)\n\n")
        else:
            self.txt.insert(tk.END, "A_file not loaded → Req2 file N/A\n\n")

        # Get the results for the third requirement.
        self.txt.insert(tk.END, "Requirement 3\n\n")
        b = np.random.randn(p)
        s, rank, cond, xI, res2 = svd_analysis(A_r.to_dense(), b, eps)
        self.txt.insert(tk.END,
            f"Singular values:    {np.array2string(s,precision=4)}\n"
            f"Rank(A_rand) =      {rank}\n"
            f"Condition # =       {cond:.6e}\n"
            f"First 10 entries of x_I: {np.array2string(xI[:min(10,len(xI))],precision=4)}\n"
            f"‖b - A_rand x_I‖₂ =  {res2:.6e}\n\n"
        )



def main ():

    '''
        Main function to run the GUI application.
    '''

    root = tk.Tk()
 
    App(root)

    root.mainloop()



if __name__ == "__main__":

    '''
        Main function to run the program.
    '''

    main()