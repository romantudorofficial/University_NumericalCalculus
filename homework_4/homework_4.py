import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np



def norm_mat (M, norm_type = 1):

    '''
        Compute the matrix norm (default 1-norm).
    '''

    return np.linalg.norm(M, ord = norm_type)



def initial_guess_square (A, method = 5):

    '''
        Returns an initial guess for the inverse of square matrix A.
        method == 5: Formula (5): V0 = A^T / (||A||_∞^2)
        method == 6: Formula (6): V0 = A^T / (||A||_∞ * ||A||_1)
    '''

    norm_inf = norm_mat(A, norm_type = np.inf)
    norm_1 = norm_mat(A, norm_type = 1)

    # If the initial guess method is 5, use the fifth formula.
    if method == 5:
        return A.T / (norm_inf**2)
    
    # If the initial guess method is 6, use the sixth formula.
    elif method == 6:
        return A.T / (norm_inf * norm_1)
    
    # If the initial guess method is neither 5, nor 6, raise an error.
    else:
        raise ValueError("\nThe initial guess method must be either 5 or 6.\n")



def iterative_inverse_newton (A, eps = 1e-6, kmax = 10000, init_method = 5):

    '''
        Method 1: Newton-Schultz (Hotelling-Bodewig):
        V_{k+1} = V_k*(2I - A*V_k)
    '''

    n = A.shape[0]
    V = initial_guess_square(A, method = init_method)
    I = np.eye(n)
    iter_count = 0

    while iter_count < kmax:

        V_old = V.copy()
        V = V_old @ (2 * I - A @ V_old)
        diff = norm_mat(V - V_old)
        iter_count += 1

        if diff < eps:
            break
        
        if diff > 1e10:
            return V, iter_count, None, "Divergence detected in Method 1!"
        
    res_norm = norm_mat(A @ V - I)

    return V, iter_count, res_norm, None



def iterative_inverse_cubic (A, eps = 1e-6, kmax = 10000, init_method = 5):

    '''
        Method 2: Cubic convergence iteration:
        V_{k+1} = (1/3)*V_k*(I + (I-A*V_k) + (I-A*V_k)^2)
    '''

    n = A.shape[0]
    V = initial_guess_square(A, method = init_method)
    I = np.eye(n)
    iter_count = 0

    while iter_count < kmax:

        V_old = V.copy()
        E = I - A @ V_old
        V = (1/3) * (V_old @ (I + E + E @ E))
        diff = norm_mat(V - V_old)
        iter_count += 1

        if diff < eps:
            break

        if diff > 1e10:
            return V, iter_count, None, "Divergence detected in Method 2!"
        
    res_norm = norm_mat(A @ V - I)

    return V, iter_count, res_norm, None



def iterative_inverse_avg (A, eps = 1e-6, kmax = 10000, init_method = 5):
    
    '''
        Method 3: Averaged Newton iteration:
        V_{k+1} = 0.5*(V_k + V_k*(2I - A*V_k))
    '''

    # This method is similar to the Newton-Schultz method, but it averages the current and next estimates.
    n = A.shape[0]
    V = initial_guess_square(A, method = init_method)
    I = np.eye(n)
    iter_count = 0

    while iter_count < kmax:

        V_old = V.copy()
        V = 0.5 * (V_old + V_old @ (2 * I - A @ V_old))
        diff = norm_mat(V - V_old)
        iter_count += 1

        if diff < eps:
            break

        if diff > 1e10:
            return V, iter_count, None, "Divergence detected in Method 3!"
    
    # Compute the residual norm ||A*V - I||.
    res_norm = norm_mat(A @ V - I)

    return V, iter_count, res_norm, None



def special_matrix (n):

    '''
        Constructs the n x n matrix A with diagonal entries 1 and superdiagonal entries 2.
    '''

    # Initialize the matrix A with zeros. ??
    A = np.eye(n)

    # Set the superdiagonal entries to 2.
    for i in range(n - 1):
        A[i, i + 1] = 2

    # Get the matrix.
    return A



def exact_inverse_special (n):

    '''
        Computes the exact inverse of the special matrix:
        For 1 <= i <= j <= n, A^{-1}(i,j)= (-2)^(j-i) and 0 if j < i.
    '''

    invA = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            invA[i, j] = (-2) ** (j - i)

    return invA



def iterative_pseudoinverse (A, eps = 1e-6, kmax = 10000):
    
    '''
        Approximates the left-inverse X (n x m) for a matrix A (m x n) with full column rank
        so that X*A ≈ I_n using the iteration:
        X_{k+1} = (2I_n - X_k*A)*X_k,
        with initial guess: X_0 = A^T/(||A||_1*||A||_∞).
    '''

    m, n = A.shape
    I_n = np.eye(n)
    norm1 = norm_mat(A, norm_type=1)
    norm_inf = norm_mat(A, norm_type=np.inf)
    X = A.T / (norm1 * norm_inf)
    iter_count = 0

    while iter_count < kmax:

        X_old = X.copy()
        X = (2 * I_n - X_old @ A) @ X_old
        iter_count += 1

        if norm_mat(I_n - X @ A) < eps:
            break

        if norm_mat(X - X_old) > 1e10:
            return X, iter_count, None, "Divergence detected in Pseudoinverse method!"
        
    res_norm = norm_mat(I_n - X @ A)

    return X, iter_count, res_norm, None



class Application (tk.Tk):

    '''
        Runs the application.
    '''

    def __init__ (self):

        '''
            Initializes the application.
        '''

        # Initialize the main window.
        super().__init__()
        self.title("Homework 4")
        self.geometry("1000x800")
        self.resizable(False, False)
        self.create_widgets()
    

    def create_widgets (self):

        '''
            Creates the widgets for the application.
        '''

        # Create a notebook for tabbed interface.
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill = tk.BOTH, expand = True)

        # Create the tab for the compulsory task.
        self.tab_square = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_square, text = "Compulsory")
        self.create_square_tab(self.tab_square)

        # Create the tab for the bonus task.
        self.tab_rect = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_rect, text = "Bonus")
        self.create_rect_tab(self.tab_rect)


    def create_square_tab (self, frame):

        '''
            Creates the widgets for the compulsory tab.
        '''

        # Get the input parameters for the compulsory task.
        param_frame = ttk.LabelFrame(frame, text = "Input Parameters")
        param_frame.pack(padx = 10, pady = 10, fill = tk.X)

        # Get the dimension.
        ttk.Label(param_frame, text = "Dimension (n):").grid(row = 0, column = 0, padx = 5, pady = 5, sticky = tk.E)
        self.n_entry = ttk.Entry(param_frame, width = 10)
        self.n_entry.grid(row = 0, column = 1, padx = 5, pady = 5)
        self.n_entry.insert(0, "4")

        # Get the tolerance.
        ttk.Label(param_frame, text = "Tolerance (ε):").grid(row = 1, column = 0, padx = 5, pady = 5, sticky = tk.E)
        self.eps_entry = ttk.Entry(param_frame, width = 10)
        self.eps_entry.grid(row = 1, column = 1, padx = 5, pady = 5)
        self.eps_entry.insert(0, "1e-6")

        # Get the max iterations.
        ttk.Label(param_frame, text = "Max Iterations (kmax):").grid(row = 2, column = 0, padx = 5, pady = 5, sticky = tk.E)
        self.kmax_entry = ttk.Entry(param_frame, width = 10)
        self.kmax_entry.grid(row = 2, column = 1, padx = 5, pady = 5)
        self.kmax_entry.insert(0, "10000")

        # Get the matrix source options.
        source_frame = ttk.LabelFrame(frame, text = "Matrix Source for A")
        source_frame.pack(padx = 10, pady = 10, fill = tk.X)
        self.square_matrix_source_var = tk.StringVar()
        self.square_matrix_source_var.set("Special")

        # Select the matrix source.
        ttk.Radiobutton(source_frame, text = "Special Matrix (Task 3)", variable = self.square_matrix_source_var,
                        value = "Special", command = self.toggle_manual_square).grid(row = 0, column = 0, padx = 5, pady = 5, sticky = tk.W)
        ttk.Radiobutton(source_frame, text = "Random Matrix", variable = self.square_matrix_source_var,
                        value = "Random", command = self.toggle_manual_square).grid(row = 0, column = 1, padx = 5, pady = 5, sticky = tk.W)
        ttk.Radiobutton(source_frame, text = "Manual Matrix Input", variable = self.square_matrix_source_var,
                        value = "Manual", command = self.toggle_manual_square).grid(row = 0, column = 2, padx = 5, pady = 5, sticky = tk.W)
        
        # Manual input text for square matrix (disabled by default).
        self.manual_square_text = scrolledtext.ScrolledText(source_frame, width = 60, height = 5, state = tk.DISABLED)
        self.manual_square_text.grid(row = 1, column = 0, columnspan = 3, padx = 5, pady = 5)
        self.manual_square_text.insert(tk.END, "Enter matrix rows with numbers separated by spaces, one row per line.\nExample for 3x3:\n1 2 3\n4 5 6\n7 8 9")
        
        # Button to run iterative inversions for square matrix.
        run_btn = ttk.Button(frame, text = "Run Square Matrix Inversion", command = self.run_square)
        run_btn.pack(padx = 10, pady = 5)
        
        # Create the output area.
        self.square_output = scrolledtext.ScrolledText(frame, width = 100, height = 20)
        self.square_output.pack(padx = 10, pady = 10)


    def toggle_manual_square (self):

        '''
            Enable manual matrix text field only if 'Manual' is selected for square matrix.
        '''

        if self.square_matrix_source_var.get() == "Manual":
            self.manual_square_text.config(state=tk.NORMAL)
        else:
            self.manual_square_text.config(state=tk.DISABLED)
    

    def create_rect_tab (self, frame):

        '''
            Creates the widgets for the bonus tab.
        '''

        # Get the input parameters for the bonus task.
        param_frame = ttk.LabelFrame(frame, text = "Input Parameters for Non-Square Matrix")
        param_frame.pack(padx = 10, pady = 10, fill = tk.X)

        # Get the number of rows (m).
        ttk.Label(param_frame, text = "Rows (m):").grid(row = 0, column = 0, padx = 5, pady = 5, sticky = tk.E)
        self.m_entry = ttk.Entry(param_frame, width = 10)
        self.m_entry.grid(row = 0, column = 1, padx = 5, pady = 5)
        self.m_entry.insert(0, "5")

        # Get the number of columns (n).
        ttk.Label(param_frame, text = "Columns (n):").grid(row = 1, column = 0, padx = 5, pady = 5, sticky = tk.E)
        self.n_rect_entry = ttk.Entry(param_frame, width = 10)
        self.n_rect_entry.grid(row = 1, column = 1, padx = 5, pady = 5)
        self.n_rect_entry.insert(0, "4")

        # Get the tolerance (ε).
        ttk.Label(param_frame, text = "Tolerance (ε):").grid(row = 2, column = 0, padx = 5, pady = 5, sticky = tk.E)
        self.eps_rect_entry = ttk.Entry(param_frame, width = 10)
        self.eps_rect_entry.grid(row = 2, column = 1, padx = 5, pady = 5)
        self.eps_rect_entry.insert(0, "1e-6")

        # Get the max iterations (kmax).
        ttk.Label(param_frame, text = "Max Iterations (kmax):").grid(row = 3, column = 0, padx = 5, pady = 5, sticky = tk.E)
        self.kmax_rect_entry = ttk.Entry(param_frame, width = 10)
        self.kmax_rect_entry.grid(row = 3, column = 1, padx = 5, pady = 5)
        self.kmax_rect_entry.insert(0, "10000")

        # Get the matrix source options for non-square matrix.
        source_frame = ttk.LabelFrame(frame, text = "Matrix Source for A (Non-Square)")
        source_frame.pack(padx = 10, pady = 10, fill = tk.X)
        self.rect_matrix_source_var = tk.StringVar()
        self.rect_matrix_source_var.set("Random")

        # Select the matrix source.
        ttk.Radiobutton(source_frame, text = "Random Matrix", variable = self.rect_matrix_source_var,
                        value = "Random", command = self.toggle_manual_rect).grid(row = 0, column = 0, padx = 5, pady = 5, sticky = tk.W)
        ttk.Radiobutton(source_frame, text = "Manual Matrix Input", variable = self.rect_matrix_source_var,
                        value = "Manual", command = self.toggle_manual_rect).grid(row = 0, column = 1, padx = 5, pady = 5, sticky = tk.W)
        
        # Manual input text for non-square matrix (disabled by default).
        self.manual_rect_text = scrolledtext.ScrolledText(source_frame, width = 60, height = 5, state = tk.DISABLED)
        self.manual_rect_text.grid(row = 1, column = 0, columnspan = 2, padx = 5, pady = 5)
        self.manual_rect_text.insert(tk.END, "Enter matrix rows with numbers separated by spaces, one row per line.\nExample for 3x2:\n1 2\n3 4\n5 6")
        
        # Button to run pseudoinverse for non-square matrix.
        run_btn = ttk.Button(frame, text = "Run Non-Square Inversion (Pseudoinverse)", command = self.run_rect)
        run_btn.pack(padx = 10, pady = 5)
        
        # Create the output area.
        self.rect_output = scrolledtext.ScrolledText(frame, width = 100, height = 20)
        self.rect_output.pack(padx = 10, pady = 10)
        

    def toggle_manual_rect (self):

        '''
            Enable manual matrix text field only if 'Manual' is selected for non-square matrix.
        '''

        if self.rect_matrix_source_var.get() == "Manual":
            self.manual_rect_text.config(state=tk.NORMAL)
        else:
            self.manual_rect_text.config(state=tk.DISABLED)


    def run_square (self):

        '''
            Runs the iterative methods for square matrix inversion.
        '''

        # Retrieve parameters for square matrix inversion.
        try:
            n = int(self.n_entry.get())
            eps = float(self.eps_entry.get())
            kmax = int(self.kmax_entry.get())
        except Exception as e:
            messagebox.showerror("Input Error", "Please check numerical inputs for dimension, ε, and kmax.")
            return
        
        source = self.square_matrix_source_var.get()
        output_str = "=== Square Matrix Inversion ===\n"

        if source == "Special":
            A = special_matrix(n)
            output_str += "Using Special Matrix A (Task 3):\n" + str(A) + "\n"
        elif source == "Random":
            A_random = np.random.rand(n, n)
            A = A_random + n * np.eye(n)
            output_str += "Using Random Matrix A:\n" + str(A) + "\n"
        elif source == "Manual":
            try:
                manual_text = self.manual_square_text.get("1.0", tk.END).strip()
                rows = manual_text.splitlines()
                matrix_list = [list(map(float, row.split())) for row in rows if row.strip() != ""]
                A = np.array(matrix_list)
                if A.shape[0] != A.shape[1]:
                    messagebox.showerror("Input Error", "The manual matrix must be square.")
                    return
                if A.shape[0] != n:
                    messagebox.showwarning("Dimension Mismatch", f"Specified dimension n = {n} but matrix is {A.shape[0]}x{A.shape[1]}. Using provided matrix.")
                    n = A.shape[0]
                output_str += "Using Manual Input Matrix A:\n" + str(A) + "\n"
            except Exception as e:
                messagebox.showerror("Input Error", "Error parsing the manual matrix input. Please check the format.")
                return
        else:
            messagebox.showerror("Input Error", "Unknown matrix source selected.")
            return
        
        # Run the three iterative methods for inversion
        methods = [("Newton-Schultz", iterative_inverse_newton),
                   ("Cubic Iteration", iterative_inverse_cubic),
                   ("Averaged Newton", iterative_inverse_avg)]
        for name, method in methods:
            V, iter_count, res_norm, err = method(A, eps, kmax, init_method=5)
            output_str += f"\nMethod: {name}\n"
            if err:
                output_str += "Error: " + err + "\n"
            else:
                output_str += f"Iterations: {iter_count}\n"
                output_str += f"Residual norm ||A*V - I|| = {res_norm:.3e}\n"
        
        # If special matrix was used, compare with the exact inverse
        if source == "Special":
            invA_exact = exact_inverse_special(n)
            V_newton, _, _, _ = iterative_inverse_newton(A, eps, kmax, init_method=5)
            diff_norm = norm_mat(invA_exact - V_newton)
            output_str += "\nComparison with Exact Inverse:\n"
            output_str += "Exact Inverse A^-1:\n" + str(invA_exact) + "\n"
            output_str += f"Difference norm ||A^-1_exact - A^-1_approx|| = {diff_norm:.3e}\n"
        
        self.square_output.delete(1.0, tk.END)
        self.square_output.insert(tk.END, output_str)
        

    def run_rect (self):

        '''
            Runs the iterative method for non-square matrix inversion (pseudoinverse).
        '''

        # Retrieve parameters for non-square matrix inversion.
        try:
            m = int(self.m_entry.get())
            n = int(self.n_rect_entry.get())
            eps = float(self.eps_rect_entry.get())
            kmax = int(self.kmax_rect_entry.get())
        except Exception as e:
            messagebox.showerror("Input Error", "Please check numerical inputs for the non-square matrix.")
            return
        
        source = self.rect_matrix_source_var.get()
        output_str = "=== Non-Square Matrix Inversion (Pseudoinverse) ===\n"

        if source == "Random":
            A_rect = np.random.rand(m, n)
            # Ensure full rank by adding identity if applicable.
            if m >= n:
                A_rect = A_rect + np.eye(m, n)
            else:
                A_rect = A_rect + np.eye(m, n)
            output_str += f"Using Random Matrix A (shape {m}x{n}):\n" + str(A_rect) + "\n"
        elif source == "Manual":
            try:
                manual_text = self.manual_rect_text.get("1.0", tk.END).strip()
                rows = manual_text.splitlines()
                matrix_list = [list(map(float, row.split())) for row in rows if row.strip() != ""]
                A_rect = np.array(matrix_list)
                if A_rect.shape[0] != m or A_rect.shape[1] != n:
                    messagebox.showwarning("Dimension Mismatch", f"Specified dimensions m={m}, n={n} do not match the provided matrix shape {A_rect.shape}. Using provided matrix dimensions.")
                    m, n = A_rect.shape
                output_str += f"Using Manual Input Matrix A (shape {m}x{n}):\n" + str(A_rect) + "\n"
            except Exception as e:
                messagebox.showerror("Input Error", "Error parsing the manual matrix input for non-square matrix. Please check the format.")
                return
        else:
            messagebox.showerror("Input Error", "Unknown matrix source selected for non-square matrix.")
            return
        
        X, iter_count, res_norm, err = iterative_pseudoinverse(A_rect, eps, kmax)
        if err:
            output_str += "Error: " + err + "\n"
        else:
            output_str += f"Iterations: {iter_count}\n"
            output_str += f"Residual norm ||I - X*A|| = {res_norm:.3e}\n"
            output_str += "Computed left-inverse X (such that X*A ~ I):\n" + str(X) + "\n"
        
        self.rect_output.delete(1.0, tk.END)
        self.rect_output.insert(tk.END, output_str)



if __name__ == '__main__':

    '''
        Starts the application.
    '''

    # Create the application.
    application = Application()

    # Start the application.
    application.mainloop()