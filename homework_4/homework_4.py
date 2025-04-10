'''
    Homework 4
    Bibliography: ChatGPT, Lecture Notes
    LLM Percentile: 30%
'''

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np



def compute_matrix_norm (matrix, norm_type = 1):

    '''
        Compute the matrix norm (default 1-norm).
    '''

    return np.linalg.norm(matrix, ord = norm_type)



def get_initial_square_matrix_inverse_guess (matrix, method = 5):

    '''
        Returns an initial guess for the inverse of the square matrix.
            -> formula 5: V0 = A^T / (||A||_∞^2)
            -> formula 6: V0 = A^T / (||A||_∞ * ||A||_1)
        Input:
            - matrix: square matrix A
            - method: initial guess method (5 or 6)
        Output:
            - initial guess for the inverse of A
    '''

    # Compute the infinity norm of the matrix A.
    matrix_norm_infinity = compute_matrix_norm(matrix, norm_type = np.inf)

    # Compute the 1-norm of the matrix A.
    matrix_norm_one = compute_matrix_norm(matrix, norm_type = 1)

    # If the initial guess method is 5, use the fifth formula.
    if method == 5:
        return matrix.T / (matrix_norm_infinity ** 2)
    
    # If the initial guess method is 6, use the sixth formula.
    elif method == 6:
        return matrix.T / (matrix_norm_infinity * matrix_norm_one)
    
    # If the initial guess method is neither 5, nor 6, raise an error.
    else:
        raise ValueError("\nThe initial guess method must be either 5 or 6.\n")



def iterative_inverse_newton (matrix, eps = 1e-6, kmax = 10000, initial_guess_method = 5):

    '''
        Computes the inverse of a square matrix A using the Newton-Schultz method.
            -> V_{k+1} = V_k*(2I - A*V_k)
        Input:
            - matrix: square matrix A
            - eps: tolerance for convergence
            - kmax: maximum number of iterations
            - initial_guess_method: initial guess method (either 5 or 6)
        Output:
            - initial_matrix_inverse_guess: approximate inverse of A
            - number_of_iterations: number of iterations performed
            - residual_norm: residual norm ||A*V - I||
            - error_message: error message (if any)
    '''

    # Get the size of the matrix.
    matrix_size = matrix.shape[0]

    # Get the initial guess for the inverse of the matrix.
    initial_matrix_inverse_guess = get_initial_square_matrix_inverse_guess(matrix, method = initial_guess_method)

    # Create the identity matrix.
    identity_matrix = np.eye(matrix_size)

    # Initialize the iteration count.
    number_of_iterations = 0

    while number_of_iterations < kmax:

        # Create a copy of the current guess for the inverse.
        initial_matrix_inverse_guess_old = initial_matrix_inverse_guess.copy()

        # Update the guess using the Newton-Schultz iteration.
        initial_matrix_inverse_guess = initial_matrix_inverse_guess_old @ (2 * identity_matrix - matrix @ initial_matrix_inverse_guess_old)

        # Compute the difference between the current and the previous guess.
        difference = compute_matrix_norm(initial_matrix_inverse_guess - initial_matrix_inverse_guess_old)

        # Increase the number of iterations.
        number_of_iterations += 1

        # If the difference is less than the tolerance, break the loop.
        if difference < eps:
            break
        
        # If the difference is too large, return an error message.
        if difference > 1e10:
            return initial_matrix_inverse_guess, number_of_iterations, None, "\nDivergence detected!\n"
    
    # Compute the residual norm ||A*V - I||.
    residual_norm = compute_matrix_norm(matrix @ initial_matrix_inverse_guess - identity_matrix)

    return initial_matrix_inverse_guess, number_of_iterations, residual_norm, None



def iterative_inverse_cubic (matrix, eps = 1e-6, kmax = 10000, initial_guess_method = 5):

    '''
        Computes the inverse of a square matrix A using the cubic iteration method.
            -> V_{k+1} = (1/3)*V_k*(I + (I-A*V_k) + (I-A*V_k)^2)
        Input:
            - matrix: square matrix A
            - eps: tolerance for convergence
            - kmax: maximum number of iterations
            - initial_guess_method: initial guess method (either 5 or 6)
        Output:
            - initial_matrix_inverse_guess: approximate inverse of A
            - number_of_iterations: number of iterations performed
            - residual_norm: residual norm ||A*V - I||
            - error_message: error message (if any)
    '''

    # Get the size of the matrix.
    matrix_size = matrix.shape[0]

    # Get the initial guess for the inverse of the matrix.
    initial_matrix_inverse_guess = get_initial_square_matrix_inverse_guess(matrix, method = initial_guess_method)

    # Create the identity matrix.
    identity_matrix = np.eye(matrix_size)

    # Initialize the iteration count.
    number_of_iterations = 0

    while number_of_iterations < kmax:

        # Create a copy of the current guess for the inverse.
        initial_matrix_inverse_guess_old = initial_matrix_inverse_guess.copy()

        # Get the new guess using the cubic iteration method.
        error_matrix = identity_matrix - matrix @ initial_matrix_inverse_guess_old

        # Update the guess using the cubic iteration method.
        initial_matrix_inverse_guess = (1 / 3) * (initial_matrix_inverse_guess_old @ (identity_matrix + error_matrix + error_matrix @ error_matrix))

        # Compute the difference between the current and the previous guess.
        difference = compute_matrix_norm(initial_matrix_inverse_guess - initial_matrix_inverse_guess_old)

        # Increase the number of iterations.
        number_of_iterations += 1

        # If the difference is less than the tolerance, break the loop.
        if difference < eps:
            break

        # If the difference is too large, return an error message.
        if difference > 1e10:
            return initial_matrix_inverse_guess, number_of_iterations, None, "\nDivergence detected!\n"
    
    # Compute the residual norm ||A*V - I||.
    residual_norm = compute_matrix_norm(matrix @ initial_matrix_inverse_guess - identity_matrix)

    return initial_matrix_inverse_guess, number_of_iterations, residual_norm, None



def iterative_inverse_average (matrix, eps = 1e-6, kmax = 10000, initial_guess_method = 5):
    
    '''
        Computes the inverse of a square matrix A using the averaged Newton-Schultz method.
            -> V_{k+1} = 0.5*(V_k + V_k*(2I - A*V_k))
        Input:
            - matrix: square matrix A
            - eps: tolerance for convergence
            - kmax: maximum number of iterations
            - initial_guess_method: initial guess method (either 5 or 6)
        Output:
            - initial_matrix_inverse_guess: approximate inverse of A
            - number_of_iterations: number of iterations performed
            - residual_norm: residual norm ||A*V - I||
            - error_message: error message (if any)
    '''

    # Get the size of the matrix.
    matrix_size = matrix.shape[0]

    # Get the initial guess for the inverse of the matrix.
    initial_matrix_inverse_guess = get_initial_square_matrix_inverse_guess(matrix, method = initial_guess_method)

    # Create the identity matrix.
    identity_matrix = np.eye(matrix_size)

    # Initialize the iteration count.
    number_of_iterations = 0

    while number_of_iterations < kmax:

        # Create a copy of the current guess for the inverse.
        initial_matrix_inverse_guess_old = initial_matrix_inverse_guess.copy()

        # Get the new guess using the averaged Newton-Schultz method.
        initial_matrix_inverse_guess = 0.5 * (initial_matrix_inverse_guess_old + initial_matrix_inverse_guess_old @ (2 * identity_matrix - matrix @ initial_matrix_inverse_guess_old))
        
        # Compute the difference between the current and the previous guess.
        difference = compute_matrix_norm(initial_matrix_inverse_guess - initial_matrix_inverse_guess_old)

        # Increase the number of iterations.
        number_of_iterations += 1

        # If the difference is less than the tolerance, break the loop.
        if difference < eps:
            break

        # If the difference is too large, return an error message.
        if difference > 1e10:
            return initial_matrix_inverse_guess, number_of_iterations, None, "\nDivergence detected!\n"
    
    # Compute the residual norm ||A*V - I||.
    residual_norm = compute_matrix_norm(matrix @ initial_matrix_inverse_guess - identity_matrix)

    return initial_matrix_inverse_guess, number_of_iterations, residual_norm, None



def get_special_matrix (size):

    '''
        Constructs the n x n matrix A with diagonal entries 1 and superdiagonal entries 2.
        Input:
            - size: dimension of the matrix
        Output:
            - matrix: the constructed matrix
    '''

    # Create an identity matrix.
    matrix = np.eye(size)

    # Set the superdiagonal entries to 2.
    for index in range(size - 1):
        matrix[index, index + 1] = 2

    return matrix



def get_special_matrix_exact_inverse (size):

    '''
        Computes the exact inverse of the special matrix:
            -> For 1 <= i <= j <= n, A^{-1}(i,j)= (-2)^(j-i) and 0 if j < i.
        Input:
            - size: dimension of the matrix
        Output:
            - inverse_matrix: the exact inverse of the special matrix
    '''

    # Create an empty matrix of the same size.
    inverse_matrix = np.zeros((size, size))

    # Fill the matrix with the exact inverse values.
    for line in range(size):
        for column in range(line, size):
            inverse_matrix[line, column] = (-2) ** (column - line)

    return inverse_matrix



def iterative_pseudoinverse (matrix, eps = 1e-6, kmax = 10000):
    
    '''
        Computes the pseudoinverse of a non-square matrix A using the iterative method.
            -> X_{k+1} = (2I - X_k*A)X_k
        Input:
            - matrix: non-square matrix A
            - eps: tolerance for convergence
            - kmax: maximum number of iterations
        Output:
            - initial_guess_matrix_pseudoinverse: approximate pseudoinverse of A
            - number_of_iterations: number of iterations performed
            - residual_norm: residual norm ||I - X*A||
            - error_message: error message (if any)
    '''

    # Get the size of the matrix.
    numberOfLines, numberOfColumns = matrix.shape

    # Create an identity matrix.
    identity_matrix = np.eye(numberOfColumns)

    # Compute the 1-norm of the matrix A.
    matrix_norm_one = compute_matrix_norm(matrix, norm_type = 1)

    # Compute the infinity norm of the matrix A.
    matrix_norm_infinity = compute_matrix_norm(matrix, norm_type = np.inf)

    # Get the initial guess for the pseudoinverse of the matrix.
    initial_guess_matrix_pseudoinverse = matrix.T / (matrix_norm_one * matrix_norm_infinity)

    # Initialize the iteration count.
    number_of_iterations = 0

    while number_of_iterations < kmax:

        # Create a copy of the current guess for the pseudoinverse.
        initial_guess_matrix_pseudoinverse_old = initial_guess_matrix_pseudoinverse.copy()

        # Update the guess using the iterative method.
        initial_guess_matrix_pseudoinverse = (2 * identity_matrix - initial_guess_matrix_pseudoinverse_old @ matrix) @ initial_guess_matrix_pseudoinverse_old

        # Increase the number of iterations.
        number_of_iterations += 1

        # If the difference is less than the tolerance, break the loop.
        if compute_matrix_norm(identity_matrix - initial_guess_matrix_pseudoinverse @ matrix) < eps:
            break

        # If the difference is too large, return an error message.
        if compute_matrix_norm(initial_guess_matrix_pseudoinverse - initial_guess_matrix_pseudoinverse_old) > 1e10:
            return initial_guess_matrix_pseudoinverse, number_of_iterations, None, "\nDivergence detected!\n"
    
    # Compute the residual norm ||I - X*A||.
    residual_norm = compute_matrix_norm(identity_matrix - initial_guess_matrix_pseudoinverse @ matrix)

    return initial_guess_matrix_pseudoinverse, number_of_iterations, residual_norm, None



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
        run_btn = ttk.Button(frame, text = "Run Compulsory", command = self.run_compulsory)
        run_btn.pack(padx = 10, pady = 5)
        
        # Create the output area.
        self.square_output = scrolledtext.ScrolledText(frame, width = 100, height = 20)
        self.square_output.pack(padx = 10, pady = 10)


    def toggle_manual_square (self):

        '''
            Enable manual matrix text field only if 'Manual' is selected for square matrix.
        '''

        if self.square_matrix_source_var.get() == "Manual":
            self.manual_square_text.config(state = tk.NORMAL)
        else:
            self.manual_square_text.config(state = tk.DISABLED)
    

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
        run_btn = ttk.Button(frame, text = "Run Bonus", command = self.run_bonus)
        run_btn.pack(padx = 10, pady = 5)
        
        # Create the output area.
        self.rect_output = scrolledtext.ScrolledText(frame, width = 100, height = 20)
        self.rect_output.pack(padx = 10, pady = 10)
        

    def toggle_manual_rect (self):

        '''
            Enable manual matrix text field only if 'Manual' is selected for non-square matrix.
        '''

        if self.rect_matrix_source_var.get() == "Manual":
            self.manual_rect_text.config(state = tk.NORMAL)
        else:
            self.manual_rect_text.config(state = tk.DISABLED)


    def run_compulsory (self):

        '''
            Runs the iterative methods for square matrix inversion.
        '''

        # Get the parameters for the square matrix inversion.
        try:
            n = int(self.n_entry.get())
            eps = float(self.eps_entry.get())
            kmax = int(self.kmax_entry.get())
        except Exception as e:
            messagebox.showerror("Input Error", "Please check numerical inputs for dimension, ε, and kmax.")
            return
        
        # Get the matrix source and construct the matrix.
        source = self.square_matrix_source_var.get()

        # Initialize the output string.
        output_str = "\n\tCompulsory - Square Matrix Inversion:\n\n"

        if source == "Special":
            A = get_special_matrix(n)
            output_str += "\tUsing Special Matrix A (Task 3):\n\n" + str(A) + "\n"

        elif source == "Random":
            A_random = np.random.rand(n, n)
            A = A_random + n * np.eye(n)
            output_str += "\tUsing Random Matrix A:\n\n" + str(A) + "\n"

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
                output_str += "\nUsing Custom Matrix A:\n\n" + str(A) + "\n"
            except Exception as e:
                messagebox.showerror("Input Error", "Error parsing the manual matrix input. Please check the format.")
                return
        else:
            messagebox.showerror("Input Error", "Unknown matrix source selected.")
            return
        
        # Run the three iterative methods for inversion.
        methods = [("Newton-Schultz", iterative_inverse_newton),
                   ("Cubic Iteration", iterative_inverse_cubic),
                   ("Averaged Newton", iterative_inverse_average)]
        
        for name, method in methods:

            V, number_of_iterations, res_norm, err = method(A, eps, kmax, initial_guess_method=5)

            output_str += f"\n\tMethod {name}:\n\n"

            if err:
                output_str += "Error: " + err + "\n"

            else:
                output_str += f"\t\tIterations: {number_of_iterations}\n"
                output_str += f"\t\tResidual norm ||A*V - I|| = {res_norm:.3e}\n"
        
        # If special matrix was used, compare with the exact inverse
        if source == "Special":
            invA_exact = get_special_matrix_exact_inverse(n)
            V_newton, _, _, _ = iterative_inverse_newton(A, eps, kmax, initial_guess_method=5)
            diff_norm = compute_matrix_norm(invA_exact - V_newton)
            output_str += "\n\tComparison with Exact Inverse:\n\n"
            output_str += "\t\tExact Inverse A^-1:\n\n" + str(invA_exact) + "\n"
            output_str += f"\n\t\tDifference norm ||A^-1_exact - A^-1_approx|| = {diff_norm:.3e}\n"
        
        self.square_output.delete(1.0, tk.END)
        self.square_output.insert(tk.END, output_str)
        

    def run_bonus (self):

        '''
            Runs the iterative method for non-square matrix inversion (pseudoinverse).
        '''

        # Get the parameters for the non-square matrix inversion.
        try:
            m = int(self.m_entry.get())
            n = int(self.n_rect_entry.get())
            eps = float(self.eps_rect_entry.get())
            kmax = int(self.kmax_rect_entry.get())
        except Exception as e:
            messagebox.showerror("Input Error", "Please check numerical inputs for the non-square matrix.")
            return
        
        source = self.rect_matrix_source_var.get()

        output_str = "\n\tNon-Square Matrix Inversion (Pseudoinverse):\n\n"

        if source == "Random":
            A_rect = np.random.rand(m, n)
            # Ensure full rank by adding identity if applicable.
            if m >= n:
                A_rect = A_rect + np.eye(m, n)
            else:
                A_rect = A_rect + np.eye(m, n)
            output_str += f"\tUsing Random Matrix A (shape {m}x{n}):\n\n" + str(A_rect) + "\n"

        elif source == "Manual":
            try:
                manual_text = self.manual_rect_text.get("1.0", tk.END).strip()
                rows = manual_text.splitlines()
                matrix_list = [list(map(float, row.split())) for row in rows if row.strip() != ""]
                A_rect = np.array(matrix_list)
                if A_rect.shape[0] != m or A_rect.shape[1] != n:
                    messagebox.showwarning("Dimension Mismatch", f"Specified dimensions m={m}, n={n} do not match the provided matrix shape {A_rect.shape}. Using provided matrix dimensions.")
                    m, n = A_rect.shape
                output_str += f"\tUsing Manual Input Matrix A (shape {m}x{n}):\n" + str(A_rect) + "\n"
            except Exception as e:
                messagebox.showerror("Input Error", "Error parsing the manual matrix input for non-square matrix. Please check the format.")
                return
            
        else:
            messagebox.showerror("Input Error", "Unknown matrix source selected for non-square matrix.")
            return
        
        X, number_of_iterations, res_norm, err = iterative_pseudoinverse(A_rect, eps, kmax)

        if err:
            output_str += "Error: " + err + "\n"
        else:
            output_str += f"\n\tIterations: {number_of_iterations}\n"
            output_str += f"\n\tResidual norm ||I - X*A|| = {res_norm:.3e}\n"
            output_str += "\n\tComputed left-inverse X (such that X*A ~ I):\n\n" + str(X) + "\n"
        
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