'''
    Homework 3
    Bibliography: ChatGPT, Lecture Notes
    LLM Percentile: 30%
'''

import tkinter as tk
from tkinter import messagebox, scrolledtext



class SparseMatrix1:

    '''
        Sparse Matrix Storage - Method 1
        Diagonal and row storage (key: i -> {j -> value})
    '''

    def __init__ (self, n):

        self.n = n
        self.diag = [0.0] * n
        self.rows = [dict() for _ in range(n)]

    
    def add_entry (self, i, j, value):

        if i == j:
            self.diag[i] += value
        else:
            self.rows[i][j] = self.rows[i].get(j, 0.0) + value


    @classmethod
    def from_file (cls, filename):

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


    def multiply_vector (self, x):

        n = self.n
        result = [0.0] * n

        for i in range(n):

            s = self.diag[i] * x[i]

            for j, aij in self.rows[i].items():
                s += aij * x[j]
                
            result[i] = s

        return result


    def add (self, other):

        if self.n != other.n:
            raise ValueError("Matrix sizes do not match")
        
        result = SparseMatrix1(self.n)

        for i in range(self.n):

            result.diag[i] = self.diag[i] + other.diag[i]
            keys = set(self.rows[i].keys()).union(other.rows[i].keys())

            for j in keys:

                result.rows[i][j] = self.rows[i].get(j, 0.0) + other.rows[i].get(j, 0.0)

        return result


    def equals (self, other, epsilon):

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



class SparseMatrix2:

    '''
        Sparse Matrix Storage - Method 2
        Dictionary of dictionaries (key: (i,j) -> value)
    '''

    def __init__ (self, n):

        self.n = n
        self.entries = {}


    def add_entry (self, i, j, value):

        self.entries[(i, j)] = self.entries.get((i, j), 0.0) + value


    @classmethod
    def from_file (cls, filename):

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


    def multiply_vector (self, x):

        n = self.n
        result = [0.0] * n

        for (i, j), value in self.entries.items():
            result[i] += value * x[j]

        return result


    def add (self, other):

        if self.n != other.n:
            raise ValueError("Matrix sizes do not match")
        
        result = SparseMatrix2(self.n)

        for key, value in self.entries.items():
            result.entries[key] = value

        for key, value in other.entries.items():
            result.entries[key] = result.entries.get(key, 0.0) + value

        return result


    def equals (self, other, epsilon):

        if self.n != other.n:
            return False
        
        keys = set(self.entries.keys()).union(other.entries.keys())

        for key in keys:
            if abs(self.entries.get(key, 0.0) - other.entries.get(key, 0.0)) >= epsilon:
                return False
            
        return True



def read_vector_from_file (filename):

    '''
        Reads a vector from a file.
    '''

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
        vec = vec[:n]

    return n, vec



def gauss_seidel (sparse_matrix, b, epsilon, kmax = 10000):

    '''
        Solves the system Ax = b using the Gauss-Seidel method.
    '''

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
        
    return x, iter_count



def residual_norm (sparse_matrix, x, b):

    '''
        Computes the residual norm.
    '''

    # Compute Ax.
    Ax = sparse_matrix.multiply_vector(x)

    # Get the result.
    res = [abs(Ax[i] - b[i]) for i in range(len(b))]

    return max(res)



def solve_system (system_index, epsilon_power):

    '''
        Solves the compulsory part.
        Solves the given systems.
    '''

    # Get the name of the files.
    file_a = f"a_{system_index}.txt"
    file_b = f"b_{system_index}.txt"

    try:

        # Read the matrix and vector from the files.
        A = SparseMatrix1.from_file(file_a)
        n_b, b = read_vector_from_file(file_b)

        if A.n != n_b:
            return f"Error: Dimension mismatch between matrix and vector in system {system_index}"
        
        # Check that diagonal elements are nonzero using tolerance ε = 10^(-p)
        eps = 10 ** (-epsilon_power)

        for i in range(A.n):
            if abs(A.diag[i]) < eps:
                return f"Error: Zero or near-zero diagonal element at row {i}"
        
        # Solve the system using the Gauss-Seidel method.
        sol, iterations = gauss_seidel(A, b, eps)

        # Compute the residual norm.
        res_norm = residual_norm(A, sol, b)

        # Get the result.
        result_text = f"\n\n\tThe system number {system_index} was solved in {iterations} iterations.\n\n"
        result_text += f"\n\tThe residual norm is: {res_norm}.\n\n"
        result_text += "\n\tThe solution is:\n\n" + "\n\n".join(f"\t\tx[{i}] = {sol[i]}" for i in range(len(sol))) + "\n"
        
        return result_text
    
    except Exception as e:

        return f"Error solving system {system_index}: {str(e)}"



def bonus_matrix_addition (epsilon_power):

    '''
        Solves the bonus part.
        Verifies the addition of two sparse matrices using two different methods.
    '''

    eps = 10 ** (-epsilon_power)

    try:

        # Solve using the first method.
        A1 = SparseMatrix1.from_file("a.txt")
        B1 = SparseMatrix1.from_file("b.txt")
        Sum1 = A1.add(B1)
        C1 = SparseMatrix1.from_file("aplusb.txt")
        eq1 = Sum1.equals(C1, eps)

        # Solve using the second method.
        A2 = SparseMatrix2.from_file("a.txt")
        B2 = SparseMatrix2.from_file("b.txt")
        Sum2 = A2.add(B2)
        C2 = SparseMatrix2.from_file("aplusb.txt")
        eq2 = Sum2.equals(C2, eps)

        result_text = "\n\n\t\t\tBonus Part - Matrix Addition Verification\n\n\n"
        result_text += "\tMethod 1: " + ("Passed" if eq1 else "Failed") + "\n\n"
        result_text += "\tMethod 2: " + ("Passed" if eq2 else "Failed") + "\n"

        return result_text
    
    except Exception as e:

        return f"There was an error! {str(e)}"



def run_application ():

    '''
        Runs the application.
    '''

    # Create the main window.
    window = tk.Tk()
    window.title("Homework 3")
    window.geometry("900x700")
    
    # Create the label for the instruction.
    instruction = tk.Label(window, text = "Select a system number (1 - 5) and a value for p (for ε = 10⁻ᵖ):")
    instruction.pack(pady = 10)
    
    # Create the frame for the input fields.
    frame = tk.Frame(window)
    frame.pack(pady = 5)
    
    # Create the input field for the system number.
    tk.Label(frame, text = "System Number:").grid(row = 0, column = 0, padx = 5, pady = 5)
    system_entry = tk.Entry(frame, width = 10)
    system_entry.grid(row = 0, column = 1, padx = 5, pady = 5)
    
    # Create the input field for the epsilon value.
    tk.Label(frame, text = "Value of p:").grid(row = 1, column = 0, padx = 5, pady = 5)
    epsilon_entry = tk.Entry(frame, width = 10)
    epsilon_entry.grid(row = 1, column = 1, padx = 5, pady = 5)
    
    # Create the result area.
    result_area = scrolledtext.ScrolledText(window, width = 90, height = 25)
    result_area.pack(pady = 10)
    

    def solve_button_action ():

        '''
            Solves the given system.
        '''

        try:

            # Get the system number and epsilon value from the input fields.
            sys_num = int(system_entry.get())
            p_val = int(epsilon_entry.get())

            # Get the result.
            result = solve_system(sys_num, p_val)

            # Display the result.
            result_area.delete(1.0, tk.END)
            result_area.insert(tk.END, result)

        except Exception as e:

            messagebox.showerror("There is an error!", str(e))
    

    def bonus_button_action ():

        '''
            Solves the bonus matrix addition.
        '''

        try:

            # Get the epsilon value from the input field.
            p_val = int(epsilon_entry.get())

            # Get the result.
            result = bonus_matrix_addition(p_val)
            
            # Display the result.
            result_area.delete(1.0, tk.END)
            result_area.insert(tk.END, result)

        except Exception as e:

            messagebox.showerror("There is an error!", str(e))
    

    # Create the buttons for solving the systems.
    solve_button = tk.Button(window, text = "Solve System", command = solve_button_action, width = 20)
    solve_button.pack(pady = 5)
    
    # Create the button for the bonus matrix addition.
    bonus_button = tk.Button(window, text = "Bonus: Matrix Addition", command = bonus_button_action, width = 20)
    bonus_button.pack(pady = 5)
    
    window.mainloop()



if __name__ == "__main__":

    '''
        Runs the application.
    '''

    # Run the application.
    run_application()