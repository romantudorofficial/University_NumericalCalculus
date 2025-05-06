'''
    Homework 5
    Name: Roman Tudor
    Student ID: 310910401ESL201031
    Email Address: romantudor.contact@gmail.com
    Discord Username: romantudorofficial
    Bibliography: ChatGPT, Lecture Notes
    LLM Percentile: 35%
'''

import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from scipy import linalg



class SparseMatrix:

    '''
        Stores a rare matrix in a sparse format.
    '''

    def __init__ (self, matrixSize):

        '''
            Initializes the sparse matrix with the given number of rows and columns.
            Input:
                - matrixSize: size of the matrix (number of rows and columns)
            Output:
                - none
        '''

        # Initialize the size of the sparse matrix.
        self.size = matrixSize

        # Initialize the diagonal of the sparse matrix.
        self.diagonal = np.zeros(matrixSize)

        # Initialize the rows of the sparse matrix.
        self.rows = [dict() for _ in range(matrixSize)]


    def addValue (self, row, column, value):

        '''
            Adds a value to the matrix at a specified position.
            Input:
                - row: row index
                - column: column index
                - value: the value to be added
            Output:
                - none
        '''

        if row == column:
            self.diagonal[row] = value
        else:
            self.rows[row][column] = value
            self.rows[column][row] = value


    def multiplyByVector (self, vector):

        '''
            Multiplies the sparse matrix by a vector.
            Input:
                - vector: the vector to be multiplied
            Output:
                - result: the result of the multiplication
        '''

        result = self.diagonal * vector

        for row, line in enumerate (self.rows):
            for column, value in line.items():
                result[row] += value * vector[column]

        return result


    def convertToDense (self):

        '''
            Converts the sparse matrix to a dense format.
            Input:
                - none
            Output:
                - matrix: the dense matrix
        '''

        matrix = np.diag(self.diagonal)
        
        for row, line in enumerate(self.rows):
            for column, value in line.items():
                matrix[row, column] = value
        
        return matrix



def readMatrix (fileName):

    '''
        Reads a sparse matrix from a file.
        Input:
            - fileName: the name of the file containing the sparse matrix
        Output:
            - matrix: the sparse matrix
    '''

    lines = [content.strip() for content in open(fileName) if content.strip()]
    matrixSize = int(lines[0])
    matrix = SparseMatrix(matrixSize)
    
    for line in lines[1:]:
        
        parts = [part.strip() for part in line.split(',')]
        
        if len(parts) == 3:

            value, row, column = float(parts[0]), int(parts[1]), int(parts[2])
            matrix.addValue(row, column, value)
    
    return matrix



def isSymmetric (matrix, tolerance = 1e-12):

    '''
        Checks if a sparse matrix is symmetric or not.
        Input:
            - matrix: the matrix
            - tolerance: the tolerance for symmetry (default = 1e-12)
        Output:
            - true, if the matrix is symmetric
            - false, otherwise
    '''

    for row, line in enumerate(matrix.rows):
        for column, value in line.items():
            if abs(value - matrix.rows[column].get(row, 0.0)) > tolerance:
                return False
    
    return True



def applyPowerMethod (matrix, epsilon, maximumNumberOfIterations = 10 ** 6):

    '''
        Finds the largest eigenvalue of a matrix using the power method.
        Input:
            - matrix: the matrix
            - epsilon: the tolerance for convergence
            - maximumNumberOfIterations: the maximum number of iterations (default = 10 ^ 6)
        Output:
            - largestEigenvalue: the largest eigenvalue of the matrix
            - residual: the residual
    '''

    # Get the size of the matrix.
    matrixSize = matrix.size

    # Generate a random vector of the same size as the matrix.
    randomVector = np.random.randn(matrixSize)
    randomVector /= np.linalg.norm(randomVector)

    multiplication = matrix.multiplyByVector(randomVector)

    # Get the largest eigenvalue of the matrix.
    largestEigenvalue = randomVector.dot(multiplication)
    
    for _ in range(maximumNumberOfIterations):

        randomVector = multiplication / np.linalg.norm(multiplication)
        multiplication = matrix.multiplyByVector(randomVector)
        newLargestEigenvalue = randomVector.dot(multiplication)

        if np.linalg.norm(multiplication - largestEigenvalue * randomVector) <= matrixSize * epsilon:
            largestEigenvalue = newLargestEigenvalue
            break

        largestEigenvalue = newLargestEigenvalue
    
    else:
        raise RuntimeError("The power method did not converge!")
    
    # Get the residual.
    residual = np.linalg.norm(matrix.multiplyByVector(randomVector) - largestEigenvalue * randomVector)
    
    return largestEigenvalue, residual



def generateRandomMatrix (matrixSize, density = 0.01):

    '''
        Generates a random sparse matrix of a given size and density.
        Input:
            - matrixSize: the size of the matrix
            - density: the density of the matrix (default = 0.01)
        Output:
            - matrix: the generated sparse matrix
    '''

    matrix = SparseMatrix(matrixSize)
    
    for row in range(matrixSize):
        matrix.addValue(row, row, np.random.rand())
    
    numberOfNonZeroElements = max(1, int(density * matrixSize))
    
    for row in range(matrixSize):
        
        columns = np.random.choice([column for column in range(matrixSize) if column != row], numberOfNonZeroElements, replace = False)
        
        for column in columns:
            matrix.addValue(row, column, np.random.rand())
    
    return matrix



def svd_analysis (matrix, vector, epsilon):

    '''
        Performs the SVD (Singular Value Decomposition) analysis on the given matrix and vector.
        Input:
            - matrix: the matrix
            - vector: the vector
            - epsilon: the tolerance for singular values
        Output:
            - singularValues: the singular values of the matrix
            - matrixRank: the rank of the matrix
            - conditionNumber: the condition number of the matrix
            - leastSquaresSolution: the least-squares solution of Ax = b
            - residual: the residual of the least-squares solution
    '''
    
    # Get the SVD of the matrix.
    leftSingularVectors, singularValues, rightSingularVectors = linalg.svd(matrix, full_matrices = False)

    # Get the rank of the matrix.
    matrixRank = np.sum(singularValues > epsilon)

    # Get the condition number of the matrix.
    conditionNumber = singularValues[0] / singularValues[matrixRank - 1] if matrixRank > 0 else np.inf

    # Get the pseudo-inverse of the matrix.
    pseudoInverseOfS = np.diag([1 / singularValue if singularValue > epsilon else 0.0 for singularValue in singularValues])
    pseudoInverseOfMatrix = rightSingularVectors.T @ pseudoInverseOfS @ leftSingularVectors.T

    # Get the least-squares solution of Ax = b.
    leastSquaresSolution = pseudoInverseOfMatrix @ vector

    # Get the residual of the least-squares solution.
    residual = np.linalg.norm(vector - matrix @ leastSquaresSolution)
  
    return singularValues, matrixRank, conditionNumber, leastSquaresSolution, residual



class Application:

    '''
        Class for the GUI application.
    '''

    def __init__ (self, root):

        '''
            Initializes the GUI application.
            Input:
                - root: the root window
            Output:
                - none
        '''

        # Set the title.
        root.title("Homework 5")

        # Set the main frame.
        frame = tk.Frame(root)
        frame.pack(padx = 5, pady = 5, anchor = "w")

        # Get the input values.
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
            M = readMatrix(f)
            self.M_file = M
            self.txt.insert(tk.END, f"\nLoaded File (n={M.size}) from {f}\n\n")
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
        A_r = generateRandomMatrix(p)
        self.A_rand = A_r
        self.txt.insert(tk.END, "Generated A_rand (dense):\n")
        self.txt.insert(tk.END, np.array2string(A_r.convertToDense(), precision=4) + "\n\n")
        if self.M_file:
            Mf = self.M_file
            self.txt.insert(tk.END, "Sparse storage of A_file:\n")
            self.txt.insert(tk.END, f"  d = {np.array2string(Mf.diagonal,precision=4)}\n")
            for i,row in enumerate(Mf.rows):
                if row:
                    self.txt.insert(tk.END, f"  row {i}: {row}\n")
            self.txt.insert(tk.END, "\n")
        else:
            self.txt.insert(tk.END, "A_file not loaded → file storage N/A\n\n")

        # Get the results for the second requirement.
        self.txt.insert(tk.END, "Requirement 2\n\n")
        lam_r, res_r = applyPowerMethod(A_r, eps)
        self.txt.insert(tk.END, f"A_rand →    λ_max = {lam_r:.6e},   residual = {res_r:.6e}\n")
        if self.M_file:
            Mf = self.M_file
            sym = isSymmetric(Mf)
            self.txt.insert(tk.END, f"A_file symmetric? {'Yes' if sym else 'No'}\n")
            if sym:
                lam_f, res_f = applyPowerMethod(Mf, eps)
                self.txt.insert(tk.END, f"A_file →    λ_max = {lam_f:.6e},   residual = {res_f:.6e}\n\n")
            else:
                self.txt.insert(tk.END, "Power Method on A_file N/A (non-symmetric)\n\n")
        else:
            self.txt.insert(tk.END, "A_file not loaded → Req2 file N/A\n\n")

        # Get the results for the third requirement.
        self.txt.insert(tk.END, "Requirement 3\n\n")
        b = np.random.randn(p)
        s, rank, cond, xI, res2 = svd_analysis(A_r.convertToDense(), b, eps)
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
 
    Application(root)

    root.mainloop()



if __name__ == "__main__":

    '''
        Main function to run the program.
    '''

    main()