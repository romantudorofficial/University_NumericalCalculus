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



def doSvdAnalysis (matrix, vector, epsilon):

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
  
    return singularValues, matrixRank, conditionNumber, pseudoInverseOfMatrix, leastSquaresSolution, residual



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

        # Set the labels and entry fields for the input values.
        for labelIndex, (labelName, widthOfField) in enumerate ([("p", 6), ("n", 6), ("ε", 8)]):

            tk.Label(frame, text = labelName + ":").grid(row = 0, column = 2 * labelIndex)
            field = tk.Entry(frame, width = widthOfField)
            field.grid(row = 0, column = 2 * labelIndex + 1)
            setattr(self, f"ent_{labelName}", field)

        # Set the load file button.
        tk.Button(frame, text = "Load File", command = self.loadFile)\
          .grid(row = 1, column = 0, columnspan = 2, pady = 4)
        
        # Set the run button.
        tk.Button(frame, text = "Run", command = self.runTasks, bg = "#aaffaa")\
          .grid(row = 1, column = 2, columnspan = 4, padx = 10, pady = 4)

        # Set the output text area.
        self.text = scrolledtext.ScrolledText(root, width = 90, height = 25)
        self.text.pack(padx = 5, pady = 5)

        # Set the initial values for the matrices.
        self.matrix = None
        self.randomMatrix = None


    def loadFile (self):

        '''
            Loads a sparse matrix from a file.
            Input:
                - none
            Output:
                - none
        '''

        file = filedialog.askopenfilename(title = "Select File")

        if not file:
            return
        
        try:
            matrix = readMatrix(file)
            self.matrix = matrix
            self.text.insert(tk.END, f"\nLoaded file (matrix size = {matrix.size}) from \"{file}\".\n\n")
        except Exception as e:
            messagebox.showerror("Load Error!", str(e))


    def runTasks (self):

        '''
            Runs all the tasks.
            Input:
                - none
            Output:
                - none
        '''

        # Clear the text area.
        self.text.delete('1.0', tk.END)

        # Get the input values.
        try:
            p = int(self.ent_p.get())
            n = int(self.ent_n.get())
            eps = float(self.ent_ε.get())
        except:
            messagebox.showerror("Input Error!", "Enter valid values for p, n and ε.")
            return

        self.text.insert(tk.END, "\n\n\tResults\n\n")
        self.text.insert(tk.END, f"p = {p}, n = {n}, ε = {eps}\n\n")

        randomMatrix = generateRandomMatrix(p)
        self.randomMatrix = randomMatrix

        matrix = self.matrix
        symmetric = isSymmetric(matrix)

        # Get the results for the first task.
        if p == n and p > 500:

            self.text.insert(tk.END, "\n\n\tTask 1 (p = n; p, n > 500)\n\n")

            self.text.insert(tk.END, "Generated Random Matrix:\n\n")
            self.text.insert(tk.END, np.array2string(randomMatrix.convertToDense(), precision = 4) + "\n\n")

            if self.matrix:

                matrix = self.matrix
                self.text.insert(tk.END, "Sparse storage:\n")
                self.text.insert(tk.END, f"  d = {np.array2string(matrix.diagonal,precision = 4)}\n")

                for i, row in enumerate(matrix.rows):
                    if row:
                        self.text.insert(tk.END, f"  row {i}: {row}\n")
                self.text.insert(tk.END, "\n")
            else:
                self.text.insert(tk.END, "A_file not loaded → file storage N/A\n")

        # Get the results for the second task.
        elif p == n and symmetric:

            self.text.insert(tk.END, "\n\n\tTask 2 (p = n; A - symmetric)\n\n")

            largestEigenvalue, residual = applyPowerMethod(randomMatrix, eps)

            self.text.insert(tk.END, f"Random Matrix -> Largest Eigenvalue = {largestEigenvalue:.6e}, residual = {residual:.6e}\n")

            if self.matrix:

                matrix = self.matrix
                symmetric = isSymmetric(matrix)

                if symmetric:
                    self.text.insert(tk.END, f"It is symmetric!\n")
                    largestEigenvalue, residual = applyPowerMethod(matrix, eps)
                    self.text.insert(tk.END, f"Read Matrix - Largest Eigenvalue = {largestEigenvalue:.6e}, residual = {residual:.6e}\n")
                else:
                    self.text.insert(tk.END, "It is non symmetric!\n")
            else:
                self.text.insert(tk.END, "File Not Loaded!\n\n")

        # Get the results for the third task.
        else:

            self.text.insert(tk.END, "\n\n\n\tTask 3 (p > n)\n\n")

            vector = np.random.randn(p)

            singularValues, matrixRank, conditionNumber, pseudoInverseOfMatrix, leastSquaresSolution, residual = doSvdAnalysis(randomMatrix.convertToDense(), vector, eps)

            self.text.insert(tk.END,
                f"\nSingular values:\n\n{np.array2string(singularValues, precision = 4)}\n"
                f"\nMatrix rank:\n\n{matrixRank}\n"
                f"\nCondition number:\n\n{conditionNumber:.6e}\n"
                f"\nPseudo inverse:\n\n{pseudoInverseOfMatrix}\n"
                f"\nLeast squares solution:\n\n{np.array2string(leastSquaresSolution[:min(10, len(leastSquaresSolution))], precision = 4)}\n"
                f"\nResidual:\n\n{residual:.6e}\n\n"
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