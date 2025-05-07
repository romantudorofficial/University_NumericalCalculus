'''
    Homework 6
    Name: Roman Tudor
    Student ID: 310910401ESL201031
    Email Address: romantudor.contact@gmail.com
    Discord Username: romantudorofficial
    Bibliography: ChatGPT, Lecture Notes
    LLM Percentile: 30%
'''

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



def f1 (x):

    '''
        This function computes the polynomial f(x) = x^4 - 12x^3 + 30x^2 + 12.
        Input:
            - x: The input value(s) for the polynomial.
        Output:
            - The computed polynomial value(s).
    '''

    return x ** 4 - 12 * x ** 3 + 30 * x ** 2 + 12



def f2 (x):

    '''
        This function computes the polynomial f(x) = sin(x) - cos(x).
        Input:
            - x: The input value(s) for the polynomial.
        Output:
            - The computed polynomial value(s).
    '''
    
    return np.sin(x) - np.cos(x)



def f3 (x):

    '''
        This function computes the polynomial f(x) = sin(2x) + sin(x) + cos(3x).
        Input:
            - x: The input value(s) for the polynomial.
        Output:
            - The computed polynomial value(s).
    '''
    
    return np.sin(2 * x) + np.sin(x) + np.cos(3 * x)



def f4 (x):

    '''
        This function computes the polynomial f(x) = sin^2(x) - cos^2(x).
        Input:
            - x: The input value(s) for the polynomial.
        Output:
            - The computed polynomial value(s).
    '''
    
    return np.sin(x) ** 2 - np.cos(x) ** 2



funcs = {
    "x^4 -12x^3 +30x^2 +12": f1,
    "sin(x)-cos(x)": f2,
    "sin(2x)+sin(x)+cos(3x)": f3,
    "sin^2(x)-cos^2(x)": f4
}



def horner (coeffs, x):

    '''
        This function evaluates a polynomial using Horner's method.
        Input:
            - coeffs: The coefficients of the polynomial.
            - x: The input value for the polynomial.
        Output:
            - The computed polynomial value.
    '''

    result = coeffs[-1]

    for c in reversed(coeffs[:-1]):
        result = result * x + c

    return result



def least_squares (x, y, m):

    '''
        This function computes the coefficients of the least squares polynomial approximation.
        Input:
            - x: The input x values (nodes).
            - y: The input y values (function values).
            - m: The degree of the polynomial.
        Output:
            - The coefficients of the least squares polynomial.
    '''

    n = len(x)

    # Build normal matrix B and vector fvec.
    B = np.zeros((m+1, m+1))
    fvec = np.zeros(m+1)

    for i in range(m+1):
        for j in range(m+1):
            B[i,j] = np.sum(x**(i+j))
        fvec[i] = np.sum(y * x**i)
    a = np.linalg.solve(B, fvec)

    return a



def trig_interp (x, y, m):

    '''
        This function computes the coefficients of the trigonometric interpolation.
        Input:
            - x: The input x values (nodes).
            - y: The input y values (function values).
            - m: The degree of the trigonometric polynomial.
        Output:
            - The coefficients of the trigonometric polynomial.
    '''

    n = 2*m
    N = n + 1
    T = np.zeros((N, N))

    # Build T matrix
    for i, xi in enumerate(x):
        row = [1]
        for k in range(1, m+1):
            row.append(np.sin(k*xi))
            row.append(np.cos(k*xi))
        T[i,:] = row

    coeffs = np.linalg.solve(T, y)

    return coeffs



def eval_trig (coeffs, xbar, m):

    '''
        This function evaluates the trigonometric polynomial at a given point.
        Input:
            - coeffs: The coefficients of the trigonometric polynomial.
            - xbar: The input value for the polynomial.
            - m: The degree of the trigonometric polynomial.
        Output:
            - The computed trigonometric polynomial value.
    '''

    val = coeffs[0]
    idx = 1

    for k in range(1, m+1):
        val += coeffs[idx] * np.sin(k*xbar)
        idx += 1
        val += coeffs[idx] * np.cos(k*xbar)
        idx += 1

    return val



class App:

    '''
        This class creates the GUI for the numerical calculus approximation program.
    '''

    def __init__ (self, root):

        '''
            Initializes the GUI components and layout.
            Input:
                - root: The main window of the application.
            Output:
                - None
        '''

        self.root = root
        root.title("Numerical Calculus Approximation")

        # Input frame
        frame = ttk.Frame(root, padding=10)
        frame.grid(row=0, column=0, sticky="W")

        # Entries
        ttk.Label(frame, text="x0:").grid(row=0, column=0)
        self.x0_entry = ttk.Entry(frame, width=10); self.x0_entry.grid(row=0,column=1)
        ttk.Label(frame, text="xn:").grid(row=0, column=2)
        self.xn_entry = ttk.Entry(frame, width=10); self.xn_entry.grid(row=0,column=3)
        ttk.Label(frame, text="n (nodes-1):").grid(row=1, column=0)
        self.n_entry = ttk.Entry(frame, width=10); self.n_entry.grid(row=1,column=1)
        ttk.Label(frame, text="m (deg <=5):").grid(row=1, column=2)
        self.m_entry = ttk.Entry(frame, width=10); self.m_entry.grid(row=1,column=3)
        ttk.Label(frame, text="xbar:").grid(row=2, column=0)
        self.xbar_entry = ttk.Entry(frame, width=10); self.xbar_entry.grid(row=2,column=1)
        ttk.Label(frame, text="Function:").grid(row=2, column=2)
        self.func_var = tk.StringVar()
        self.func_combo = ttk.Combobox(frame, textvariable=self.func_var, values=list(funcs.keys()), state='readonly')
        self.func_combo.grid(row=2, column=3); self.func_combo.current(0)

        # Compute button
        ttk.Button(frame, text="Compute", command=self.compute).grid(row=3, column=0, columnspan=4, pady=5)

        # Output text
        self.output = tk.Text(root, width=80, height=10)
        self.output.grid(row=1, column=0)

        # Plot area
        self.fig = Figure(figsize=(5,4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=2)


    def compute (self):

        '''
            Computes the least squares polynomial and trigonometric interpolation for the given function and inputs.
            Input:
                - None
            Output:
                - None
        '''

        try:

            x0 = float(self.x0_entry.get())
            xn = float(self.xn_entry.get())
            n = int(self.n_entry.get())
            m = int(self.m_entry.get())
            xbar = float(self.xbar_entry.get())

            if m > 5:
                messagebox.showerror("Error", "m must be <= 5")
                return
            
            # Generate nodes
            xi = np.sort(np.random.uniform(x0, xn, n+1))
            f = funcs[self.func_var.get()]
            yi = f(xi)

            # True value
            fbar = f(xbar)

            self.output.delete(1.0, tk.END)

            # LS polynomial
            a = least_squares(xi, yi, m)
            Pm = horner(a, xbar)
            err_ls = abs(Pm - fbar)
            sum_ls = np.sum(np.abs(horner(a, xi) - yi))

            self.output.insert(tk.END, f"Least Squares P_{m}({xbar:.4f}) = {Pm:.6f}\n")
            self.output.insert(tk.END, f"|P_{m}({xbar:.4f}) - f({xbar:.4f})| = {err_ls:.6f}\n")
            self.output.insert(tk.END, f"Sum |P_{m}(x_i) - y_i| = {sum_ls:.6f}\n\n")

            # Trig interp
            # Require periodic [0,2pi)
            xi_t = np.sort(np.random.uniform(0, 2*np.pi, 2*m+1))
            yi_t = f(xi_t)
            coeffs = trig_interp(xi_t, yi_t, m)
            Tn = eval_trig(coeffs, xbar, m)
            err_trig = abs(Tn - fbar)
            
            self.output.insert(tk.END, f"Trig Interp T_{2*m}({xbar:.4f}) = {Tn:.6f}\n")
            self.output.insert(tk.END, f"|T_{2*m}({xbar:.4f}) - f({xbar:.4f})| = {err_trig:.6f}\n")

            # Plot
            xs = np.linspace(x0, xn, 200)
            ys = f(xs)
            self.ax.clear()
            self.ax.plot(xs, ys, label='f')

            # Plot LS approx
            ys_ls = [horner(a, xx) for xx in xs]
            self.ax.plot(xs, ys_ls, '--', label=f'P_{m}')

            # Plot trig approx projected onto [x0,xn]
            ys_trig = [eval_trig(coeffs, xx, m) for xx in xs]
            self.ax.plot(xs, ys_trig, ':', label=f'T_{2*m}')
            self.ax.legend()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))



if __name__ == '__main__':

    '''
        This is the starting point of the program.
    '''

    root = tk.Tk()
    app = App(root)
    root.mainloop()