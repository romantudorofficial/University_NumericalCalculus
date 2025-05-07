'''
    Homework 8
    Name: Roman Tudor
    Student ID: 310910401ESL201031
    Email Address: romantudor.contact@gmail.com
    Discord Username: romantudorofficial
    Bibliography: ChatGPT, Lecture Notes
    LLM Percentile: 35%
'''

import math
import random
import tkinter as tk
from tkinter import ttk, scrolledtext



def sigma (z):

    '''
        Sigmoid function for logistic regression.
        Input:
            - z: float or list of floats
        Output:
            - sigmoid(z): float or list of floats
    '''

    return 1.0 / (1 + math.exp(-z))



def F1 (x):

    '''
        Function F1: f(x) = x1^2 + x2^2 - 2*x1 - 4*x2 - 1
        Input:
            - x: list of floats [x1, x2]
        Output:
            - f(x): float
    '''

    x1, x2 = x

    return x1**2 + x2**2 - 2*x1 - 4*x2 - 1



def grad_F1 (x):

    '''
        Gradient of F1: ∇f(x) = [2*x1 - 2, 2*x2 - 4]
        Input:
            - x: list of floats [x1, x2]
        Output:
            - ∇f(x): list of floats [∂f/∂x1, ∂f/∂x2]
    '''

    x1, x2 = x

    return [2*x1 - 2, 2*x2 - 4]



def F2 (x):

    '''
        Function F2: f(x) = 3*x1^2 - 12*x1 + 2*x2^2 + 16*x2 - 10
        Input:
            - x: list of floats [x1, x2]
        Output:
            - f(x): float
    '''

    x1, x2 = x

    return 3*x1**2 - 12*x1 + 2*x2**2 + 16*x2 - 10



def grad_F2 (x):

    '''
        Gradient of F2: ∇f(x) = [6*x1 - 12, 4*x2 + 16]
        Input:
            - x: list of floats [x1, x2]
        Output:
            - ∇f(x): list of floats [∂f/∂x1, ∂f/∂x2]
    '''

    x1, x2 = x

    return [6*x1 - 12, 4*x2 + 16]



def F3 (x):

    '''
        Function F3: f(x) = x1^2 - 4*x1*x2 + 5*x2^2 - 4*x2 + 3
        Input:
            - x: list of floats [x1, x2]
        Output:
            - f(x): float
    '''

    x1, x2 = x

    return x1**2 - 4*x1*x2 + 5*x2**2 - 4*x2 + 3



def grad_F3 (x):

    '''
        Gradient of F3: ∇f(x) = [2*x1 - 4*x2, -4*x1 + 10*x2 - 4]
        Input:
            - x: list of floats [x1, x2]
        Output:
            - ∇f(x): list of floats [∂f/∂x1, ∂f/∂x2]
    '''

    x1, x2 = x

    return [2*x1 - 4*x2, -4*x1 + 10*x2 - 4]



def F4 (x):

    '''
        Function F4: f(x) = x1^2*x2 - 2*x1*x2^2 + 3*x1*x2 + 4
        Input:
            - x: list of floats [x1, x2]
        Output:
            - f(x): float
    '''

    x1, x2 = x

    return x1**2 * x2 - 2*x1 * x2**2 + 3*x1*x2 + 4



def grad_F4 (x):

    '''
        Gradient of F4: ∇f(x) = [2*x1*x2 - 2*x2 + 3, x1^2 - 4*x1*x2 + 3*x1]
        Input:
            - x: list of floats [x1, x2]
        Output:
            - ∇f(x): list of floats [∂f/∂x1, ∂f/∂x2]
    '''

    x1, x2 = x

    return [2*x1*x2 - 2*x2 + 3, x1**2 - 4*x1*x2 + 3*x1]



def loglik (w, data):

    '''
        Log-likelihood function for logistic regression.
        Input:
            - w: list of floats (weights)
            - data: list of tuples (x, y) where x is a list of floats (features) and y is a float (label)
        Output:
            - log-likelihood: float
    '''

    ll = 0.0

    for x, y in data:
        z = sum(wi*xi for wi, xi in zip(w, x))
        ll += y*math.log(sigma(z)) + (1-y)*math.log(1-sigma(z))
    
    return ll



def grad_loglik (w, data):

    '''
        Gradient of the log-likelihood function for logistic regression.
        Input:
            - w: list of floats (weights)
            - data: list of tuples (x, y) where x is a list of floats (features) and y is a float (label)
        Output:
            - gradient: list of floats (gradients w.r.t. weights)
    '''

    g = [0.0]*len(w)

    for x, y in data:
        z = sum(wi*xi for wi, xi in zip(w, x))
        h = sigma(z)
        for i in range(len(w)):
            g[i] += (y - h)*x[i]
    
    return g



# RL.pdf dataset reconstruction
RL_data = [
    ([1,1,0,0,0], 1),
    ([1,1,0,1,0], 1),
    ([1,0,1,0,1], 1),
    ([1,0,0,0,1], 0),
    ([1,1,1,1,0], 0),
    ([1,1,0,1,1], 0),
    ([1,1,0,0,1], 0),
    ([1,0,1,0,0], 0),
]



def numeric_grad (F, x, h):

    '''
        Numerical gradient using central difference method.
        Input:
            - F: function to compute the gradient
            - x: list of floats (point at which to compute the gradient)
            - h: float (step size for finite difference approximation)
        Output:
            - g: list of floats (numerical gradient)
    '''

    n = len(x)
    g = [0.0]*n

    for i in range(n):
        xp2 = x.copy(); xp1 = x.copy()
        xm1 = x.copy(); xm2 = x.copy()
        xp2[i] += 2*h; xp1[i] += h; xm1[i] -= h; xm2[i] -= 2*h
        F1, F2, F3, F4 = F(xp2), F(xp1), F(xm1), F(xm2)
        g[i] = (-F1 + 8*F2 - 8*F3 + F4) / (12*h)
    
    return g



def constant_eta (_, grad, eta, *args):

    '''
        Constant step-size function.
        Input:
            - _: unused parameter (for compatibility with backtracking function)
            - grad: list of floats (gradient at current point)
            - eta: float (step size)
            - args: unused parameters (for compatibility with backtracking function)
        Output:
            - eta: float (constant step size)
    '''

    return eta



def backtracking (F, x, grad, beta, *args):

    '''
        Backtracking line search to find step size.
        Input:
            - F: function to compute the objective function value
            - x: list of floats (current point)
            - grad: list of floats (gradient at current point)
            - beta: float (backtracking parameter)
            - args: unused parameters (for compatibility with constant_eta function)
        Output:
            - eta: float (step size found by backtracking line search)
    '''

    eta = 1.0
    it = 0
    fx = F(x)
    norm2 = sum(g*g for g in grad)

    while F([xi - eta*gi for xi,gi in zip(x, grad)]) > fx - 0.5*eta*norm2 and it < 8:
        eta *= beta; it += 1

    return eta



def gradient_descent (F, gradF, x0, eps, kmax, eta0, beta,
                     use_numeric = False, h = 1e-6, constant = True):
    
    '''
        Gradient descent algorithm.
        Input:
            - F: function to minimize
            - gradF: function to compute the gradient of F
            - x0: list of floats (initial point)
            - eps: float (tolerance for convergence)
            - kmax: int (maximum number of iterations)
            - eta0: float (initial step size)
            - beta: float (backtracking parameter)
            - use_numeric: bool (use numerical gradient if True, else use analytic gradient)
            - h: float (step size for finite difference approximation if use_numeric is True)
            - constant: bool (use constant step size if True, else use backtracking line search)
        Output:
            - x: list of floats (solution found by gradient descent)
            - k: int (number of iterations performed)
            - conv: bool (True if converged, False otherwise)
    '''

    x = x0.copy()

    for k in range(1, kmax+1):

        grad = numeric_grad(F, x, h) if use_numeric else gradF(x)
        eta = constant_eta(None, grad, eta0) if constant else backtracking(F, x, grad, beta)
        step_norm = math.sqrt(sum((eta*g)**2 for g in grad))

        if step_norm < eps:
            return x, k, True
        
        x = [xi - eta*gi for xi, gi in zip(x, grad)]

    return x, kmax, False



class App (tk.Tk):

    '''
        Main application class for the GUI.
    '''

    def __init__ (self):

        '''
            Initialize the application.
            Input:
                - self: instance of the class
            Output:
                - None
        '''

        super().__init__()
        self.title("Homework 8")

        self.funcs = {
            "F1": (F1, grad_F1), "F2": (F2, grad_F2),
            "F3": (F3, grad_F3), "F4": (F4, grad_F4),
            "Log-Lik (RL.pdf)": (
                lambda w: -loglik(w, RL_data),
                lambda w: [-g for g in grad_loglik(w, RL_data)]
            )
        }

        self.create_widgets()


    def create_widgets (self):

        '''
            Create the GUI widgets.
            Input:
                - self: instance of the class
            Output:
                - None
        '''

        row=0
        ttk.Label(self, text="Function:").grid(row=row, column=0, sticky="e")
        self.func_cb = ttk.Combobox(self, values=list(self.funcs), state="readonly")
        self.func_cb.current(0); self.func_cb.grid(row=row, column=1, sticky="w")
        row+=1

        ttk.Label(self, text="Gradient:").grid(row=row, column=0, sticky="e")
        self.grad_var = tk.StringVar(value="Analytic")
        ttk.Radiobutton(self, text="Analytic", variable=self.grad_var, value="Analytic").grid(row=row, column=1)
        ttk.Radiobutton(self, text="Numeric", variable=self.grad_var, value="Numeric").grid(row=row, column=2)
        row+=1

        ttk.Label(self, text="Step‑size:").grid(row=row, column=0, sticky="e")
        self.step_var = tk.StringVar(value="Constant")
        ttk.Radiobutton(self, text="Constant", variable=self.step_var, value="Constant").grid(row=row, column=1)
        ttk.Radiobutton(self, text="Backtracking", variable=self.step_var, value="Backtracking").grid(row=row, column=2)
        row+=1

        ttk.Label(self, text="η (if constant):").grid(row=row, column=0, sticky="e")
        self.eta_ent = ttk.Entry(self); self.eta_ent.insert(0, "0.01")
        self.eta_ent.grid(row=row, column=1, sticky="w")
        row+=1

        ttk.Label(self, text="β (for backtracking):").grid(row=row, column=0, sticky="e")
        self.beta_ent = ttk.Entry(self); self.beta_ent.insert(0, "0.8")
        self.beta_ent.grid(row=row, column=1, sticky="w")
        row+=1

        ttk.Label(self, text="ε (tol):").grid(row=row, column=0, sticky="e")
        self.eps_ent = ttk.Entry(self); self.eps_ent.insert(0, "1e-5")
        self.eps_ent.grid(row=row, column=1, sticky="w")
        row+=1

        ttk.Label(self, text="kmax:").grid(row=row, column=0, sticky="e")
        self.kmax_ent = ttk.Entry(self); self.kmax_ent.insert(0, "30000")
        self.kmax_ent.grid(row=row, column=1, sticky="w")
        row+=1

        ttk.Button(self, text="Run", command=self.run).grid(row=row, column=0, columnspan=2)
        row+=1

        self.txt = scrolledtext.ScrolledText(self, width=60, height=15)
        self.txt.grid(row=row, column=0, columnspan=3)



    def run (self):

        '''
            Run the gradient descent algorithm with the selected parameters.
            Input:
                - self: instance of the class
            Output:
                - None
        '''

        name = self.func_cb.get()
        F, gradF = self.funcs[name]
        use_num = (self.grad_var.get()=="Numeric")
        const = (self.step_var.get()=="Constant")
        eta0 = float(self.eta_ent.get())
        beta = float(self.beta_ent.get())
        eps = float(self.eps_ent.get())
        kmax = int(self.kmax_ent.get())

        # initial x
        dim = 2 if name.startswith("F") else len(RL_data[0][0])

        x0 = [random.uniform(-5,5) for _ in range(dim)]

        sol, iters, conv = gradient_descent(
            F, gradF, x0, eps, kmax, eta0, beta,
            use_numeric=use_num, h=1e-6, constant=const
        )

        self.txt.insert("end", f"Function: {name}\n")
        self.txt.insert("end", f"Initial x0 = {x0}\n")
        self.txt.insert("end", f"Converged: {conv} in {iters} iters\n")
        self.txt.insert("end", f"Solution: {sol}\n\n")
        self.txt.see("end")



if __name__ == "__main__":

    '''
        Starts the application.
    '''

    App().mainloop()