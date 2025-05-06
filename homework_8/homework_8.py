#!/usr/bin/env python3
"""
Tema 8: Gradient Descent Framework + Bonus Logistic Regression
Implements:
 - Gradient descent with analytic vs. 4-point numerical gradients
 - Two step-size strategies: constant η and backtracking line search
 - Test suite over the functions listed in Tema 8
 - Bonus: logistic-regression minimization on the RL.pdf dataset
 - Simple Tkinter GUI to choose function, gradient and strategy
"""

import math
import random
import tkinter as tk
from tkinter import ttk, scrolledtext

# -----------------------------
# 1) Functions and analytic gradients
# -----------------------------
def sigma(z):
    return 1.0 / (1 + math.exp(-z))

# Tema 8 test functions (x is list of length 2)
def F1(x):
    x1, x2 = x
    return x1**2 + x2**2 - 2*x1 - 4*x2 - 1

def grad_F1(x):
    x1, x2 = x
    return [2*x1 - 2, 2*x2 - 4]

def F2(x):
    x1, x2 = x
    return 3*x1**2 - 12*x1 + 2*x2**2 + 16*x2 - 10

def grad_F2(x):
    x1, x2 = x
    return [6*x1 - 12, 4*x2 + 16]

def F3(x):
    x1, x2 = x
    return x1**2 - 4*x1*x2 + 5*x2**2 - 4*x2 + 3

def grad_F3(x):
    x1, x2 = x
    return [2*x1 - 4*x2, -4*x1 + 10*x2 - 4]

def F4(x):
    x1, x2 = x
    return x1**2 * x2 - 2*x1 * x2**2 + 3*x1*x2 + 4

def grad_F4(x):
    x1, x2 = x
    return [2*x1*x2 - 2*x2 + 3, x1**2 - 4*x1*x2 + 3*x1]

# logistic-regression log-likelihood (analytic)
def loglik(w, data):
    ll = 0.0
    for x, y in data:
        z = sum(wi*xi for wi, xi in zip(w, x))
        ll += y*math.log(sigma(z)) + (1-y)*math.log(1-sigma(z))
    return ll

def grad_loglik(w, data):
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

# -----------------------------
# 2) Numerical gradient (4-point formula)
# -----------------------------
def numeric_grad(F, x, h):
    n = len(x)
    g = [0.0]*n
    for i in range(n):
        xp2 = x.copy(); xp1 = x.copy()
        xm1 = x.copy(); xm2 = x.copy()
        xp2[i] += 2*h; xp1[i] += h; xm1[i] -= h; xm2[i] -= 2*h
        F1, F2, F3, F4 = F(xp2), F(xp1), F(xm1), F(xm2)
        g[i] = (-F1 + 8*F2 - 8*F3 + F4) / (12*h)
    return g

# -----------------------------
# 3) Learning-rate strategies
# -----------------------------
def constant_eta(_, grad, eta, *args):
    return eta

def backtracking(F, x, grad, beta, *args):
    eta = 1.0; it = 0
    fx = F(x)
    norm2 = sum(g*g for g in grad)
    while F([xi - eta*gi for xi,gi in zip(x, grad)]) > fx - 0.5*eta*norm2 and it < 8:
        eta *= beta; it += 1
    return eta

# -----------------------------
# 4) Gradient descent core
# -----------------------------
def gradient_descent(F, gradF, x0, eps, kmax, eta0, beta,
                     use_numeric=False, h=1e-6, constant=True):
    x = x0.copy()
    for k in range(1, kmax+1):
        grad = numeric_grad(F, x, h) if use_numeric else gradF(x)
        eta = constant_eta(None, grad, eta0) if constant else backtracking(F, x, grad, beta)
        step_norm = math.sqrt(sum((eta*g)**2 for g in grad))
        if step_norm < eps:
            return x, k, True
        x = [xi - eta*gi for xi, gi in zip(x, grad)]
    return x, kmax, False

# -----------------------------
# 5) GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tema 8 – Gradient Descent & Logistic Regression")
        self.funcs = {
            "F1": (F1, grad_F1), "F2": (F2, grad_F2),
            "F3": (F3, grad_F3), "F4": (F4, grad_F4),
            "Log-Lik (RL.pdf)": (
                lambda w: -loglik(w, RL_data),
                lambda w: [-g for g in grad_loglik(w, RL_data)]
            )
        }
        self.create_widgets()

    def create_widgets(self):
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

    def run(self):
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
    App().mainloop()
