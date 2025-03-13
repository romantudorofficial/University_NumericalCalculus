'''
    Homework 1
'''



# Task 1

def find_machine_precision():
    m = 1
    while 1 + 10**-m == 1:
        m += 1
    return 10**-m, m

u, m = find_machine_precision()
print(f"Machine precision: u = 10^{-m} = {u}")



def check_addition_non_associativity():
    # Compute machine precision u
    u, _ = find_machine_precision()
    
    x = 1.0
    y = u / 10
    z = u / 10
    
    # Compute both sides of the associativity equation
    left_side = (x + y) + z
    right_side = x + (y + z)
    
    print(f"(x + y) + z = {left_side}")
    print(f"x + (y + z) = {right_side}")
    print("Addition is non-associative:", left_side != right_side)

check_addition_non_associativity()



def check_multiplication_non_associativity():
    a = 1.0e30
    b = 1.0e-30
    c = 1.0e30
    
    left_side = (a * b) * c
    right_side = a * (b * c)
    
    print(f"(a * b) * c = {left_side}")
    print(f"a * (b * c) = {right_side}")
    print("Multiplication is non-associative:", left_side != right_side)

check_multiplication_non_associativity()



# Task 3

import numpy as np
import math
import time

# --------------------------
# 1. Define constants c_i
# --------------------------
c1 = 1 / math.factorial(3)   # 1/3! = 0.16666666666666667
c2 = 1 / math.factorial(5)   # 1/5! = 0.008333333333333333
c3 = 1 / math.factorial(7)   # 1/7! = 1.984126984126984e-4
c4 = 1 / math.factorial(9)   # 1/9! = 2.755731922398589e-6
c5 = 1 / math.factorial(11)  # 1/11! = 2.505210838544172e-8
c6 = 1 / math.factorial(13)  # 1/13! = 1.6059043836821615e-10

# --------------------------
# 2. Generate 10,000 random numbers in [-pi/2, pi/2]
# --------------------------
n_points = 10000
xs = np.random.uniform(-math.pi/2, math.pi/2, n_points)
sin_exact = np.sin(xs)  # "Exact" sine values

# --------------------------
# 3. Define polynomial approximations using Horner's scheme
# --------------------------
def P1(x):
    # P1(x) = x - c1*x^3 + c2*x^5 = x*(1 - c1*x^2 + c2*x^4)
    y = x * x
    return x * (1 + y * (-c1 + c2 * y))

def P2(x):
    # P2(x) = x - c1*x^3 + c2*x^5 - c3*x^7 
    # Horner: P2(x) = x*(1 + y*(-c1 + y*(c2 - c3*y))) , with y=x^2
    y = x * x
    return x * (1 + y * (-c1 + y * (c2 - c3 * y)))

def P3(x):
    # P3(x) = x - c1*x^3 + c2*x^5 - c3*x^7 + c4*x^9 
    # Horner: P3(x) = x*(1 + y*(-c1 + y*(c2 + y*(-c3 + c4*y))))
    y = x * x
    return x * (1 + y * (-c1 + y * (c2 + y * (-c3 + c4 * y))))

def P4(x):
    # P4(x) = x - 0.166*x^3 + 0.00833*x^5 - c3*x^7 + c4*x^9
    # Using the approximated coefficients for x^3 and x^5.
    y = x * x
    return x * (1 + y * (-0.166 + y * (0.00833 + y * (-c3 + c4 * y))))

def P5(x):
    # P5(x) = x - 0.1666*x^3 + 0.008333*x^5 - c3*x^7 + c4*x^9
    y = x * x
    return x * (1 + y * (-0.1666 + y * (0.008333 + y * (-c3 + c4 * y))))

def P6(x):
    # P6(x) = x - 0.16666*x^3 + 0.0083333*x^5 - c3*x^7 + c4*x^9
    y = x * x
    return x * (1 + y * (-0.16666 + y * (0.0083333 + y * (-c3 + c4 * y))))

def P7(x):
    # P7(x) = x - c1*x^3 + c2*x^5 - c3*x^7 + c4*x^9 - c5*x^11
    # Horner: P7(x) = x*(1 + y*(-c1 + y*(c2 + y*(-c3 + y*(c4 - c5*y))))
    y = x * x
    return x * (1 + y * (-c1 + y * (c2 + y * (-c3 + y * (c4 - c5 * y)))))

def P8(x):
    # P8(x) = x - c1*x^3 + c2*x^5 - c3*x^7 + c4*x^9 - c5*x^11 + c6*x^13
    # Horner: P8(x) = x*(1 + y*(-c1 + y*(c2 + y*(-c3 + y*(c4 + y*(-c5 + c6*y))))))
    y = x * x
    return x * (1 + y * (-c1 + y * (c2 + y * (-c3 + y * (c4 + y * (-c5 + c6 * y))))))

# List of polynomial functions and their names for reference
polynomials = [P1, P2, P3, P4, P5, P6, P7, P8]
poly_names = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

# --------------------------
# 4. Compute errors and record the best three approximations for each x
# --------------------------
errors = np.zeros((8, n_points))  # errors[i,j] = error of polynomial i at x[j]
times = np.zeros(8)               # computation time for each polynomial

# Evaluate each polynomial on the 10,000 points and measure time
for i, poly in enumerate(polynomials):
    start_time = time.perf_counter()
    poly_vals = poly(xs)
    times[i] = time.perf_counter() - start_time
    errors[i, :] = np.abs(poly_vals - sin_exact)

# For each x, determine which three polynomials gave the smallest error
top3_counts = np.zeros(8, dtype=int)
for j in range(n_points):
    # Find indices of the three smallest errors at x[j]
    best_indices = np.argsort(errors[:, j])[:3]
    for idx in best_indices:
        top3_counts[idx] += 1

# Create hierarchy: rank polynomials by the number of times they were among the best three
hierarchy = sorted([(poly_names[i], top3_counts[i]) for i in range(8)],
                   key=lambda x: x[1], reverse=True)

print("Hierarchy of polynomials based on top-3 approximations (over 10,000 samples):")
for name, count in hierarchy:
    print(f"{name}: {count} times")

# --------------------------
# 5. Measure and sort computation times for each polynomial
# --------------------------
time_ranking = sorted([(poly_names[i], times[i]) for i in range(8)],
                      key=lambda x: x[1])
print("\nComputation times (sorted in increasing order):")
for name, t in time_ranking:
    print(f"{name}: {t:.6e} seconds")




# GUI

import tkinter as tk
from tkinter import messagebox
import time
import math
import numpy as np

# --- Assume your previously defined functions for the problems are already in this file ---
# For example:
def find_machine_precision():
    m = 1
    while 1 + 10**-m == 1:
        m += 1
    return 10**-m, m

def run_problem1_code():
    # Run the machine precision problem
    u, m = find_machine_precision()
    result = f"Machine precision found:\n u = 10^-{m} = {u}"
    return result

def run_problem2_code():
    # Placeholder: run your associativity tests and collect the outputs.
    # You would typically capture the printed output or build a result string.
    # For demonstration, here is a dummy result.
    result = ("Associativity Test Results:\n"
              "Addition is non-associative: True\n"
              "Multiplication is non-associative: True")
    return result

def run_problem3_code():
    # Placeholder: run the polynomial approximations, error counts, and timing measurements.
    # Build a result string summarizing the hierarchy and computation times.
    result = ("Polynomial Approximation Results:\n"
              "Hierarchy (best approximations): P2 > P5 > P3 > ...\n"
              "Timing (sorted, ascending): P1: 1.2e-06 s, P2: 1.5e-06 s, ...")
    return result

# --- Callback functions for GUI buttons ---
def on_problem1():
    result = run_problem1_code()
    messagebox.showinfo("Problem 1: Machine Precision", result)

def on_problem2():
    result = run_problem2_code()
    messagebox.showinfo("Problem 2: Non-Associativity", result)

def on_problem3():
    result = run_problem3_code()
    messagebox.showinfo("Problem 3: Polynomial Approximations", result)

# --- Setup tkinter window ---
root = tk.Tk()
root.title("Numerical Calculus Homework Results")

# Create buttons for each problem
btn1 = tk.Button(root, text="Problem 1: Machine Precision", command=on_problem1, width=40)
btn1.pack(padx=10, pady=10)

btn2 = tk.Button(root, text="Problem 2: Non-Associativity", command=on_problem2, width=40)
btn2.pack(padx=10, pady=10)

btn3 = tk.Button(root, text="Problem 3: Polynomial Approximations", command=on_problem3, width=40)
btn3.pack(padx=10, pady=10)

root.mainloop()
