'''
    Homework 1
'''

import tkinter as tk
from tkinter import messagebox
import time
import math
import numpy as np



# Task 1

def solveTaskOne ():

    m = 1

    while 1 + 10 ** -m != 1:
        m += 1

    return 10 ** (-m + 1)


def getSolutionForTaskOne ():

    u = solveTaskOne()

    result = f"\n\tu = {u}.\n"

    messagebox.showinfo("Task 1", result)



# Task 2

def solveTaskTwoPartOne ():

    u = solveTaskOne()
    
    x = 1.0
    y = u / 10
    z = u / 10
    
    left_side = (x + y) + z
    right_side = x + (y + z)

    if left_side != right_side:
        return True
    
    return False


def solveTaskTwoPartTwo ():

    x = 0.3
    y = 0.7
    z = 0.8
    
    left_side = (x * y) * z
    right_side = x * (y * z)
    
    if left_side != right_side:
        return f"x = {x}, y = {y}, z = {z}"
    
    return "No solution found."


def getSolutionForTaskTwo ():

    part_1 = solveTaskTwoPartOne()
    part_2 = solveTaskTwoPartTwo()

    result = f"Part 1: {part_1}\nPart 2: {part_2}"

    messagebox.showinfo("Task 2", result)



# Task 3

c1 = 1 / math.factorial(3)
c2 = 1 / math.factorial(5)
c3 = 1 / math.factorial(7)
c4 = 1 / math.factorial(9)
c5 = 1 / math.factorial(11)
c6 = 1 / math.factorial(13)


def P1 (x):

    # P1(x) = x - c1 * x^3 + c2 * x^5 = x * (1 - c1 * x^2 + c2 * x^4)
    # Horner: P1(x) = x * (1 + y * (-c1 + c2 * y)), with y = x^2

    y = x * x

    return x * (1 + y * (-c1 + c2 * y))


def P2 (x):

    # P2(x) = x - c1 * x^3 + c2 * x^5 - c3 * x^7 
    # Horner: P2(x) = x * (1 + y * (-c1 + y * (c2 - c3 * y))) , with y = x^2

    y = x * x

    return x * (1 + y * (-c1 + y * (c2 - c3 * y)))


def P3 (x):

    # P3(x) = x - c1 * x^3 + c2 * x^5 - c3 * x^7 + c4 * x^9 
    # Horner: P3(x) = x * (1 + y * (-c1 + y * (c2 + y * (-c3 + c4 * y))))

    y = x * x

    return x * (1 + y * (-c1 + y * (c2 + y * (-c3 + c4 * y))))


def P4 (x):

    # P4(x) = x - 0.166 * x^3 + 0.00833 * x^5 - c3 * x^7 + c4 * x^9
    # Using the approximated coefficients for x^3 and x^5.

    y = x * x

    return x * (1 + y * (-0.166 + y * (0.00833 + y * (-c3 + c4 * y))))


def P5 (x):

    # P5(x) = x - 0.1666 * x^3 + 0.008333 * x^5 - c3 * x^7 + c4 * x^9

    y = x * x

    return x * (1 + y * (-0.1666 + y * (0.008333 + y * (-c3 + c4 * y))))


def P6 (x):

    # P6(x) = x - 0.16666 * x^3 + 0.0083333 * x^5 - c3 * x^7 + c4 * x^9

    y = x * x

    return x * (1 + y * (-0.16666 + y * (0.0083333 + y * (-c3 + c4 * y))))


def P7 (x):

    # P7(x) = x - c1*x^3 + c2*x^5 - c3*x^7 + c4*x^9 - c5*x^11
    # Horner: P7(x) = x*(1 + y*(-c1 + y*(c2 + y*(-c3 + y*(c4 - c5*y))))

    y = x * x

    return x * (1 + y * (-c1 + y * (c2 + y * (-c3 + y * (c4 - c5 * y)))))


def P8 (x):

    # P8(x) = x - c1*x^3 + c2*x^5 - c3*x^7 + c4*x^9 - c5*x^11 + c6*x^13
    # Horner: P8(x) = x*(1 + y*(-c1 + y*(c2 + y*(-c3 + y*(c4 + y*(-c5 + c6*y))))))

    y = x * x

    return x * (1 + y * (-c1 + y * (c2 + y * (-c3 + y * (c4 + y * (-c5 + c6 * y))))))


# List of polynomial functions and their names.
polynomials = [P1, P2, P3, P4, P5, P6, P7, P8]
poly_names = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]


def solveTaskThree ():

    x_values = np.random.uniform(-math.pi / 2, math.pi / 2, 10_000)
    exact_values = np.sin(x_values)

    errors = []
    times = []
    
    for poly in polynomials:

        start_time = time.time()
        poly_values = poly(x_values)
        elapsed_time = time.time() - start_time

        errors.append(np.abs(poly_values - exact_values))
        times.append(elapsed_time)

    errors = np.array(errors)

    best_polynomials = np.argsort(errors, axis = 0)[:3]

    best_counts = np.bincount(best_polynomials[0], minlength = 8)
    second_best_counts = np.bincount(best_polynomials[1], minlength = 8)
    third_best_counts = np.bincount(best_polynomials[2], minlength = 8)

    total_errors = np.sum(errors, axis = 1)
    ranking = np.argsort(total_errors)

    time_ranking = np.argsort(times)

    result = "Polynomial Approximation Results:\n\n"

    result += "Hierarchy of polynomials (best approximations count):\n"

    for i in ranking:
        result += f"{poly_names[i]} (best: {best_counts[i]}, second: {second_best_counts[i]}, third: {third_best_counts[i]})\n"
    
    result += "\nExecution times (sorted in ascending order):\n"

    for i in time_ranking:
        result += f"{poly_names[i]}: {times[i]:.6f} s\n"

    return result


def getSolutionForTaskThree ():

    solution = solveTaskThree()

    result = f"{solution}"

    messagebox.showinfo("Task 3", result)



# Display the results.

root = tk.Tk()
root.title("Homework 1")

button_1 = tk.Button(root, text = "Task 1", command = getSolutionForTaskOne, width = 40)
button_1.pack(padx = 10, pady = 10)

button_2 = tk.Button(root, text = "Task 2", command = getSolutionForTaskTwo, width = 40)
button_2.pack(padx = 10, pady = 10)

button_3 = tk.Button(root, text = "Task 3", command = getSolutionForTaskThree, width = 40)
button_3.pack(padx = 10, pady = 10)

root.mainloop()