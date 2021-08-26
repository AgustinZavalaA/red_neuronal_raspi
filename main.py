from ctypes import CDLL, c_double, c_int
import numpy as np


def main():
    so_file = "/home/agustin/Code/Python/CFunctionsInPython/squareC.so"
    functions = CDLL(so_file)

    # functions.sum.argtypes = [c_int * 2]
    functions.divide.restype = c_double
    functions.divide.argtypes = c_double, c_double
    seq = c_int * 2

    print(functions.squareC(10))
    print(functions.squareC(5))
    print(functions.sum(seq(5, 9)))
    print(functions.divide(3.0, 1.5))

    pyar = [[8, 1, 6], [3, 5, 7], [4, 9, 2]]
    pyar = np.array(pyar)
    print(pyar.flatten("F"))


if __name__ == "__main__":
    main()
