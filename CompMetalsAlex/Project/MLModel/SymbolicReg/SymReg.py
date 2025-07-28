from pysr import PySRRegressor
import pandas as pd
import numpy as np
import sympy
from sympy import Abs
from sklearn.model_selection import train_test_split


def splitter(filename_X, filename_y, destination):

    X = pd.read_csv(filename_X)  
    y = pd.read_csv(filename_y)

    testdatax = "testdatax2.csv"
    testdatay = "testdatay2.csv"
    traindatax = "traindatax2.csv"
    traindatay = "traindatay2.csv"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43, shuffle=True)
    X_test.to_csv(destination + testdatax, index=False)
    y_test.to_csv(destination + testdatay, index=False)
    X_train.to_csv(destination + traindatax, index=False)
    y_train.to_csv(destination + traindatay, index=False)

def run_symbolic_reg(train_file_x, train_file_y, latex_dest, n_iter):
    
    df_x = pd.read_csv(train_file_x)
    df_y = pd.read_csv(train_file_y)
    X = df_x["Overlap Pen. Energy"].to_numpy().reshape(-1,1)
    y = df_y["SAPT - Undamped"].to_numpy()


    # For integer powers only add: "pow_int(x, y) = pow(x, round(y))" to bin_op
    # And uncomment extra_sympy_mappings
    # Change complexity of variables to 2, and uncomment constraints
    # If you want to avoid variables in the exponent
    model = PySRRegressor(
        niterations = n_iter,
        binary_operators=["+", "*", "-", "/", "pow_int(x, y) = pow(x, abs(round(y)))", "^"], #
        extra_sympy_mappings={
            "pow_int": lambda x, y: x ** abs(sympy.ceiling(y-0.5)), 
            },
        unary_operators=["sin", "cos","sinh","cosh"],
        elementwise_loss="loss(x, y) = (x - y)^2",
        maxsize=30,
        model_selection="best",
        verbosity=1,
        complexity_of_variables=2,
        constraints={"pow_int": (-1, 1), "/": (-1, 1), "^": (1, 4)}, # -1: Arbitrary complexity; 1: Only constants
    )

    model.fit(X, y, variable_names=["a"])

    latex_output = model.latex_table()
    with open(latex_dest, "w") as f:
        f.write(latex_output)

if __name__ == "__main__":
    #run_symbolic_reg("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/Trial_3/traindatax2.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/Trial_3/traindatay2.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/Trial_3/Latex_Eq_T3.txt", 1000000)
    #splitter("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/overlap_pen.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/signed_errors_only.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/Trial_2/")
    # df1 = pd.read_csv("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/overlap_pen.csv")
    # df2 = pd.read_csv("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/signed_errors_only.csv")

    # df_sorted1 = df1.sort_values(by='File')
    # df_sorted2 = df2.sort_values(by='File')
    # df_sorted1.to_csv("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/overlap_pen.csv", index=False)
    # df_sorted2.to_csv("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/signed_errors_only.csv", index=False)


