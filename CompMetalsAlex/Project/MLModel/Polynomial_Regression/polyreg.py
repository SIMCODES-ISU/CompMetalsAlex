import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def poly_regression(file_x, file_y, plotname):

    df_x = pd.read_csv(file_x)
    df_y = pd.read_csv(file_y)

    X = df_x["Overlap Pen. Energy"].to_numpy().reshape(-1,1)
    y = df_y["SAPT - Undamped"].to_numpy()

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    # Construct and print the equation
    equation_str = f"y = {intercept:.4f}"

    for i, coef in enumerate(coefficients):
        if i == 0: # This is the coefficient for x^0, which is the intercept in the transformed features, but handled separately
            continue
        elif i == 1:
            equation_str += f" + {coef:.4f}x"
        else:
            equation_str += f" + {coef:.4f}x^{i}" # Adjust index as x_poly includes x^0
    print(equation_str)

    y_pred = model.predict(X_poly)

    r2 = r2_score(y, y_pred)
    print(f"The R2 score is {r2}")

    plot_predictions_poly(X,y,y_pred,plotname)

def plot_predictions_poly(X,y,y_pred,plotname):

    plt.scatter(X, y, color='blue', label='Original Data')
    plt.plot(X, y_pred, color='red', label='Polynomial Regression Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.savefig(plotname)

def test_predictions(test_x,test_y, destination):
    
    df_x = pd.read_csv(test_x)
    df_y = pd.read_csv(test_y)
    df_x["Polynomial: -0.0008 + 1.6968 OPE + 17.0727 OPE^2"] = -0.0008 + 1.6968*df_x["Overlap Pen. Energy"] + 17.0727*(df_x["Overlap Pen. Energy"]**2)
    
    y_pred = df_x["Polynomial: -0.0008 + 1.6968 OPE + 17.0727 OPE^2"].to_numpy()
    y = df_y["SAPT - Undamped"].to_numpy()
    r2 = r2_score(y, y_pred)
    
    print(f"The R2 score is {r2}")
    
    df_x.to_csv(destination, index = False)



if __name__ == "__main__":

    #poly_regression("/mnt/c/c++_tests/games_work/ml_work/MLModel/Polynomial Regression/Trial 1/traindatax2.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/Polynomial Regression/Trial 1/traindatay2.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/Polynomial Regression/Trial 1/poly_fit.png")
    test_predictions("/mnt/c/c++_tests/games_work/ml_work/MLModel/Polynomial Regression/Trial 1/traindatax2.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/Polynomial Regression/Trial 1/traindatay2.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/Polynomial Regression/Trial 1/train_data_pred.csv")