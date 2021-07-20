from flask import Flask, render_template,request
import re
import numpy as np
from urllib.request import Request, urlopen
from unicodedata import normalize
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('portafolio.hbs')

@app.route('/', methods=['post'])
def primero ():

    Papers = pd.read_csv('datos_pro.csv') 
    Papers = pd.DataFrame(Papers)
    Papers1 = Papers.drop(['Fecha'], axis = 1)
    #print(Papers)

    tit = Papers1.iloc[:,4].values
    tit = list(tit)

    # LIMPIEZA DE DATOS
    def control (Matriz):
        n6 = []
        n7 = []
        for i in Matriz:
            leer = str([i])
            # Eliminar carateres
            n1 = re.sub('[^a-zA-Z \n\.]+',' ', leer)
            # Minusculas
            n2 = n1.lower()
            n3 = n2.split()
            n6.append(n2)  
        
        return n6

    titulos = control(tit)
    #print(titulos)

    ######################################## PROBLEMAS DE MACHINE LEARNING 

    ########### TAREAS DE REGRESIÓN

    ###### PREDICCIÓN DE VENTAS
    print(' ========================== PREDICCIÓN DE VENTAS ============================')
    cant_datos = 162
    d = Papers1.head(cant_datos)
    #print(d) 
    train = 0.7
    test = 1-train

    Papers2 = Papers1.drop(['Prod_Disponibles','Tipo'],1)
    #df_shuffled = Papers2.sample(frac=1, random_state=8).reset_index(drop=True) 

    suma = (Papers2['Cantidad_Ventas'] + Papers2['Monto_Compras'])
    dataX =  pd.DataFrame()
    dataX["suma"] = suma
    XY_train = np.array(dataX)
    z_train = Papers2['Monto_Ventas'].values

    # Split-out validation dataset
    X_train, X_test, y_train, y_test = train_test_split(XY_train, z_train, test_size = 0.3, random_state=0) #20

    print()
    regr2 = linear_model.LinearRegression()
    regr2.fit(X_train, y_train)

    # EVALUACIÓN DE RENDIMIENTO
    y_pred = regr2.predict(X_test)
    print()
    print('EVALUACIÓN DE RENDIMIENTO REGRESIÓN:')
    pres = regr2.score(X_train, y_train)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    print()
    print("Presición del modelo:", pres)
    print('MAE: ', mae)
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('R2: ', r2)
    print()
    print()

    val1 = int(request.form['name'])
    val2 = float(request.form['name1'])
    z_Dosmil = regr2.predict([[val1 + val2]])
    resp = float(z_Dosmil)
    print('PREDICCIÓN MONTO VENTAS: ', resp)
    print()


    ###### PREDICCIÓN DE COMPRAS
    print(' ========================== PREDICCIÓN DE COMPRAS ============================')
    d1 = Papers1.head(cant_datos)
    #print(d) 
    train1 = 0.7
    test1 = 1-train1

    print()

    ### Training
    #tra1 = cant_datos*train1
    suma1 = (Papers2['Cantidad_Ventas'] + Papers2['Monto_Ventas'])
    dataX1 =  pd.DataFrame()
    dataX1["suma"] = suma1
    XY_train1 = np.array(dataX1)
    z_train1 = Papers2['Monto_Compras'].values

    # Split-out validation dataset
    X_train1, X_test1, y_train1, y_test1 = train_test_split(XY_train1, z_train1, test_size = 0.3, random_state = 20)

    print()
    regr22 = linear_model.LinearRegression()
    regr22.fit(X_train1, y_train1)

    # EVALUACIÓN DE RENDIMIENTO
    y_pred1 = regr22.predict(X_test1)
    print()
    print('EVALUACIÓN DE RENDIMIENTO REGRESIÓN:')
    print()
    pres1 = regr22.score(X_train1, y_train1)
    mae1 = metrics.mean_absolute_error(y_test1, y_pred1)
    mse1 = metrics.mean_squared_error(y_test1, y_pred1)
    rmse1 = np.sqrt(metrics.mean_squared_error(y_test1, y_pred1))
    r21 = metrics.r2_score(y_test1, y_pred1)
    print("Presición del modelo:", pres1)
    print('MAE: ', mae1)
    print('MSE: ', mse1)
    print('RMSE: ', rmse1)
    print('R2: ', r21)
    print()
    print()

    """ val11 = int(request.form['name2'])
    val22 = float(request.form['name3'])
    z_Dosmil1 = regr2.predict([[val11 + val22]])
    print('PREDICCIÓN MONTO COMPRAS: ', float(z_Dosmil1)) """

    return render_template('portafolio.hbs',val1=val1,val2=val2,pres=pres,mae=mae,mse=mse,rmse=rmse,r2=r2,resp=resp,pres1=pres1,mae1=mae1,mse1=mse1,rmse1=rmse1,r21=r21)

if __name__ == '__main__':
    app.run(debug=True)