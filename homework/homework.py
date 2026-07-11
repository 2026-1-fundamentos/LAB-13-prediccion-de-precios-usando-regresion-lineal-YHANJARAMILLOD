import pandas as pd
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression , mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
import pickle
import gzip
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
train_df = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
test_df = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

train_df['Age'] = 2021 - train_df['Year'].astype(int)
test_df['Age'] = 2021 - test_df['Year'].astype(int)
test_df = test_df.drop(columns=['Year', 'Car_Name'])
train_df = train_df.drop(columns=['Year', 'Car_Name'])

x_train=train_df.drop(columns=['Present_Price'])
y_train=train_df['Present_Price']
x_test=test_df.drop(columns=['Present_Price'])
y_test=test_df['Present_Price']
print("Columnas que le llegan al modelo:", x_train.columns.tolist())
columnas_categoricas = ['Fuel_Type', 'Selling_type', 'Transmission']
columnas_numericas = [col for col in x_train.columns if col not in columnas_categoricas]
preprocesador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first',handle_unknown='ignore'), columnas_categoricas),
        ('num', MinMaxScaler(), columnas_numericas)
    
    ],
    remainder='passthrough' # ¡Muy importante! Le dice que deje las columnas numéricas intactas
)
pipeline_modelo = Pipeline(steps=[
        ("preprocessor", preprocesador),
        ("select_k_best", SelectKBest(score_func=f_regression)),
        ("regressor", LinearRegression())
])
parametros_a_probar = {
    "select_k_best__k": range(1, 20),
    "select_k_best__score_func": [f_regression, mutual_info_regression],
    "regressor__fit_intercept": [True, False],
}
np.random.seed(42)
# 2. Configuramos la Validación Cruzada y la Métrica
optimizador = GridSearchCV(
    estimator=pipeline_modelo,            
    param_grid=parametros_a_probar,       
    cv=KFold(n_splits=10, shuffle=True, random_state=42),                                
    scoring='neg_mean_absolute_error',          
    n_jobs=6,
    verbose=3,
    refit=True,                          
)
optimizador.fit(x_train, y_train)

mejor_modelo = optimizador.best_estimator_

ruta_carpeta = 'files/models'
nombre_archivo = 'model.pkl.gz'
ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)

#print(f"📂 Paso 3: Verificando/Creando la carpeta en '{ruta_carpeta}'...")
os.makedirs(ruta_carpeta, exist_ok=True) # Esto hace lo mismo que tu IF pero en una sola línea limpia

#print("💾 Paso 4: Escribiendo el archivo model.pkl.gz en el disco...")
with gzip.open(ruta_completa, 'wb') as f:
    pickle.dump(optimizador, f, protocol=4)

#print("🎉 ¡PROCESO TERMINADO! El modelo se guardó y creó correctamente.")

y_train_pred = mejor_modelo.predict(x_train)
y_test_pred = mejor_modelo.predict(x_test)

metrics_train = {
    'type': 'metrics',
    'dataset': 'train',
    'r2': float(r2_score(y_train, y_train_pred)),
    'mse': float(mean_squared_error(y_train, y_train_pred)),
    'mad': float(mean_absolute_error(y_train, y_train_pred))
    
}
metrics_test = {
    'type': 'metrics',
    'dataset': 'test',
    'r2': float(r2_score(y_test, y_test_pred)),
    'mse': float(mean_squared_error(y_test, y_test_pred)),
    'mad': float(mean_absolute_error(y_test, y_test_pred))
    
}
ruta_archivo = 'files/output/metrics.json'
os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
    # Escribimos el diccionario de train y un salto de línea (\n)
    archivo.write(json.dumps(metrics_train) + '\n')
    # Escribimos el diccionario de test y un salto de línea (\n)
    archivo.write(json.dumps(metrics_test) + '\n')

#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
