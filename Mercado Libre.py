import requests
import pandas as pd
import matplotlib.pyplot as plt
%pip install seaborn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Insertamos la clave de país a analizar
clavepais = 'MLA'

# Importamos de la API el modelo de las categorías
cats = requests.get(f'https://api.mercadolibre.com/sites/{clavepais}/categories')
categories = cats.json()  # Convertimos el modelo a una lista para poder trabajar con él

# Convertimos la lista a un DataFrame
dfcategories = pd.DataFrame(categories)

# Comentar o Descomentar esta linea para solo una categoria
dfcategories = dfcategories[dfcategories['id'] == 'MLA1051']
dfcategories = dfcategories.reset_index(drop=True)

# DataFrame vacío para almacenar los datos completos
df_completo = pd.DataFrame()

# Recorrer cada categoría
for i in range(len(dfcategories)):
    cat_id = dfcategories['id'][i]
    offset = 0  # Inicializamos el offset
    data = []

    # Paginación de 20 páginas, suponiendo 50 resultados por página
    for j in range(20):
        url = f'https://api.mercadolibre.com/sites/{clavepais}/search?category={cat_id}&offset={offset}'
        response = requests.get(url)
        items = response.json()

        # Añadir cada item a la lista de datos
        for item in items['results']:
            data.append({
                'id_category' : dfcategories['id'][i],
                'name_category' : dfcategories['name'][i],
                'seller_id': item['seller']['id'],
                'seller_nickname': item['seller']['nickname'],
                'id_product' : item['id'],
                'currency' : item['currency_id'],
                'title': item['title'],
                'price': item['price'],
                'condition': item['condition'],
                'available_quantity' : item['available_quantity'],
                'free_shipping' : item['shipping']['free_shipping'],
            })
        offset += 50  # Incrementamos el offset para la siguiente página

    # Crear un DataFrame temporal y concatenar al DataFrame completo
    df_temp = pd.DataFrame(data)
    df_completo = pd.concat([df_completo, df_temp], ignore_index=True)

# Creamos columnas vacías que llenamos iterando sobre la calificación de cada vendedor
df_completo['user_type'] = ''
df_completo['level_id'] = ''
df_completo['power_seller_status'] = ''
df_completo['period'] = ''
df_completo['total'] = ''

for i in range(len(df_completo)):
    key = df_completo['seller_id'][i]
    infosellers = requests.get(f'https://api.mercadolibre.com/users/{key}')
    infosellerslist = infosellers.json()
    df_completo['user_type'][i] = infosellerslist['user_type']
    df_completo['level_id'][i] = infosellerslist['seller_reputation']['level_id']
    df_completo['power_seller_status'][i] = infosellerslist['seller_reputation']['power_seller_status']
    df_completo['period'][i] = infosellerslist['seller_reputation']['transactions']['period']
    df_completo['total'][i] = infosellerslist['seller_reputation']['transactions']['total']

# Creamos diccioarios vacíos para guardar dataframes separados por categoria
dfs_separados = {}
average_price_per_seller = {}
products_per_seller = {}
condition_per_seller = {}

# Iteramos por categoría para realizar el análisis exploratorio de datos
for i in dfcategories['name'].unique():
    dfs_separados[f'df_{i}'] = df_completo[df_completo['name_category'] == i]
    
    # Calculamos la media de precio por vendedor
    average_price_per_seller[f'df_{i}'] = dfs_separados[f'df_{i}'].groupby(['seller_nickname'])['price'].mean().sort_values(ascending=False)
    
    # Graficamos la distribución de precios promedio de los primeros 20 vendedores
    plt.figure(figsize=(10, 8))
    if not average_price_per_seller[f'df_{i}'].empty:
        average_price_per_seller[f'df_{i}'].head(20).plot(kind='bar')
        plt.title(f'Precio Promedio por Vendedor - Top 20 {i}')
        plt.xlabel('Vendedor (Nickname)')
        plt.ylabel('Precio Promedio')
        plt.xticks(rotation=90)
        plt.show()
    
    # Contamos cuántos productos tiene listado cada vendedor
    products_per_seller[f'df_{i}'] = dfs_separados[f'df_{i}'].groupby('seller_nickname').size().sort_values(ascending=False)
    
    # Graficamos los top 20 vendedores con más productos listados
    plt.figure(figsize=(10, 8))
    if not products_per_seller[f'df_{i}'].empty:
        products_per_seller[f'df_{i}'].head(20).plot(kind='bar', color='green')
        plt.title(f'Cantidad de Productos por Vendedor - Top 20 {i}')
        plt.xlabel('Vendedor (Nickname)')
        plt.ylabel('Cantidad de Productos')
        plt.xticks(rotation=90)
        plt.show()
    
    # Creamos un boxplot para ver la relación entre el envío gratis y los precios
    plt.figure(figsize=(10, 6))
    if not dfs_separados[f'df_{i}'].empty:
        sns.boxplot(x='free_shipping', y='price', data=dfs_separados[f'df_{i}'])
        plt.title(f'Impacto del Envío Gratis en los Precios de los Productos {i}')
        plt.xlabel('Envío Gratis')
        plt.ylabel('Precio')
        plt.show()
    
    # Contamos la condición de los productos (nuevo/usado) por vendedor
    condition_per_seller[f'df_{i}'] = dfs_separados[f'df_{i}'].groupby(['seller_nickname', 'condition']).size().unstack().fillna(0)
    
    # Graficamos para algunos vendedores seleccionados al azar o top vendedores
    if not condition_per_seller[f'df_{i}'].empty:
        num_samples = min(len(condition_per_seller[f'df_{i}']), 10)  # Tomar como máximo 10 muestras o menos si no hay suficientes
        condition_per_seller[f'df_{i}'].sample(num_samples).plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.title(f'Distribución de Condiciones de Productos por Vendedor {i}')
        plt.xlabel('Vendedor (Nickname)')
        plt.ylabel('Cantidad de Productos')
        plt.xticks(rotation=90)
        plt.legend(title='Condición del Producto')
        plt.show()

# Preparamos datos para entrenar un modelo GLM multinomial y evaluarlo con la variable de respuesta que será "power_seller_status"
# Las variables predictivas seran:
categorical_features = ['currency', 'condition', 'user_type', 'level_id', 'free_shipping']
numeric_features = ['price', 'available_quantity', 'total']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500))
])

X = {}
y = {}
X_train = {}
X_test = {}
y_train = {}
y_test = {}
y_pred = {}
reportdf = {}

# Iteramos sobre cada categoría
for i in dfcategories['name'].unique():
    
    # Asegurarse de llenar los valores faltantes
    for col in categorical_features + ['power_seller_status']:
        dfs_separados[f'df_{i}'][col].fillna('Vacio', inplace=True)
    for col in numeric_features:
        dfs_separados[f'df_{i}'][col].fillna(dfs_separados[f'df_{i}'][col].mean(), inplace=True)
    
    # Preparar X e y excluyendo correctamente las columnas que no son características
    X[f'df_{i}'] = dfs_separados[f'df_{i}'].drop(['power_seller_status', 'id_category', 'name_category', 'seller_id', 'seller_nickname', 'id_product', 'title', 'period'], axis=1)
    y[f'df_{i}'] = dfs_separados[f'df_{i}']['power_seller_status']

    # Dividir en conjuntos de entrenamiento y prueba
    X_train[f'df_{i}'], X_test[f'df_{i}'], y_train[f'df_{i}'], y_test[f'df_{i}'] = train_test_split(X[f'df_{i}'], y[f'df_{i}'], test_size=0.3, random_state=42)
    
    # Procesamiento y modelado para subconjuntos con suficiente variabilidad
    if len(dfs_separados[f'df_{i}']['power_seller_status'].unique()) < 2:
        print(f"Skipping category {i} due to insufficient class variability.")
        continue

    # Entrenar y predecir
    pipeline.fit(X_train[f'df_{i}'], y_train[f'df_{i}'])
    y_pred[f'df_{i}'] = pipeline.predict(X_test[f'df_{i}'])

    # Mostrar reporte de clasificación para ver que tan buenos fueron los resultados
    report = classification_report(y_test[f'df_{i}'], y_pred[f'df_{i}'], output_dict=True)
    reportdf[f'df_{i}'] = pd.DataFrame(report).transpose()
