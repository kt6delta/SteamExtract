#  importaciones de la biblioteca estándar
from ast import literal_eval
from multiprocessing import Process
from IPython.display import display
import itertools
import time
import re
import os

# importaciones de terceros
import numpy as np
import pandas as pd

# personalizaciones
pd.set_option("display.max_columns", 100)

# leer los datos descargados
raw_steam_data = pd.read_csv('../data/download/steam_app_data.csv')

# imprimir el número de filas y columnas
print('Rows:', raw_steam_data.shape[0])
print('Columns:', raw_steam_data.shape[1])

#ver las cinco primeras filas 
raw_steam_data.head()
null_counts = raw_steam_data.isnull().sum()
null_counts

threshold = raw_steam_data.shape[0] // 2

print('Drop columns with more than {} missing rows'.format(threshold))
print()

drop_rows = raw_steam_data.columns[null_counts > threshold]

print('Columns to drop: {}'.format(list(drop_rows)))
#print('Rows to remove:', raw_steam_data[raw_steam_data['type'].isnull()].shape[0])

# vista previa de las filas con datos de tipo que faltan
raw_steam_data[raw_steam_data['type'].isnull()].head(3)

#print('Rows to remove:', raw_steam_data[raw_steam_data['type'].isnull()].shape[0])

# vista previa de las filas con datos de tipo que faltan
raw_steam_data[raw_steam_data['type'].isnull()].head(3)

duplicate_rows = raw_steam_data[raw_steam_data.duplicated()]

#print('Duplicate rows to remove:', duplicate_rows.shape[0])


def drop_null_cols(df, thresh=0.5):
    """Elimina las columnas con más de una determinada proporción de valores perdidos (Predeterminado: 50%)."""
    cutoff_count = len(df) * thresh    
    return df.dropna(thresh=cutoff_count, axis=1)


def process_name_type(df):
    """Eliminar valores nulos en las columnas name y type, y eliminar la columna type."""
    df = df[df['type'].notnull()]    
    df = df[df['name'].notnull()]
    df = df[df['name'] != 'none']    
    df = df.drop('type', axis=1)    
    return df
    

def process(df):
    """Conjunto de datos de proceso. Eventualmente contendrá llamadas a todas las funciones que escribamos"""    
    # Copie el marco de datos de entrada para evitar modificar accidentalmente los datos originales
    df = df.copy()    
    # Eliminar filas duplicadas - todos los appids deben ser únicos
    df = df.drop_duplicates()    
    # Eliminar columnas con más de un 50% de valores nulos
    df = drop_null_cols(df)    
    # Procesar el resto de columnas
    df = process_name_type(df)    
    return df

#print(raw_steam_data.shape)
initial_processing = process(raw_steam_data)
#print(initial_processing.shape)
initial_processing.head()

def process_age(df):
    """Las clasificaciones de formato en la columna de edad deben ajustarse al sistema de clasificación por edades PEGI."""
    # PEGI Age ratings: 3, 7, 12, 16, 18
    cut_points = [-1, 0, 3, 7, 12, 16, 2000]
    label_values = [0, 3, 7, 12, 16, 18]    
    df['required_age'] = pd.cut(df['required_age'], bins=cut_points, labels=label_values)    
    return df


age_df = process_age(initial_processing)
age_df['required_age'].value_counts().sort_index()



platforms_first_row = age_df['platforms'].iloc[0]
eval_first_row = literal_eval(platforms_first_row)

# crear cadena de claves, unidas por un punto y coma
';'.join(eval_first_row.keys())

platforms = {'windows': True, 'mac': True, 'linux': False}

# comprensión de la lista
#print([x for x in platforms.keys() if platforms[x]])
# uso de la comprensión de listas en join
';'.join(x for x in platforms.keys() if platforms[x])

def process_platforms(df):
    """Dividir la columna de plataformas en columnas booleanas separadas para cada plataforma."""
    # evalúa valores en la columna de plataformas, por lo que puede indexar en diccionarios
    df = df.copy()
    
    def parse_platforms(x):
        
        d = literal_eval(x)
        
        return ';'.join(platform for platform in d.keys() if d[platform])
    
    df['platforms'] = df['platforms'].apply(parse_platforms)
    
    return df

platforms_df = process_platforms(age_df)
platforms_df['platforms'].value_counts()

free_and_null_price = platforms_df[(platforms_df['is_free']) & (platforms_df['price_overview'].isnull())]
free_and_null_price.shape[0]

not_free_and_null_price = platforms_df[(platforms_df['is_free'] == False) & (platforms_df['price_overview'].isnull())]
not_free_and_null_price.head()

def print_steam_links(df):
    """Imprimir enlaces a la página de la tienda para aplicaciones en un marco de datos."""
    url_base = "https://store.steampowered.com/app/"
    
    for i, row in df.iterrows():
        appid = row['steam_appid']
        name = row['name']
        
        print(name + ':', url_base + str(appid))
        
def process_price(df):
    df = df.copy()        
    def parse_price(x):
        if x is not np.nan:
            return literal_eval(x)
        else:
            return {'currency': 'COP', 'initial': -1}
    
    # evaluar como diccionario y poner a -1 si falta
    df['price_overview'] = df['price_overview'].apply(parse_price)
    
    # Crear columnas a partir de valores monetarios e iniciales
    df['currency'] = df['price_overview'].apply(lambda x: x['currency'])
    df['price'] = df['price_overview'].apply(lambda x: x['initial'])
    
    # Fijar el precio de los juegos gratuitos en 0
    df.loc[df['is_free'], 'price'] = 0
    
    return df

price_data = process_price(platforms_df)[['name', 'currency', 'price']]
price_data.head()
price_data[price_data['currency'] != 'COP']

def process_price(df):
    """Procesar la columna precio_vista general en una columna de precio formateada."""
    df = df.copy()
    
    def parse_price(x):
        if x is not np.nan:
            return literal_eval(x)
        else:
            return {'currency': 'COP', 'initial': -1}
    
    # evaluar como diccionario y poner a -1 si falta
    df['price_overview'] = df['price_overview'].apply(parse_price)
    
    # crear columnas a partir de moneda y valores iniciales
    df['currency'] = df['price_overview'].apply(lambda x: x['currency'])
    df['price'] = df['price_overview'].apply(lambda x: x['initial'])
    
    # fijar el precio de los juegos gratuitos en 0
    df.loc[df['is_free'], 'price'] = 0
    
    # eliminar filas no COP
    df = df[df['currency'] == 'COP']    
    
    # eliminar las filas en las que el precio es -1
    df = df[df['price'] != -1]
       
    # eliminar las columnas que ya no sean necesarias
    df = df.drop(['is_free', 'currency', 'price_overview'], axis=1)    
    return df

price_df = process_price(platforms_df)
price_df[['name', 'price']].head()

#with pd.option_context("display.max_colwidth", 500):
    #display(price_df[['steam_appid', 'packages', 'package_groups', 'price']].head(3))
    
missing_price_and_package = price_df[(price_df['price'] == -1) & (price_df['package_groups'] == "[]")]
missing_price_have_package = price_df.loc[(price_df['price'] == -1) & (price_df['package_groups'] != "[]"), ['name', 'steam_appid', 'package_groups', 'price']]

def process_language(df):
    """Procesar la columna supported_languages en una columna booleana 'is english'."""
    df = df.copy()
    
    # eliminar las filas con datos lingüísticos no disponibles
    df = df.dropna(subset=['supported_languages'])
    
    df['english'] = df['supported_languages'].apply(lambda x: 1 if 'english' in x.lower() else 0)
    df = df.drop('supported_languages', axis=1)
    
    return df


language_df = process_language(price_df)
#language_df[['name', 'english']].head()

no_dev = language_df[language_df['developers'].isnull()]
no_pub = language_df[language_df['publishers'] == "['']"]
no_dev_or_pub = language_df[(language_df['developers'].isnull()) & (language_df['publishers'] == "['']")]

language_df[['developers', 'publishers']].iloc[24:28]

def process_developers_and_publishers(df):
    # eliminar filas con datos que faltan
    df = df[(df['developers'].notnull()) & (df['publishers'] != "['']")].copy()
    
    for col in ['developers', 'publishers']:
        df[col] = df[col].apply(lambda x: literal_eval(x))
        
        # filtrar dataframe a filas con listas mayores que 1, y almacenar el número de filas
        num_rows = df[df[col].str.len() > 1].shape[0]
        
        #print('Rows in {} column with multiple values:'.format(col), num_rows)

process_developers_and_publishers(language_df)
', '.join(['one item'])
', '.join(['multiple', 'different', 'items'])
language_df.loc[language_df['developers'].str.contains(",", na=False), ['steam_appid', 'developers', 'publishers']].head(4)
language_df.loc[language_df['developers'].str.contains(";", na=False), ['steam_appid', 'developers', 'publishers']]
language_df[(language_df['publishers'] == "['NA']") | (language_df['publishers'] == "['N/A']")].shape[0]

def process_developers_and_publishers(df):
    """Analizar columnas como cadenas separadas por punto y coma."""
    # eliminar las filas con datos que faltan (~ significa que no)
    df = df[(df['developers'].notnull()) & (df['publishers'] != "['']")].copy()
    df = df[~(df['developers'].str.contains(';')) & ~(df['publishers'].str.contains(';'))]
    df = df[(df['publishers'] != "['NA']") & (df['publishers'] != "['N/A']")]
    
    # crear una lista para cada
    df['developer'] = df['developers'].apply(lambda x: ';'.join(literal_eval(x)))
    df['publisher'] = df['publishers'].apply(lambda x: ';'.join(literal_eval(x)))

    df = df.drop(['developers', 'publishers'], axis=1)
    
    return df

dev_pub_df = process_developers_and_publishers(language_df)
dev_pub_df[['name', 'steam_appid', 'developer', 'publisher']].head()

example_category = "[{'id': 1, 'description': 'Multi-player'}, {'id': 36, 'description': 'Online Multi-Player'}, {'id': 37, 'description': 'Local Multi-Player'}]"
#[x['description'] for x in literal_eval(example_category)]

def process_categories_and_genres(df):
    df = df.copy()
    df = df[(df['categories'].notnull()) & (df['genres'].notnull())]
    
    for col in ['categories', 'genres']:
        df[col] = df[col].apply(lambda x: ';'.join(item['description'] for item in literal_eval(x)))
    
    return df

cat_gen_df = process_categories_and_genres(dev_pub_df)

def process_achievements_and_descriptors(df):
    """Parse" como número total de logros."""
    df = df.copy()
    
    df = df.drop('content_descriptors', axis=1)
    
    def parse_achievements(x):
        if x is np.nan:
            # faltan datos, se supone que no tiene logros
            return 0
        else:
            # else tiene datos, por lo que puede extraer y devolver el número bajo el total
            return literal_eval(x)['total']
        
    #df['achievements'] = df['achievements'].apply(parse_achievements)
    
    return df


achiev_df = process_achievements_and_descriptors(cat_gen_df)

def process(df):
    """Conjunto de datos de proceso. Eventualmente contendrá llamadas a todas las funciones que escribamos."""
    
    # Copie el marco de datos de entrada para evitar modificar accidentalmente los datos originales
    df = df.copy()
    
    #  Copy the input data frame to avoid accidentally modifying the original data.
    df = df.drop_duplicates()
    
    # Eliminar columnas con más de un 50% de valores nulos
    df = drop_null_cols(df)
    
    # Columnas de proceso
    df = process_name_type(df)
    df = process_age(df)
    df = process_platforms(df)
    df = process_price(df)
    df = process_language(df)
    df = process_developers_and_publishers(df)
    df = process_categories_and_genres(df)
    df = process_achievements_and_descriptors(df)
    
    return df

partially_clean = process(raw_steam_data)

partially_clean[partially_clean['detailed_description'].str.len() <= 20]

#CREACION DE CARPETAS DESCOMENTAR LA PRIMERA EJECUCION LUEGO COMENTARIOR DE NUEVO
#os.mkdir("../data/exports")
def export_data(df, filename):
    """Exportar marco de datos a archivo csv, nombre de archivo precedido de 'steam_'.    
    filename : str sin extensión de archivo
    """
    
    filepath = '../data/exports/steam_' + filename + '.csv'
    
    df.to_csv(filepath, index=False)    
    print_name = filename.replace('_', ' ')
    print("Exported {} to '{}'".format(print_name, filepath))


def process_descriptions(df, export=False):
    """Exporte las descripciones a un archivo csv externo y elimine estas columnas."""
    # eliminar las filas en las que faltan datos de descripción
    df = df[df['detailed_description'].notnull()].copy()
    
    # eliminar filas con descripción inusualmente pequeña
    df = df[df['detailed_description'].str.len() > 20]
    
    # por defecto no exportamos, útil si se llama a la función más tarde
    if export:
        # create dataframe of description columns 
        description_data = df[['steam_appid', 'detailed_description', 'about_the_game', 'short_description']]
        
        export_data(description_data, filename='description_data')
    
    # eliminar las columnas de descripción del marco de datos principal
    df = df.drop(['detailed_description', 'about_the_game', 'short_description'], axis=1)    
    return df

desc_df = process_descriptions(partially_clean, export=True)

image_cols = ['header_image', 'screenshots', 'background']
for col in image_cols:
    print(col+':', desc_df[col].isnull().sum())

desc_df[image_cols].head()
no_screenshots = desc_df[desc_df['screenshots'].isnull()]
print_steam_links(no_screenshots)

def process_media(df, export=False):
    """Elimina las columnas multimedia del marco de datos, opcionalmente exportándolas primero a csv."""
    df = df[df['screenshots'].notnull()].copy()
    
    if export:
        media_data = df[['steam_appid', 'header_image', 'screenshots', 'background']]        
        export_data(media_data, 'media_data')
        
    df = df.drop(['header_image', 'screenshots', 'background'], axis=1)    
    return df

media_df = process_media(desc_df, export=True)

"""
print('Before removing data:\n')
achiev_df.info(verbose=False, memory_usage="deep")
print('\nData with descriptions and media removed:\n')
media_df.info(verbose=False, memory_usage="deep")
"""

#with pd.option_context("display.max_colwidth", 100): # ensures strings not cut short
 #   display(media_df[['name', 'website', 'support_info']][75:80])
 
def process_info(df, export=False):
    """Elimina la información de soporte del marco de datos, opcionalmente exportándola de antemano."""
    if export:
        support_info = df[['steam_appid', 'website', 'support_info']].copy()        
        support_info['support_info'] = support_info['support_info'].apply(lambda x: literal_eval(x))
        support_info['support_url'] = support_info['support_info'].apply(lambda x: x['url'])
        support_info['support_email'] = support_info['support_info'].apply(lambda x: x['email'])        
        support_info = support_info.drop('support_info', axis=1)
        
        # conservar sólo las filas que contengan al menos un dato
        support_info = support_info[(support_info['website'].notnull()) | (support_info['support_url'] != '') | (support_info['support_email'] != '')]
        export_data(support_info, 'support_info')
    
    df = df.drop(['website', 'support_info'], axis=1)    
    return df

info_df = process_info(media_df, export=True)

requirements_cols = ['pc_requirements', 'mac_requirements', 'linux_requirements']
for col in ['mac_requirements', 'linux_requirements']:
    platform = col.split('_')[0]
    #print(platform+':', info_df[(info_df[col] == '[]') & (info_df['platforms'].str.contains(platform))].shape[0])
    

print('windows:', info_df[(info_df['pc_requirements'] == '[]') & (info_df['platforms'].str.contains('windows'))].shape[0])
missing_windows_requirements = info_df[(info_df['pc_requirements'] == '[]') & (info_df['platforms'].str.contains('windows'))]
view_requirements = info_df['pc_requirements'].iloc[[0, 2, 15]].copy()

view_requirements = (view_requirements
                         .str.replace(r'\\[rtn]', '')
                         .str.replace(r'<[pbr]{1,2}>', ' ')
                         .str.replace(r'<[\/"=\w\s]+>', '')
                    )
def process_requirements(df, export=False):
    if export:
        requirements = df[['steam_appid', 'pc_requirements', 'mac_requirements', 'linux_requirements']].copy()
        
        # eliminar filas con requisitos de pc que faltan
        requirements = requirements[requirements['pc_requirements'] != '[]']
        
        requirements['requirements_clean'] = (requirements['pc_requirements']
                                                  .str.replace(r'\\[rtn]', '')
                                                  .str.replace(r'<[pbr]{1,2}>', ' ')
                                                  .str.replace(r'<[\/"=\w\s]+>', '')
                                             )
        
        requirements['requirements_clean'] = requirements['requirements_clean'].apply(lambda x: literal_eval(x))
        
        # dividir el mínimo y el recomendado en columnas separadas
        requirements['minimum'] = requirements['requirements_clean'].apply(lambda x: x['minimum'].replace('Minimum:', '').strip() if 'minimum' in x.keys() else np.nan)
        requirements['recommended'] = requirements['requirements_clean'].apply(lambda x: x['recommended'].replace('Recommended:', '').strip() if 'recommended' in x.keys() else np.nan)
        
        requirements = requirements.drop('requirements_clean', axis=1)
        
        export_data(requirements, 'requirements_data')
        
    df = df.drop(['pc_requirements', 'mac_requirements', 'linux_requirements'], axis=1)
    
    return df

reqs_df = process_requirements(info_df, export=True)
reqs_df.to_csv('../data/exports/steam_partially_clean.csv', index=False)

def process_release_date(df):
    df = df.copy()
    
    def eval_date(x):
        x = literal_eval(x)
        if x['coming_soon']:
            return '' # devuelve una cadena en blanco para que se pueda eliminar lo que falte al final
        else:
            return x['date']
    
    df['release_date'] = df['release_date'].apply(eval_date)
    
    def parse_date(x):
        if re.search(r'[\d]{1,2} [A-Za-z]{3}, [\d]{4}', x):
            return x.replace(',', '')
        elif re.search(r'[A-Za-z]{3} [\d]{4}', x):
            return '1 ' + x
        elif x == '':
            return np.nan
            
    df['release_date'] = df['release_date'].apply(parse_date)
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d %b %Y', errors='coerce')
    
    df = df[df['release_date'].notnull()]
    
    return df




def process(df):
    """Conjunto de datos de proceso. Eventualmente contendrá llamadas a todas las funciones que escribamos."""
    
    # Copie el marco de datos de entrada para evitar modificar accidentalmente los datos originales
    df = df.copy()
    
    # Eliminar filas duplicadas - todos los appids deben ser únicos
    df = df.drop_duplicates()
    
    # Eliminar columnas con más de un 50% de valores nulos
    df = drop_null_cols(df)
    
    # Columnas de proceso
    df = process_name_type(df)
    df = process_age(df)
    df = process_platforms(df)
    df = process_price(df)
    df = process_language(df)
    df = process_developers_and_publishers(df)
    df = process_categories_and_genres(df)
    df = process_achievements_and_descriptors(df)  
    df = process_release_date(df)
    
    # Columnas de proceso que exportan datos
    df = process_descriptions(df, export=True)
    df = process_media(df, export=True)
    df = process_info(df, export=True)
    df = process_requirements(df, export=True)
    
    return df

steam_data = process(raw_steam_data)
limpio=drop_null_cols(steam_data)
limpio.to_csv('../data/exports/steam_data_clean.csv', index=False)





