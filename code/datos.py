# standard library imports
import csv
import datetime as dt
import json
import os
from ssl import SSLError
import statistics
import time

# instalar con pip para funcionamiento
import numpy as np
import pandas as pd
import requests

pd.set_option("display.max_columns", 100)

def get_request(url, parameters=None):
    """
    Devuelve la respuesta con formato json de una petición get utilizando parámetros opcionales.
    
    Parámetros
    ----------
    url : cadena
    parámetros : {'parámetro': 'valor'}
        parámetros a pasar como parte de la petición get
    
    Devuelve
    -------
    datos_json
        respuesta con formato json (tipo dict)
    """
    try:
        response = requests.get(url=url, params=parameters)
    except SSLError as s:
        print('SSL Error:', s)
        
        for i in range(5, 0, -1):
            print('\rWaiting... ({})'.format(i), end='')
            time.sleep(1)
        print('\rRetrying.' + ' '*10)
        
        # inténtalo de nuevo recusivamente
        return get_request(url, parameters)
    
    if response:
        return response.json()
    else:
        # la respuesta es ninguna suele significar que hay demasiadas peticiones. Espere y vuelva a intentarlo 
        print('No response, waiting 10 seconds...')
        time.sleep(10)
        print('Retrying.')
        return get_request(url, parameters)




url = "https://steamspy.com/api.php"
parameters = {"request": "all"}

# solicitar 'all' a steam spy y parsear en dataframe
json_data = get_request(url, parameters=parameters)
steam_spy_all = pd.DataFrame.from_dict(json_data, orient='index')

#  generar app_list ordenada a partir de los datos de steamspy
app_list = steam_spy_all[['appid', 'name']].sort_values('appid').reset_index(drop=True)

#CREACION DE CARPETAS DESCOMENTAR LA PRIMERA VEZ QUE SE EJECUTE YA LUEGO VUELVE A COMENTAR
#os.mkdir("../data")
#os.mkdir("../data/download")

app_list.to_csv('../data/download/app_list.csv', index=False)

# en lugar de leer del csv almacenado
app_list = pd.read_csv('../data/download/app_list.csv')

# mostrar las primeras filas
app_list.head()


def get_app_data(start, stop, parser, pause):
    """Return list of app data generated from parser.
    
    parser : function to handle request
    """
    app_data = []
       # iterar a través de cada fila de app_list, confinado por start y stop
    for index, row in app_list[start:stop].iterrows():
        print('Current index: {}'.format(index), end='\r')
        
        appid = row['appid']
        name = row['name']

         # recuperar los datos de aplicación de una fila, gestionados por el analizador sintáctico suministrado, y añadirlos a la lista
        data = parser(appid, name)
        app_data.append(data)

        time.sleep(pause)  # prevenir la sobrecarga de la api con peticiones
    
    return app_data


def process_batches(parser, app_list, download_path, data_filename, index_filename,
                    columns, begin=0, end=-1, batchsize=100, pause=1):
    """  
    Procesa los datos de la aplicación por lotes, escribiéndolos directamente en un archivo.
    
    parser : función personalizada para formatear la solicitud
    app_list : dataframe de appid y nombre
    download_path : ruta para almacenar los datos
    data_filename : nombre de archivo para guardar los datos de la aplicación
    index_filename : nombre de archivo para almacenar el índice más alto escrito
    columns : nombres de columnas para el archivo
    
    Argumentos de palabras clave:
    
    begin : índice inicial (se obtiene de index_filename, por defecto 0)
    end : índice final (por defecto al final de app_list)
    batchsize : número de aplicaciones a escribir en cada lote (por defecto 100)
    pause : tiempo de espera después de cada petición api (por defecto 1)
    
    retornos: ninguno

    """
    print('Starting at index {}:\n'.format(begin))
    
    # 
    if end == -1:
        end = len(app_list) + 1
    
    # por defecto, procesa todas las aplicaciones de app_list
    batches = np.arange(begin, end, batchsize)
    batches = np.append(batches, end)
    
    apps_written = 0
    batch_times = []
    
    for i in range(len(batches) - 1):
        start_time = time.time()
        
        start = batches[i]
        stop = batches[i+1]
        
        app_data = get_app_data(start, stop, parser, pause)
        
        rel_path = os.path.join(download_path, data_filename)
        
        # escribir los datos de la aplicación en un archivo
        with open(rel_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            
            for j in range(3,0,-1):
                print("\rAbout to write data, don't stop script! ({})".format(j), end='')
                time.sleep(0.5)
            
            writer.writerows(app_data)
            print('\rExported lines {}-{} to {}.'.format(start, stop-1, data_filename), end=' ')
            
        apps_written += len(app_data)
        
        idx_path = os.path.join(download_path, index_filename)
        
        #  escribir el último índice en un archivo
        with open(idx_path, 'w') as f:
            index = stop
            print(index, file=f)
            
        # tiempo de registro
        end_time = time.time()
        time_taken = end_time - start_time
        
        batch_times.append(time_taken)
        mean_time = statistics.mean(batch_times)
        
        est_remaining = (len(batches) - i - 2) * mean_time
        
        remaining_td = dt.timedelta(seconds=round(est_remaining))
        time_td = dt.timedelta(seconds=round(time_taken))
        mean_td = dt.timedelta(seconds=round(mean_time))
        
        print('Batch {} time: {} (avg: {}, remaining: {})'.format(i, time_td, mean_td, remaining_td))
            
    print('\nProcessing batches complete. {} apps written'.format(apps_written))


def reset_index(download_path, index_filename):
    """Reset index in file to 0."""
    rel_path = os.path.join(download_path, index_filename)
    
    with open(rel_path, 'w') as f:
        print(0, file=f)
        

def get_index(download_path, index_filename):
    """Retrieve index from file, returning 0 if file not found."""
    try:
        rel_path = os.path.join(download_path, index_filename)

        with open(rel_path, 'r') as f:
            index = int(f.readline())
    
    except FileNotFoundError:
        index = 0
        
    return index


def prepare_data_file(download_path, filename, index, columns):
    """Crea el fichero y escribe las cabeceras si el índice es 0."""
    if index == 0:
        rel_path = os.path.join(download_path, filename)

        with open(rel_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def parse_steam_request(appid, name):
    """Parser único para manejar datos de la API de la tienda de Steam.
    
    Devuelve : datos con formato json (tipo dict)
    """
    url = "http://store.steampowered.com/api/appdetails/"
    parameters = {"appids": appid}
    
    json_data = get_request(url, parameters=parameters)
    json_app_data = json_data[str(appid)]
    
    if json_app_data['success']:
        data = json_app_data['data']
    else:
        data = {'name': name, 'steam_appid': appid}
        
    return data


# Set file parameters
download_path = '../data/download'
steam_app_data = 'steam_app_data.csv'
steam_index = 'steam_index.txt'

steam_columns = [
    'type', 'name', 'steam_appid', 'required_age', 'is_free', 'controller_support',
    'dlc', 'detailed_description', 'about_the_game', 'short_description', 'fullgame',
    'supported_languages', 'header_image', 'website', 'pc_requirements', 'mac_requirements',
    'linux_requirements', 'legal_notice', 'drm_notice', 'ext_user_account_notice',
    'developers', 'publishers', 'demos', 'price_overview', 'packages', 'package_groups',
    'platforms', 'metacritic', 'reviews', 'categories', 'genres', 'screenshots',
    'movies', 'recommendations', 'achievements', 'release_date', 'support_info',
    'background', 'content_descriptors'
]

# Sobrescribe el último índice para la demostración (normalmente almacenaría el índice más alto para poder continuar a través de las sesiones)
reset_index(download_path, steam_index)

# Recuperar el último índice descargado del archivo
index = get_index(download_path, steam_index)

# Borrar o crear fichero de datos y escribir cabeceras si el índice es 0
prepare_data_file(download_path, steam_app_data, index, steam_columns)

# Establecer el final y el tamaño de los trozos para la demostración - eliminar para ejecutar a través de toda la lista de aplicaciones
process_batches(
    parser=parse_steam_request,
    app_list=app_list,
    download_path=download_path,
    data_filename=steam_app_data,
    index_filename=steam_index,
    columns=steam_columns, #se ajusta el valor de cuantos juegos se quieren obtener
    begin=index,
    end=100,
    batchsize=5
)

# inspeccionar los datos descargados
pd.read_csv('../data/download/steam_app_data.csv').head()

def parse_steamspy_request(appid, name):
    """Parser to handle SteamSpy API data."""
    url = "https://steamspy.com/api.php"
    parameters = {"request": "appdetails", "appid": appid}
    
    json_data = get_request(url, parameters)
    return json_data


# establecer archivos y columnas
download_path = '../data/download'
steamspy_data = 'steamspy_data.csv'
steamspy_index = 'steamspy_index.txt'

steamspy_columns = [
    'appid', 'name', 'developer', 'publisher', 'score_rank', 'positive',
    'negative', 'userscore', 'owners', 'average_forever', 'average_2weeks',
    'median_forever', 'median_2weeks', 'price', 'initialprice', 'discount',
    'languages', 'genre', 'ccu', 'tags'
]

reset_index(download_path, steamspy_index)
index = get_index(download_path, steamspy_index)

# Borrar archivo de datos si el índice es 0
prepare_data_file(download_path, steamspy_data, index, steamspy_columns)

process_batches(
    parser=parse_steamspy_request,
    app_list=app_list,
    download_path=download_path, 
    data_filename=steamspy_data,
    index_filename=steamspy_index,
    columns=steamspy_columns, #se ajusta el valor de cuantos juegos se quieren obtener
    begin=index,
    end=100,
    batchsize=5,
    pause=0.3
)

# inspeccionar los datos de steamspy descargados
pd.read_csv('../data/download/steamspy_data.csv').head()





