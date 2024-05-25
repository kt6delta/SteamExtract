# importaciones de la biblioteca estándar
from ast import literal_eval
import itertools
# import time
# import re

# importaciones de terceros
import numpy as np
import pandas as pd

# personalizaciones
pd.set_option("display.max_columns", 100)

raw_steamspy_data = pd.read_csv('../data/download/steamspy_data.csv')

tags = raw_steamspy_data['tags']
eval_row = literal_eval(tags[0])
tags[tags == '[]'].shape[0]
parsed_tags = tags.apply(lambda x: literal_eval(x))
cols = set(itertools.chain(*parsed_tags))

def parse_tags(x):
    x = literal_eval(x)
    
    if isinstance(x, dict):
        return x
    elif isinstance(x, list):
        return {}
    else:
        raise TypeError('Se ha encontrado algo distinto de dict o list')
        
parsed_tags = tags.apply(parse_tags)        
tag_data = pd.DataFrame()

for col in sorted(cols):
    # normalizar los nombres de las columnas
    col_name = col.lower().replace(' ', '_').replace('-', '_').replace("'", "")

    # comprueba si la columna está en el diccionario de etiquetas de la fila y devuelve ese valor si lo está, o 0 si no lo está
    tag_data[col_name] = parsed_tags.apply(lambda x: x[col] if col in x.keys() else 0)
    


def parse_tags(x):
    x = literal_eval(x)

    if isinstance(x, dict):
        return ';'.join(list(x.keys())[:3])
    else:
        return np.nan
    
tags.apply(parse_tags)
owners = raw_steamspy_data['owners']
owners_split = owners.str.replace(',', '').str.split(' .. ')
owners_split.apply(lambda x: int(x[0])).head()
owners_split.apply(lambda x: (int(x[0]) + int(x[1])) // 2).head()
owners.str.replace(',', '').str.replace(' .. ', '-').head()

def process_tags(df, export=False):
    if export:         
        tag_data = df[['appid', 'tags']].copy()
        
        def parse_export_tags(x):
            x = literal_eval(x)
            if isinstance(x, dict):
                return x
            elif isinstance(x, list):
                return {}
            else:
                raise TypeError('Se ha encontrado algo distinto de dict o lists')
        tag_data['tags'] = tag_data['tags'].apply(parse_export_tags)
        cols = set(itertools.chain(*tag_data['tags']))
        for col in sorted(cols):
            col_name = col.lower().replace(' ', '_').replace('-', '_').replace("'", "")
            tag_data[col_name] = tag_data['tags'].apply(lambda x: x[col] if col in x.keys() else 0)
        tag_data = tag_data.drop('tags', axis=1)
        tag_data.to_csv('../data/exports/steamspy_tag_data.csv', index=False)
        print("Exported tag data to '../data/exports/steamspy_tag_data.csv'")        
        
    def parse_tags(x):
        x = literal_eval(x)        
        if isinstance(x, dict):
            return ';'.join(list(x.keys())[:3])
        else:
            return np.nan    
    df['tags'] = df['tags'].apply(parse_tags)    
    # las filas con etiquetas nulas parecen haber sido sustituidas por una nueva versión, por lo que deben eliminarse (por ejemplo, isla muerta)
    df = df[df['tags'].notnull()]    
    return df

def process(df):
    df = df.copy()    
    # tratar los valores omitidos
    df = df[(df['name'].notnull()) & (df['name'] != 'none')]
    df = df[df['developer'].notnull()]
    df = df[df['languages'].notnull()]
    df = df[df['price'].notnull()]    
    # remove unwanted columns
    df = df.drop([
        'genre', 'developer', 'publisher', 'score_rank', 'userscore', 'average_2weeks',
        'median_2weeks', 'price', 'initialprice', 'discount', 'ccu'
    ], axis=1)    
    # conservar las etiquetas principales, exportar los datos completos de las etiquetas a un archivo
    df = process_tags(df, export=True)    
    # reformatear la columna de propietarios
    df['owners'] = df['owners'].str.replace(',', '').str.replace(' .. ', '-')    
    return df

steamspy_data = process(raw_steamspy_data)
steamspy_data.to_csv('../data/exports/steamspy_clean.csv', index=False)
steam_data = pd.read_csv('../data/exports/steam_data_clean.csv')
    
merged = steam_data.merge(steamspy_data, left_on='steam_appid', right_on='appid', suffixes=('', '_steamspy'))
# eliminar columnas superpuestas
steam_clean = merged.drop(['name_steamspy', 'languages', 'steam_appid'], axis=1)
# reindexar para reordenar las columnas
steam_clean = steam_clean[[
    'appid',
    'name',
    'release_date',
    'english',
    'developer',
    'publisher',
    'platforms',
    'required_age',
    'categories',
    'genres',
    'tags',    
    'positive',
    'negative',
    'average_forever',
    'median_forever',
    'owners',
    'price'
]]

steam_clean = steam_clean.rename({
    'tags': 'steamspy_tags',
    'positive': 'positive_ratings',
    'negative': 'negative_ratings',
    'average_forever': 'average_playtime',
    'median_forever': 'median_playtime'
}, axis=1)

steam_clean.to_csv('../data/steam.csv', index=False)