import itertools
import re

# importaciones de terceros
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Personalizaciones
pd.set_option("display.max_columns", 1000)
plt.style.use('default')
plt.rcdefaults()
sns.set()

pd.read_csv('../data/steam.csv').head()

def remove_non_english(df):
    # mantener solo las filas marcadas como compatibles con inglés
    df = df[df['english'] == 1].copy()    
    # mantener filas que no contienen 3 o más caracteres no ASCII sucesivos
    df = df[~df['name'].str.contains('[^\u0001-\u007F]{3,}')]    
    # eliminar la columna de inglés, ahora redundante
    df = df.drop('english', axis=1)    
    return df

def calc_rating(row):
    """Calcular la puntuación de calificación basada en el método de SteamDB."""
    import math

    pos = row['positive_ratings']
    neg = row['negative_ratings']

    total_reviews = pos + neg
    average = pos / total_reviews
    
    # tira la puntuación hacia 50, tira más fuertemente para juegos con pocas reseñas
    score = average - (average*0.5) * 2**(-math.log10(total_reviews + 1))

    return score * 100

def get_unique(series):
    """Obtener valores únicos de una serie de Pandas que contiene cadenas delimitadas por punto y coma."""
    return set(list(itertools.chain(*series.apply(lambda x: [c for c in x.split(';')]))))

def process_cat_gen_tag(df):
    """Procesar las columnas de categorías, géneros y etiquetas de steamspy."""
    # obtener todos los nombres de categorías únicos
    cat_cols = get_unique(df['categories'])
    
    
    cat_cols = [
        'Local Multi-Player',
        'MMO',        
        'Multi-player',
        'Online Co-op',
        'Online Multi-Player',
        'Single-player'
    ]
    
    # crear una nueva columna para cada categoría, con 1s indicando pertenencia y 0s para no miembros
    for col in sorted(cat_cols):
        col_name = re.sub(r'[\s\-\/]', '_', col.lower())
        col_name = re.sub(r'[()]', '', col_name)
        
        df[col_name] = df['categories'].apply(lambda x: 1 if col in x.split(';') else 0)
        
    # repetir para los nombres de columna de géneros (get_unique se usa para encontrar nombres de géneros únicos,

    gen_cols = get_unique(df['genres'])
    
    # solo manteniendo los géneros 'principales' similares a la tienda de steam store
    gen_cols = [
        
        'Action',
        'Adventure',        
        'Casual',        
        'Free to Play',        
        'Indie',
        'Massively Multiplayer',        
        'RPG',
        'Racing',        
        'Simulation',        
        'Sports',
        'Strategy'        
        
    ]
    
    gen_col_names = []
    
    # crear nuevas columnas para cada género con 1s para juegos de ese género
    for col in sorted(gen_cols):
        col_name = col.lower().replace('&', 'and').replace(' ', '_')
        gen_col_names.append(col_name)
        
        df[col_name] = df['genres'].apply(lambda x: 1 if col in x.split(';') else 0)
        
    
    # eliminar "no-juegos" basados en género
    # si una fila tiene todos ceros en las nuevas columnas de género, probablemente no es un juego, así que eliminar (principalmente software)
    gen_sums = df[gen_col_names].sum(axis=1)
    df = df[gen_sums > 0].copy()
    
   
        
    df = df.drop(['categories', 'steamspy_tags'], axis=1)
    
    return df
def pre_process():
    """Preprocesar el conjunto de datos de Steam para análisis exploratorio."""
    df = pd.read_csv('../data/steam.csv')
    
    # mantener solo en inglés
    df = remove_non_english(df)
    
    # mantener solo windows, y eliminar la columna de plataformas
    df = df[df['platforms'].str.contains('windows')].drop('platforms', axis=1).copy()
    
    # mantener el límite inferior de la columna de propietarios, como entero
    df['owners'] = df['owners'].str.split('-').apply(lambda x: x[0]).astype(int)
    
    # calcular calificación, así como una proporción simple para comparación
    df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
    df['rating_ratio'] = df['positive_ratings'] / df['total_ratings']
    df['rating'] = df.apply(calc_rating, axis=1)
    
    # convertir release_date a tipo datetime y crear una columna separada para release_year
    df['release_date'] = df['release_date'].astype('datetime64[ns]')
    df['release_year'] = df['release_date'].apply(lambda x: x.year)
    
    # procesar columnas de géneros, categorías y etiquetas de steamspy
    df = process_cat_gen_tag(df)
    
    return df

data = pre_process()

print('Verify no missing values:')
print(data.isnull().sum().value_counts())

data.head()

import warnings
warnings.filterwarnings('ignore')

# Crear una columna para dividir juegos gratuitos y pagados
data['type'] = 'Free'
data.loc[data['price'] > 0, 'type'] = 'Paid'

# asegurar que no haya 0s en las columnas a las que aplicaremos log
df = data[(data['owners'] > 0) & (data['total_ratings'] > 0)].copy()

eda_df = pd.DataFrame(zip(df['rating'],
                          np.log10(df['total_ratings']),
                          np.log10(df['owners']),
                          df['release_year'],
                          df.price,
                          df['type']
                         ),
                      columns=['Rating Score', 'Total Ratings (log)', 'Owners (log)', 'Release Year', 'Current Price', 'Type'])


#esta linea nos da un resumen general usando la libreria seaborn
#sns.pairplot(eda_df, hue='Type')
"QUIEN TIENE MEJORES CALIFICACIONES JUEGOS GRATIS VS JUEGOS PAGOS"

plt.show()

fig = plt.figure(figsize=(10,6))

dfa = data[data.owners >= 20000].copy()
dfa['subset'] = '20,000+ Owners'

dfb = data.copy()
dfb['subset'] = 'All Data'

ax = sns.boxplot(x='subset', y='rating', hue='type', data=pd.concat([dfa, dfb]))

ax.set(xlabel='', ylabel='Rating (%)')
plt.show()


"MEJORES 10 JUEGOS SEGUN SU SCORE Y A QUE GENERO PERTENECEN"
display_cols = ['name', 'developer', 'publisher', 'release_year', 'genres', 'average_playtime', 'owners', 'rating', 'price']
top_ten = df.sort_values(by='rating', ascending=False).head(10) #el 10 define la cantidad de juegos cambiar para mas
top_ten.plot(x='name', y='rating', kind='bar', figsize=(10, 6), legend=False)
plt.ylabel('Rating')
plt.title('Top Ten Games by Rating')
plt.xticks(rotation=45, ha='right')
plt.show()
# almacenando las columnas de categoría y género en una variable, ya que las accederemos con frecuencia
cat_gen_cols = df.columns[-13:-1]
ax = top_ten[cat_gen_cols].sum().plot.bar(figsize=(8,5))
ax.fill_between([-.5, 1.5], 10, alpha=.2)
ax.text(0.5, 9.1, 'Categories', fontsize=11, color='tab:blue', alpha=.10, horizontalalignment='center') #valores del plot cambiar si se cambian la cantidad de juegos
ax.set_ylim([0, 9.5])
ax.set_ylabel('Count')
ax.set_title('Frequency of categories and genres in top ten games')
plt.show()

"QUE CATEGORIA TIENE MAS JUEGOS"
ax = df[cat_gen_cols].sum().plot.bar()
ax.fill_between([-.5, 1.5], 7000, alpha=.2)   
ax.set_ylim([0, 1000])
ax.set_ylim([0, 1000]) #por defecto el dataframe de prueba tiene tamaño 1000
ax.set_title(f'Creation Popularity: Frequency of genres and categories across dataset of {df.shape[0]:,} games')
plt.show()


"""
df = data.copy()

years = []
lt_20k = []
gt_20k = []

for year in sorted(df['release_year'].unique()):
    if year < 2006:
        # very few releases in data prior to 2006, and we're still in 2019 (at time of writing)
        # so ignore these years
        continue

    # subset dataframe by year
    year_df = df[df.release_year == year]

    # calculate total with less than 20,000 owners, and total with 20,000 or more
    total_lt_20k = year_df[year_df.owners < 20000].shape[0]
    total_gt_20k = year_df[year_df.owners >= 20000].shape[0]

    years.append(year)
    lt_20k.append(total_lt_20k)
    gt_20k.append(total_gt_20k)

owners_df = pd.DataFrame(zip(years, lt_20k, gt_20k), 
                         columns=['year', 'Under 20,000 Owners', '20,000+ Owners'])

ax = owners_df.plot(x='year', y=[1, 2], kind='bar', stacked=True, color=['tab:red', 'gray'])

ax.set_xlabel('')
ax.set_ylabel('Number of Releases')
ax.set_title('Number of releases by year, broken down by number of owners')
sns.despine()
plt.show()
"""

