# Fuentes
Steam y SteamSpy

# Instrucciones de Ejecución

## Códigos de Datos y Limpieza

Sigue estos pasos para ejecutar los códigos de datos y limpieza:

1. **Verifica la instalación de Python**: Asegúrate de tener instalado una *version de Python >=3.7 o <3.11*  en tu sistema. Puedes verificarlo ejecutando el siguiente comando en tu terminal:
    ```bash
    python --version
    ```

2. **Descarga los archivos**: Descarga el comprimido con los archivos `.py` a utilizar y guárdalo en una ubicación accesible en tu sistema.

3. **Preparación para la ejecución**:

    a. **Navega hasta el archivo de código**: Abre una terminal y navega hasta la ubicación del archivo de código de datos y limpieza.

    b. **Instala las dependencias**: Ejecuta el siguiente comando para instalar las dependencias necesarias:
        
        pip install -r requirements.txt
        
    c. **Ajusta la cantidad de juegos**: En el archivo `datos.py` en las `líneas 267 y 309`, ajusta el valor de cuántos juegos se obtienen (son 2 valores porque tomamos info de Steam y SteamSpy).

    d. **Descomenta la creación de carpetas**: En el archivo `datos.py` en las `líneas 66 y 67`, y en el archivo `limpieza.py` en la `línea 330`, descomenta la creación de carpetas para que funcione el programa. Después de ejecutarlo la primera vez, vuelve a comentar estas líneas.

5. **Ejecuta los códigos**: Ejecuta los siguientes comandos para ejecutar los códigos de datos, limpieza y para graficar:
    ```bash
    python datos.py
    python union.py
    python limpieza.py
    python graficos.ipynb
    ```

6. **Verifica los resultados**: Ahora, deberías tener un archivo `steam.csv` con la información filtrada. (este archivo se genera en una carpeta data fuera de la carpeta actual)

   ![Captura de pantalla 2024-05-25 150902](https://github.com/kt6delta/SteamExtract/assets/92498586/cd833748-5e63-4029-a878-0c04ff86229a)


> **Nota**: Se generan varios archivos CSV. Los que se encuentran en la carpeta `download` son los que se generan al descargar la información, y los datos dentro de la carpeta `exports` son los que "pueden llegar a ser útiles" pero que no son tan importantes.

Si tienes alguna otra pregunta, no dudes en preguntar.
