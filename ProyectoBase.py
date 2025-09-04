'''
--- Proyecto Final - Coterminal ----

- Carlos Andrés Zuluaga Mora
- Ricardo Andrés Cortés Coronell
- Andrés Felipe Sánchez Rincón

'''

'''
--- Requisitos de ejecución ---

- Cree un entorno Python (Versión 3.11 como mínimo) en su máquina 
mediante el comando "conda create -n {nombre del entorno}".

- Instale todas las librerías faltantes con el comando 
"conda install {nombre de la librería}". De presentarse el caso en el que dicho comando no funcione adecuadamente, 
recurra a opciones como "conda-forge" o "pip" para continuar con el proceso de instalación

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import logging
import psycopg2



'''
--- Montaje del contenedor donde se alojará la base de datos ----
'''
print("\nMontaje del contenedor donde se alojará la base de datos")

result = subprocess.run(
    ["docker", "compose", "up", "-d"],
    capture_output=True,
    text=True
)

# Ver salida
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Código de salida:", result.returncode)

'''
--- Preparación y limpieza de datos ---
'''
print("\nPreparación y limpieza de datos")

df_covid19 = pd.read_csv("estimated-cumulative-excess-deaths-per-100000-people-during-covid-19.csv")

# Limpieza de datos

# Eliminación de columnas no empleadas 
df_covid19.drop(columns=['Cumulative excess deaths per 100,000 people (95% CI, lower bound)', 
        'Cumulative excess deaths per 100,000 people (95% CI, upper bound)'], inplace=True)

# Cambio de nombres de columnas
df_covid19.rename(columns={
    'Entity': 'País',
    'Day': 'Fecha',
    'Cumulative excess deaths per 100,000 people (central estimate)': 'Exceso_Muertes',
    'Total confirmed deaths due to COVID-19 per 100,000 people': 'Muertes_Confirmadas_100k'
}, inplace=True)

# Convertir la columna 'Fecha' a datetime
df_covid19['Fecha'] = pd.to_datetime(df_covid19['Fecha'])

# La columna de muertes confirmadas tiene nulos al inicio. Asumimos que son 0.
df_covid19['Muertes_Confirmadas_100k'].fillna(0, inplace=True)

# Para el análisis, las filas sin una estimación central de exceso de muertes no son útiles.
df_covid19.dropna(subset=['País', 'Fecha', 'Exceso_Muertes'], inplace=True)


df_covid19.info()
df_covid19.head()


'''
--- Creación de DB y cargue de información ----
'''
print("\nCreación de DB y cargue de información")

import psycopg2

# === 1. Verificar/crear la base de datos ===
try:
    # Conexión a la base por defecto
    default_conn = psycopg2.connect(
        dbname="postgres",   # Base inicial por defecto
        user="psqluser",
        password="psqlpass",
        host="localhost",
        port="5433"
    )
    default_conn.autocommit = True  # Necesario para CREATE DATABASE

    with default_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", ('covid19-project',))
        exists = cur.fetchone()
        if not exists:
            cur.execute('CREATE DATABASE "covid19-project";')
            print("Base de datos 'covid19-project' creada exitosamente.")
        else:
            print("La base de datos 'covid19-project' ya existe.")

except Exception as e:
    print(f"Error al crear la base de datos: {e}")
    raise

finally:
    if 'default_conn' in locals():
        default_conn.close()

# === 2. Conexión a la base 'covid19-project' ===
db_params = {
    'dbname': 'covid19-project',
    'user': 'psqluser',
    'password': 'psqlpass',
    'host': 'localhost',
    'port': '5433'
}

try:
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # === 3. Crear la tabla ===
    cursor.execute("DROP TABLE IF EXISTS muertes_covid19;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS muertes_covid19 (
            id SERIAL PRIMARY KEY,
            pais VARCHAR(100),
            fecha DATE,
            exceso_muertes INT,
            muertes_confirmadas INT
        )
    """)
    conn.commit()
    print("Tabla creada exitosamente.")

    # === 4. Preparar e insertar los datos desde tu DataFrame ===
    data = list(df_covid19[['País', 'Fecha', 'Exceso_Muertes', 'Muertes_Confirmadas_100k']].itertuples(index=False, name=None))

    cursor.executemany("""
        INSERT INTO muertes_covid19 (pais, fecha, exceso_muertes, muertes_confirmadas)
        VALUES (%s, %s, %s, %s)
    """, data)
    conn.commit()
    print(f"{len(data)} filas insertadas exitosamente en la tabla.")

except Exception as e:
    print(f"Error al insertar datos: {e}")
    raise

finally:
    if 'conn' in locals():
        conn.close()


'''
--- Sección de Análisis ----
'''
print("\nSección de Análisis")

print("\nAnálisis 1: ¿Cómo evolucionaron las muertes confirmadas por COVID-19 a lo largo del tiempo en los 10 países más afectados?")
'''
¿Cómo evolucionaron las muertes confirmadas por COVID-19 a lo largo del tiempo en los 10 países más afectados?
Los resultados muestra la evolución de muertes confirmadas por COVID-19 de los 10 paises mas afectados por la pandemia. 
El análisis revela que países de Europa del Este y los Balcanes dominan la lista, lo que indica que estas regiones fueron de las más afectadas por la crisis sanitaria.
'''

from prefect import flow, task, get_run_logger

db_params = {
    'dbname': 'covid19-project',
    'user': 'psqluser',
    'password': 'psqlpass',
    'host': 'localhost',
    'port': '5433'
}

@task
def extract_from_db(db_params, table_name="muertes_covid19"):
    logger = get_run_logger()
    try:
        conn = psycopg2.connect(**db_params)
        query = f"""
            SELECT "pais", "fecha", "muertes_confirmadas"
            FROM {table_name}
            WHERE "muertes_confirmadas" IS NOT NULL;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error extrayendo datos de PostgreSQL: {e}")
        raise

@task
def plot_excess_deaths(df):
    # Asegurarse de que la columna fecha es datetime
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Graficar todos los países (solo los principales para evitar saturación visual)
    top_countries = df.groupby('pais')['muertes_confirmadas'].max().sort_values(ascending=False).head(10).index
    plt.figure(figsize=(14, 7))
    for country in top_countries:
        country_data = df[df['pais'] == country]
        plt.plot(country_data['fecha'], country_data['muertes_confirmadas'], label=country)

    plt.title("Evolución de las muertes confirmadas por COVID-19 (Top 10 países)")
    plt.xlabel("Fecha")
    plt.ylabel("Muertes confirmadas acumuladas por 100k habitantes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

@flow
def analyze_excess_deaths_flow():
    df = extract_from_db(db_params)
    plot_excess_deaths(df)

# Ejecutar el flow
analyze_excess_deaths_flow()

print("\nAnálisis 2: ¿Cuál fue el comportamiento de Colombia a nivel de muertes confirmadas y muertes en exceso durante el periodo evaluado?")
'''
¿Cuál fue el comportamiento de Colombia a nivel de muertes confirmadas y muertes en exceso durante el periodo evaluado?
El comportamiento del exceso de muertes en Colombia siguió muy de cerca a las muertes confirmadas,
lo que confirma que las cifras oficiales son un buen indicador de las tendencias de la pandemia, a pesar de un evidente subregistro.
'''

from prefect import flow, task, get_run_logger

db_params = {
    'dbname': 'covid19-project',
    'user': 'psqluser',
    'password': 'psqlpass',
    'host': 'localhost',
    'port': '5433'
}

@task
def extract_from_db(db_params, table_name="muertes_covid19"):
    logger = get_run_logger()
    try:
        conn = psycopg2.connect(**db_params)
        query = f"""
            SELECT "pais", "fecha", "muertes_confirmadas", "exceso_muertes"
            FROM {table_name}
            WHERE "muertes_confirmadas" IS NOT NULL AND "pais" = 'Colombia';
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error extrayendo datos de PostgreSQL: {e}")
        raise

@task
def plot_COLOMBIA_excessANDconfirmed_deaths(df):
    # Asegurarse de que la columna fecha es datetime
    df['fecha'] = pd.to_datetime(df['fecha'])

    plt.figure(figsize=(14, 7))
    plt.plot(df['fecha'], df['muertes_confirmadas'], label='Muertes Confirmadas')
    plt.plot(df['fecha'], df['exceso_muertes'], label='Exceso de Muertes')
    plt.title('Evolución de Muertes Confirmadas y Exceso de Muertes por COVID-19 en Colombia')
    plt.xlabel('Fecha')
    plt.ylabel('Muertes por 100,000 habitantes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

@flow
def analyze_excess_deaths_flow():
    df = extract_from_db(db_params)
    plot_COLOMBIA_excessANDconfirmed_deaths(df)

# Ejecutar el flow
analyze_excess_deaths_flow()

print("\nAnálisis 3: Distribución de Países por Muertes Confirmadas de COVID-19")
'''
Distribución de Países por Muertes Confirmadas de COVID-19
La media de muertes confirmadas es de aproximadamente **124.81**. El análisis de los datos revela que la mayoría de los países se encuentran por debajo de
este valor, mientras que un numero reducido de países experimentó una mortalidad bastante superior. Esto refleja un patrón de dispersión de la pandemia,
donde algunos países sufrieron un impacto mucho mayor en términos de muertes confirmadas.
'''

from prefect import flow, task, get_run_logger

db_params = {
    'dbname': 'covid19-project',
    'user': 'psqluser',
    'password': 'psqlpass',
    'host': 'localhost',
    'port': '5433'
}

@task
def extract_from_db(db_params, table_name="muertes_covid19"):
    logger = get_run_logger()
    try:
        conn = psycopg2.connect(**db_params)
        query = f"""
            SELECT
                "pais",
                MAX("muertes_confirmadas") AS "muertes_confirmadas"
            FROM {table_name}
            GROUP BY "pais"
            ORDER BY "muertes_confirmadas" DESC;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error extrayendo datos de PostgreSQL: {e}")
        raise

@task
def plot_deaths_histogram(df):
    logger = get_run_logger()
    if df is None or df.empty:
        logger.warning("No hay datos para graficar.")
        return
        
    df.dropna(subset=['muertes_confirmadas'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['muertes_confirmadas'], inplace=True)
    
    bins = np.arange(0, df['muertes_confirmadas'].max() + 20, 20)

    plt.figure(figsize=(12, 8))
    plt.hist(df['muertes_confirmadas'], bins=bins, edgecolor='black', alpha=0.7)
    
    media = df['muertes_confirmadas'].mean()
    plt.axvline(media, color='red', linestyle='dashed', linewidth=2, label=f'Media: {media:.2f}')
    
    plt.title('Distribución de Países por Muertes Confirmadas de COVID-19', fontsize=16)
    plt.xlabel('Muertes Confirmadas por 100,000 personas', fontsize=12)
    plt.ylabel('Cantidad de Países', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    logger.info("Histograma generado correctamente.")

@task
def analyze_top_countries(df, top_n=10):
    logger = get_run_logger()
    if df is None or df.empty:
        logger.warning("No hay datos para analizar.")
        return

    top_countries = df.sort_values(by='muertes_confirmadas', ascending=False).head(top_n)
    
    logger.info("\nAnálisis de Países con Mayor Mortalidad Confirmada")
    logger.info(f"Los {top_n} países con el mayor número de muertes confirmadas (por 100,000 habitantes) son:")

    print(top_countries.to_string(index=False))

@flow
def analyze_deaths_distribution_flow():
    df = extract_from_db(db_params)
    plot_deaths_histogram(df)
    analyze_top_countries(df)

# Ejecutar el flow
analyze_deaths_distribution_flow()

print("\nAnálisis 4: ¿Existe correlación entre el exceso de muertes y las muertes confirmadas por COVID-19?")
'''
¿Existe correlación entre el exceso de muertes y las muertes confirmadas por COVID-19?

1. Correlación Global

La correlación de Pearson global tiene un valor de 0.65, esto indica una relación positiva y moderadamente "fuerte". 
Los países con más muertes confirmadas también contaron con un mayor exceso de muertes, dando validez a que las cifras oficiales siguen la tendencia de la mortalidad real.

2. Correlación por País

El análisis detallado por país muestra que la mayoría de los países tienen una correlación bastante alta, reforzando la conclusión global. 
Sin embargo, algunos paises presentan correlaciones muy bajas e incluso negativas, como Groenlandia (-0.88) y Antigua y Barbuda (-0.76). Esto puede sugerir problemas en el registro de datos o un subregistro significativo.

3. Conclusión

Los resultados demuestran que las muertes confirmadas fueron un indicador fundamental en el impacto de la pandemia, ya que su tendencia coincide con el exceso de muertes. 
Sin embargo, la existencia de correlaciones bastante bajas y negativas en algunos países remarca la importancia de analizar la calidad de los datos y no depender unicamente por las cifras oficiales.
'''

from prefect import flow, task, get_run_logger

db_params = {
    'dbname': 'covid19-project',
    'user': 'psqluser',
    'password': 'psqlpass',
    'host': 'localhost',
    'port': '5433'
}

@task
def extract_from_db(db_params, table_name="muertes_covid19"):
    logger = get_run_logger()
    try:
        conn = psycopg2.connect(**db_params)
        query = f"""
            SELECT
                "pais",
                "fecha",
                "muertes_confirmadas",
                "exceso_muertes"
            FROM {table_name}
            WHERE "muertes_confirmadas" IS NOT NULL AND "exceso_muertes" IS NOT NULL;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error extrayendo datos de PostgreSQL: {e}")
        raise

@task
def calculate_and_plot_global_correlation(df):
    """
    Calcula la correlación global y genera un mapa de calor.
    """
    logger = get_run_logger()
    if df is None or df.empty:
        logger.warning("No hay datos para calcular la correlación global.")
        return None

    # Agrupar por país y tomar el último valor acumulado
    df_final = df.groupby('pais').max(numeric_only=True).reset_index()

    # Calcular la matriz de correlación
    corr_matrix = df_final[['muertes_confirmadas', 'exceso_muertes']].corr()
    correlation_global = corr_matrix.loc['muertes_confirmadas', 'exceso_muertes']
    
    logger.info(f"\n--- Correlación Global ---")
    logger.info(f"El coeficiente de correlación de Pearson (Global) es: {correlation_global:.4f}")
    
    # Generar el mapa de calor
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
    plt.title('Mapa de Calor de la Correlación Global', fontsize=16)
    plt.show()

    return correlation_global

@task
def calculate_correlation_by_country(df):
    logger = get_run_logger()
    if df is None or df.empty:
        logger.warning("No hay datos para calcular la correlación por país.")
        return None

    correlations = df.groupby('pais').apply(
        lambda x: x['muertes_confirmadas'].corr(x['exceso_muertes'])
    ).reset_index(name='correlacion')
    
    correlations.dropna(inplace=True)
    correlations.sort_values(by='correlacion', inplace=True)

    logger.info("\n--- Correlación por País (ejemplo de 5 valores más bajos y más altos) ---")
    logger.info("Países con la correlación más baja:")
    logger.info(correlations.head(5).to_string(index=False))
    
    logger.info("\nPaíses con la correlación más alta:")
    logger.info(correlations.tail(5).to_string(index=False))

    return correlations

@flow(name="Análisis de Correlación de Muertes")
def analyze_correlation_flow():
    df = extract_from_db(db_params)
    if df is not None:
        calculate_and_plot_global_correlation(df)
        calculate_correlation_by_country(df)

# Ejecutar el flow
analyze_correlation_flow()

print("\nAnálisis 5: ¿Cuál es el tiempo de retraso entre los picos de muertes confirmadas y  exceso de muertes en los países más afectados?")
'''
¿Cuál es el tiempo de retraso entre los picos de muertes confirmadas y  exceso de muertes en los países más afectados?

Los datos indican que los países de la región de Europa del Este tienen el mayor exceso de mortalidad, con Lituania, Bulgaria y Rusia ocupando las tres posiciones mas altas.

Alto Subregistro: Países como Mónaco y Rusia muestran porcentajes de subregistro altos superando el 70%. 
Esto representa que por cada muerte confirmada, hubo muchas más que no se le atribuyeron de forma oficial a la pandemia.

Caso Extremo: El caso de Niue, cuenta con un porcentaje del 100% de subregistro. 
Esto podría deberse a una falta completa de reporte de muertes por COVID-19, una población muy pequeña, o un problema con los datos.

Conclusión
Los resultados revelan que en estos paises el sistema de reporte de muertes confirmadas fue deficiente, 
se podría deber a una falta de pruebas masivas, clasificación errónea de las muertes, o porque una saturación extrema de los sistemas de salud debido a la crisis sanitaria.
'''

from prefect import flow, task, get_run_logger

db_params = {
    'dbname': 'covid19-project',
    'user': 'psqluser',
    'password': 'psqlpass',
    'host': 'localhost',
    'port': '5433'
}

@task
def extract_from_db(db_params, table_name="muertes_covid19"):
    logger = get_run_logger()
    try:
        conn = psycopg2.connect(**db_params)
        query = f"""
            SELECT
                "pais",
                MAX("muertes_confirmadas") AS "muertes_confirmadas",
                MAX("exceso_muertes") AS "exceso_muertes"
            FROM {table_name}
            WHERE "muertes_confirmadas" IS NOT NULL AND "exceso_muertes" IS NOT NULL
            GROUP BY "pais";
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error extrayendo datos de PostgreSQL: {e}")
        raise

@task
def analyze_excess_mortality(df, top_n=10):
    logger = get_run_logger()
    if df is None or df.empty:
        logger.warning("No hay datos para el análisis.")
        return None

    df_sorted = df.sort_values(by="exceso_muertes", ascending=False)

    df_sorted['diferencia'] = df_sorted['exceso_muertes'] - df_sorted['muertes_confirmadas']
    df_sorted['subregistro_pct'] = (df_sorted['diferencia'] / df_sorted['exceso_muertes']) * 100
    df_sorted.loc[df_sorted['subregistro_pct'] < 0, 'subregistro_pct'] = 0

    top_excess = df_sorted.head(top_n)
    
    logger.info(f"\n--- Los {top_n} países con mayor exceso de mortalidad (por 100,000 habitantes) ---")
    logger.info(top_excess[['pais', 'exceso_muertes', 'muertes_confirmadas', 'diferencia', 'subregistro_pct']].to_string(index=False))

    max_subregistro_pais = df_sorted.sort_values(by='subregistro_pct', ascending=False).iloc[0]
    logger.info(f"\n--- País con el mayor subregistro de muertes ---")
    logger.info(f"El país con el mayor subregistro es {max_subregistro_pais['pais']} con un {max_subregistro_pais['subregistro_pct']:.2f}% de subregistro.")
    
    return top_excess

@task
def plot_mortality_comparison(df_top_excess):
    if df_top_excess is None or df_top_excess.empty:
        return

    df_top_excess.set_index('pais', inplace=True)
    df_top_excess[['exceso_muertes', 'muertes_confirmadas']].plot(kind='bar', figsize=(14, 8), width=0.8)
    
    plt.title('Comparación de Exceso de Muertes vs. Muertes Confirmadas en los Países más Afectados', fontsize=16)
    plt.xlabel('País')
    plt.ylabel('Muertes por 100,000 habitantes')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Exceso de Muertes', 'Muertes Confirmadas'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

@flow
def excess_mortality_analysis_flow():
    df = extract_from_db(db_params)
    if df is not None:
        top_excess_df = analyze_excess_mortality(df)
        plot_mortality_comparison(top_excess_df)

# Ejecutar el flow
excess_mortality_analysis_flow()