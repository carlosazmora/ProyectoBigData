# Celda 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import logging
import psycopg2



# Celda 2
# Ejecutar un comando de docker compose
result = subprocess.run(
    ["docker", "compose", "up", "-d"],
    capture_output=True,
    text=True
)

# Ver salida
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Código de salida:", result.returncode)


# Celda 3
# Extraer información de archivo .csv
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


# Celda 4
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


# Grafica 1
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

