from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.utils.timezone import convert_to_utc
from datetime import datetime, timedelta

# Operador para instalar pendulum
install_pendulum = BashOperator(
    task_id='install_pendulum',
    bash_command='pip install pendulum',
)

import pendulum

# Definir argumentos por defecto para el DAG
local_tz = pendulum.timezone("America/Bogota")
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 17),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Crear el DAG
with DAG(
    dag_id='docker_run_and_cleanup',
    default_args=default_args,
    schedule_interval='*/5 * * * *',
    catchup=False,
    tags=['docker'],
) as dag:

    # Operador para iniciar el contenedor
    start_container = BashOperator(
        task_id='start_docker_container',
        bash_command='docker run -d --name my_django_container -p 8000:8000 django-vip-url python investment_project/manage.py runserver 0.0.0.0:8000',
    )

    # Operador dummy para esperar 5 minutos
    wait_2_5_minutes = BashOperator(
        task_id='wait_5_minutes',
        bash_command='sleep 150',  # 150 segundos = 2.5 minutos
    )

    # Operador para detener el contenedor
    stop_container = BashOperator(
        task_id='stop_docker_container',
        bash_command='docker stop my_django_container',
    )

    # Operador para eliminar el contenedor
    remove_container = BashOperator(
        task_id='remove_docker_container',
        bash_command='docker rm my_django_container',
    )

    # Definir el flujo de tareas
    install_pendulum >> start_container >> wait_2_5_minutes >> stop_container >> remove_container


#################################### DAG ultimo dia del mes #################################### 
"""
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pendulum

# Definir zona horaria
local_tz = pendulum.timezone("America/Bogota")

# Definir argumentos por defecto para el DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 17, tzinfo=local_tz),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Crear el DAG
with DAG(
    dag_id='docker_run_and_cleanup',
    default_args=default_args,
    schedule_interval='0 0 L * *',  # Ejecutar a las 00:00 horas el último día de cada mes
    catchup=False,
    tags=['docker'],
) as dag:

    # Operador para instalar pendulum
    install_pendulum = BashOperator(
        task_id='install_pendulum',
        bash_command='pip install pendulum',
    )

    # Operador para iniciar el contenedor
    start_container = BashOperator(
        task_id='start_docker_container',
        bash_command='docker run -d --name my_django_container -p 8000:8000 django-vip-url python investment_project/manage.py runserver 0.0.0.0:8000',
    )

    # Operador para esperar hasta el final del día
    wait_until_end_of_day = BashOperator(
        task_id='wait_until_end_of_day',
        bash_command='sleep $(( ( $(date -d tomorrow +%s) - $(date +%s) ) ))',  # Esperar hasta el final del día
    )

    # Operador para detener el contenedor
    stop_container = BashOperator(
        task_id='stop_docker_container',
        bash_command='docker stop my_django_container',
    )

    # Operador para eliminar el contenedor
    remove_container = BashOperator(
        task_id='remove_docker_container',
        bash_command='docker rm my_django_container',
    )

    # Definir el flujo de tareas
    install_pendulum >> start_container >> wait_until_end_of_day >> stop_container >> remove_container

"""