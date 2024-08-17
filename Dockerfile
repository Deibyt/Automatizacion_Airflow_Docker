# Usa una imagen base de Python más reciente
FROM python:3.11-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos de requerimientos al contenedor
COPY requirements.txt /app/

# Actualiza pip y luego instala las dependencias
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia todo el contenido del proyecto al contenedor
COPY . /app/

# Expone el puerto 8000 para el contenedor
EXPOSE 8000

# Comando para correr el servidor Django
CMD ["python", "investment_project/manage.py", "runserver", "0.0.0.0:8000"]
