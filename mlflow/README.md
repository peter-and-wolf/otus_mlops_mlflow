# Запуск mlflow + postgres + minio

## Установка переменных окружения

Создайте файлик `.env`, в который добавьте примерно следующее:

```bash
PG_USER=mlflow
PG_PASSWORD=mlflow
PG_DATABASE=mlflow
MLFLOW_BUCKET_NAME=otus-mlflow-bucket
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=dZh5222@
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_AWS_ACCESS_KEY_ID=pmwGHCLBuSTM5eHzyWmB
MLFLOW_AWS_SECRET_ACCESS_KEY=26GOdhDyvBP78wEPuh4nP76KPUbeSNjZKWXGINtq
```

## Запуск сервисов

Выполните:

```bash
docker-compose up -d
```

Когда контейнеры стартанут, на: 

* `http://localhost:9001/` будет доступен GUI Minio (логин/пароль – переменные `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` в `.env`);
* на `http://localhost:5050/` будет доступен GUI MLFlow;