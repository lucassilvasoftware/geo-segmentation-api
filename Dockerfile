FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  libgl1 \
  libglib2.0-0 \
  gdal-bin \
  libgdal-dev \
  && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY . .

# Certifique-se de que o modelo esteja em /app/models dentro do contexto de build
# (ex: models/deeplabv3plus_best_fold3_weights.pth)

EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
