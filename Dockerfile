FROM python:3.11-slim

# Não gerar .pyc e sempre logar direto
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências mínimas de sistema
# libgl1 e libglib2.0-0 para opencv/pillow em alguns cenários
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia apenas o requirements primeiro (melhor cache)
COPY requirements.txt .

# Instala PyTorch CPU-only (leve) + torchvision CPU-only
# Ajuste de versão se quiser, mas essas funcionam bem com segmentation-models-pytorch 0.3.x
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Instala o restante das dependências
RUN pip install --no-cache-dir -r requirements.txt

# Agora copia o código do app
COPY . .

# Certifique-se que o modelo .pth esteja em /app/models/
# ex: models/deeplabv3plus_best_fold3_weights.pth

EXPOSE 8000

# Comando para subir a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]