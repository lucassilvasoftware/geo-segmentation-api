import io
import base64
import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv  # <- NEW

from inference import (
    load_model,
    run_inference_raw,
    mask_to_color,
    compute_class_stats,
    DEVICE,
    NUM_CLASSES,
)

# Carrega variáveis do .env (para ambiente local)
load_dotenv()

app = FastAPI(
    title="LULC Segmentation API",
    description=(
        "API para segmentação semântica com DeepLabV3Plus. "
        "Endpoints: /segment (PNG) e /segment-full (PNG + estatísticas por classe)."
    ),
    version="1.3.0",
)

# ========================
# CORS
# ========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois restringe pro domínio do front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# API Key via variável de ambiente
# ========================
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_expected_api_key() -> str:
    api_key = os.getenv("API_KEY")
    if not api_key:
        # Em prod isso deve ser tratado como erro de config:
        raise HTTPException(
            status_code=500, detail="API_KEY não configurada no servidor."
        )
    return api_key


def validate_api_key(
    api_key: str = Depends(api_key_header),
    expected_api_key: str = Depends(get_expected_api_key),
):
    if api_key is None or api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ========================
# Modelo
# ========================
model = None


@app.on_event("startup")
def on_startup():
    global model
    try:
        model = load_model()
    except FileNotFoundError as e:
        # Se o modelo não for encontrado, deixa claro no log/erro:
        # Em produção, o container vai falhar — o que é bom para você perceber o problema.
        raise RuntimeError(str(e))


# ========================
# Rotas
# ========================


@app.get("/health", dependencies=[Depends(validate_api_key)])
def health():
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")
    return {"status": "ok"}


@app.get("/info", dependencies=[Depends(validate_api_key)])
def info():
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")
    return {
        "model_name": "DeepLabV3Plus",
        "num_classes": NUM_CLASSES,
        "device": DEVICE,
        "description": "Use /segment para PNG e /segment-full para PNG + estatísticas.",
    }


@app.post("/segment", dependencies=[Depends(validate_api_key)])
async def segment(file: UploadFile = File(...)):
    """
    Retorna apenas a máscara segmentada como image/png.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="O arquivo enviado não é uma imagem."
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Não foi possível abrir a imagem enviada."
        )

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    mask = run_inference_raw(model, img)
    mask_img = mask_to_color(mask)

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/segment-full", dependencies=[Depends(validate_api_key)])
async def segment_full(file: UploadFile = File(...)):
    """
    Retorna:
      - mask_png_base64: máscara colorida em PNG (base64)
      - stats: pixels e percentuais por classe
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="O arquivo enviado não é uma imagem."
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Não foi possível abrir a imagem enviada."
        )

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    mask = run_inference_raw(model, img)
    stats = compute_class_stats(mask)

    # máscara colorida → base64
    mask_img = mask_to_color(mask)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_bytes = buf.getvalue()
    mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")

    return {
        "mask_png_base64": mask_b64,
        "stats": stats,
    }
