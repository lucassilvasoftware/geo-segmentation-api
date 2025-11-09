import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import segmentation_models_pytorch as smp

# ========================
# Configurações
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 8

# Caminho dos pesos (ajuste se necessário)
MODEL_PATH = Path("models/deeplabv3plus_best_fold3_weights.pth")

# ========================
# Cores das classes (dataset oficial)
# ========================
CLASS_COLORS = [
    (255, 0, 0),  # 0 - Urbano
    (38, 115, 0),  # 1 - Vegetação Densa
    (0, 0, 0),  # 2 - Sombra
    (133, 199, 126),  # 3 - Vegetação Esparsa
    (255, 255, 0),  # 4 - Agricultura
    (128, 128, 128),  # 5 - Rocha
    (139, 69, 19),  # 6 - Solo Exposto
    (84, 117, 168),  # 7 - Água
]

# ========================
# Modelo
# ========================


def get_model() -> torch.nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,
        classes=NUM_CLASSES,
    )
    return model


def load_model() -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de pesos não encontrado em {MODEL_PATH}. "
            "Coloque o .pth treinado dentro da pasta 'models/'."
        )

    model = get_model().to(DEVICE)
    state = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ========================
# Pré-processamento
# ========================

_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def preprocess(img: Image.Image) -> torch.Tensor:
    return _transform(img).unsqueeze(0)


# ========================
# Inferência bruta
# ========================


def run_inference_raw(model: torch.nn.Module, img: Image.Image) -> np.ndarray:
    """
    Retorna a máscara (H, W) com valores [0..NUM_CLASSES-1].
    """
    x = preprocess(img).to(DEVICE)

    with torch.no_grad():
        output = model(x)  # (1, C, H, W)
        pred = torch.argmax(output, dim=1)  # (1, H, W)

    return pred.squeeze(0).cpu().numpy()


# ========================
# Máscara colorida
# ========================


def mask_to_color(mask: np.ndarray) -> Image.Image:
    """
    Converte uma máscara (H, W) em imagem RGB usando CLASS_COLORS.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS):
        color_mask[mask == cls_idx] = color
    return Image.fromarray(color_mask)


# ========================
# Estatísticas por classe
# ========================


def compute_class_stats(mask: np.ndarray):
    """
    Retorna lista:
      [{class_id, class_name, pixels, percent}, ...]
    """
    CLASS_NAMES = [
        "Urbano",
        "Vegetação Densa",
        "Sombra",
        "Vegetação Esparsa",
        "Agricultura",
        "Rocha",
        "Solo Exposto",
        "Água",
    ]

    total_pixels = int(mask.size)
    counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)

    stats = []
    for class_id in range(NUM_CLASSES):
        pixels = int(counts[class_id])
        percent = (pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
        stats.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "pixels": pixels,
                "percent": round(percent, 4),
            }
        )

    return stats
