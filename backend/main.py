"""
Backend API - LnT Camp 2026 Final Project
Prediksi profitabilitas produk (Regression & Classification)
menggunakan model Random Forest yang sudah dilatih di notebook.
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Load model & metadata (dilakukan SEKALI saat server start, bukan per-request)
# ---------------------------------------------------------------------------

# Folder model/ ada satu level di atas folder backend/ (struktur monorepo)
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

model_reg = joblib.load(MODEL_DIR / "model_reg.pkl")
model_clf = joblib.load(MODEL_DIR / "model_clf.pkl")
FEATURE_COLUMNS: list[str] = joblib.load(MODEL_DIR / "feature_columns.pkl")

# Kategori yang valid (sesuai data training).
# "Furniture" dan "Accessories" adalah baseline dari One-Hot Encoding
# (drop_first=True), sehingga tidak muncul sebagai kolom terpisah di
# FEATURE_COLUMNS, tapi tetap merupakan input yang sah dari user.
VALID_CATEGORIES = ["Furniture", "Office Supplies", "Technology"]
VALID_SUB_CATEGORIES = [
    "Accessories", "Appliances", "Art", "Binders", "Bookcases", "Chairs",
    "Copiers", "Envelopes", "Fasteners", "Furnishings", "Labels", "Machines",
    "Paper", "Phones", "Storage", "Supplies", "Tables",
]

app = FastAPI(
    title="LnT Camp 2026 - Product Profitability API",
    description="API untuk memprediksi rata-rata profit (Regression) dan "
                "profitabilitas (Classification) suatu produk berdasarkan "
                "pola order historisnya.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# 2. Skema input (Pydantic) - ini yang dilihat/diisi oleh frontend
# ---------------------------------------------------------------------------

class ProductFeatures(BaseModel):
    total_orders: int = Field(..., ge=0, description="Jumlah order unik untuk produk ini")
    avg_discount: float = Field(..., ge=0, le=1, description="Rata-rata diskon (0.0 - 1.0)")
    avg_sales: float = Field(..., ge=0, description="Rata-rata nilai penjualan per order")
    total_quantity_sold: int = Field(..., ge=0, description="Total kuantitas terjual")
    category: Literal["Furniture", "Office Supplies", "Technology"]
    sub_category: str = Field(..., description=f"Salah satu dari: {VALID_SUB_CATEGORIES}")

    class Config:
        json_schema_extra = {
            "example": {
                "total_orders": 5,
                "avg_discount": 0.15,
                "avg_sales": 250.0,
                "total_quantity_sold": 20,
                "category": "Technology",
                "sub_category": "Phones",
            }
        }


class RegressionResponse(BaseModel):
    predicted_avg_profit: float


class ClassificationResponse(BaseModel):
    predicted_label: Literal["Profitable", "Not Profitable"]
    predicted_class: int
    probability_profitable: float


# ---------------------------------------------------------------------------
# 3. Fungsi konversi input -> format one-hot encoding (SAMA PERSIS seperti
#    saat training di notebook)
# ---------------------------------------------------------------------------

def build_feature_vector(payload: ProductFeatures) -> pd.DataFrame:
    """
    Mengubah input mentah dari user (category, sub_category dalam bentuk
    teks biasa) menjadi satu baris DataFrame dengan kolom yang PERSIS SAMA
    (nama & urutan) seperti FEATURE_COLUMNS hasil training.
    """
    if payload.sub_category not in VALID_SUB_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"sub_category '{payload.sub_category}' tidak dikenali. "
                   f"Pilihan valid: {VALID_SUB_CATEGORIES}",
        )

    # Mulai dengan semua kolom one-hot bernilai 0
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Isi fitur numerik
    row["total_orders"] = payload.total_orders
    row["avg_discount"] = payload.avg_discount
    row["avg_sales"] = payload.avg_sales
    row["total_quantity_sold"] = payload.total_quantity_sold

    # Isi fitur kategorikal (one-hot).
    # Kalau kategori/sub_category adalah baseline (Furniture / Accessories),
    # semua kolom category_*/sub_category_* tetap 0 - ini sudah benar
    # sesuai skema drop_first=True.
    category_col = f"category_{payload.category}"
    if category_col in row:
        row[category_col] = 1

    sub_category_col = f"sub_category_{payload.sub_category}"
    if sub_category_col in row:
        row[sub_category_col] = 1

    # Susun jadi DataFrame 1 baris, dengan URUTAN KOLOM sama persis
    # seperti FEATURE_COLUMNS (wajib, karena model belajar berdasarkan
    # urutan kolom saat training)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# 4. Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Cek apakah server dan model berhasil dimuat dengan baik."""
    return {
        "status": "ok",
        "model_regression_loaded": model_reg is not None,
        "model_classification_loaded": model_clf is not None,
        "n_features": len(FEATURE_COLUMNS),
    }


@app.get("/categories")
def get_valid_categories():
    """Daftar category & sub_category yang valid (dipakai frontend untuk dropdown)."""
    return {
        "category": VALID_CATEGORIES,
        "sub_category": VALID_SUB_CATEGORIES,
    }


@app.post("/predict/regression", response_model=RegressionResponse)
def predict_regression(payload: ProductFeatures):
    """
    Memprediksi rata-rata profit (avg_profit) suatu produk berdasarkan
    pola order historisnya.
    """
    x = build_feature_vector(payload)
    prediction = model_reg.predict(x)[0]
    return RegressionResponse(predicted_avg_profit=round(float(prediction), 2))


@app.post("/predict/classification", response_model=ClassificationResponse)
def predict_classification(payload: ProductFeatures):
    """
    Mengklasifikasikan suatu produk sebagai Profitable atau Not Profitable
    berdasarkan pola order historisnya.
    """
    x = build_feature_vector(payload)
    predicted_class = int(model_clf.predict(x)[0])
    probability = float(model_clf.predict_proba(x)[0][1])  # probabilitas kelas "Profitable"

    label = "Profitable" if predicted_class == 1 else "Not Profitable"

    return ClassificationResponse(
        predicted_label=label,
        predicted_class=predicted_class,
        probability_profitable=round(probability, 4),
    )


# ---------------------------------------------------------------------------
# Menjalankan server secara langsung: python main.py
# (alternatif: uvicorn main:app --reload)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
