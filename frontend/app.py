"""
Frontend - LnT Camp 2026 Final Project
Simulasi prediksi profitabilitas produk (Regression & Classification).
Frontend ini memanggil Backend API (FastAPI) untuk mendapatkan hasil prediksi.
"""

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Product Profitability Predictor", page_icon="📦", layout="centered")

# Fallback daftar kategori, dipakai kalau backend sedang tidak bisa diakses
# saat halaman pertama kali dibuka (misal backend belum dinyalakan).
FALLBACK_CATEGORIES = ["Furniture", "Office Supplies", "Technology"]
FALLBACK_SUB_CATEGORIES = [
    "Accessories", "Appliances", "Art", "Binders", "Bookcases", "Chairs",
    "Copiers", "Envelopes", "Fasteners", "Furnishings", "Labels", "Machines",
    "Paper", "Phones", "Storage", "Supplies", "Tables",
]

# ---------------------------------------------------------------------------
# Sidebar: alamat Backend API
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")
backend_url = st.sidebar.text_input(
    "Backend API URL",
    value="http://localhost:8000",
    help="Alamat tempat backend FastAPI berjalan. Default: localhost (backend dijalankan lokal saat demo).",
).rstrip("/")

st.sidebar.markdown(
    """
    ---
    **Catatan:** Backend tidak wajib dideploy publik dan boleh dijalankan
    secara lokal saat demo. Jika backend dideploy (misal di Render/Railway),
    ubah URL di atas sesuai alamat backend tersebut.
    """
)


def check_backend_health():
    """Mengecek apakah backend bisa dihubungi dan model berhasil dimuat."""
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except requests.exceptions.RequestException:
        return False, None


def get_categories():
    """Mengambil daftar category & sub_category dari backend, dengan fallback."""
    try:
        response = requests.get(f"{backend_url}/categories", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data["category"], data["sub_category"]
    except requests.exceptions.RequestException:
        pass
    return FALLBACK_CATEGORIES, FALLBACK_SUB_CATEGORIES


# ---------------------------------------------------------------------------
# Header + status koneksi backend
# ---------------------------------------------------------------------------
st.title("📦 Product Profitability Predictor")
st.write(
    "Simulasi prediksi profitabilitas produk berdasarkan pola order "
    "historisnya, menggunakan model Machine Learning yang dilatih pada "
    "dataset Global Superstore."
)

is_healthy, health_data = check_backend_health()
if is_healthy:
    st.success(f"✅ Backend terhubung — {health_data['n_features']} fitur siap digunakan.")
else:
    st.error(
        "⚠️ Backend tidak dapat dihubungi. Pastikan backend FastAPI sudah "
        "dijalankan (`python main.py`) dan URL di sidebar sudah benar."
    )

categories, sub_categories = get_categories()

tab_regression, tab_classification = st.tabs(["📈 Prediksi Profit (Regression)", "🏷️ Klasifikasi Profitabilitas"])

# ---------------------------------------------------------------------------
# TAB 1: Regression
# ---------------------------------------------------------------------------
with tab_regression:
    st.subheader("Prediksi Rata-rata Profit Produk")
    st.caption("Model akan memprediksi perkiraan rata-rata profit (avg_profit) suatu produk.")

    with st.form("form_regression"):
        col1, col2 = st.columns(2)
        with col1:
            reg_total_orders = st.number_input(
                "Total Orders", min_value=0, value=5, step=1,
                help="Jumlah order unik untuk produk ini", key="reg_orders",
            )
            reg_avg_discount = st.slider(
                "Rata-rata Diskon", min_value=0.0, max_value=1.0, value=0.15, step=0.01,
                key="reg_discount",
            )
            reg_avg_sales = st.number_input(
                "Rata-rata Sales", min_value=0.0, value=250.0, step=10.0, key="reg_sales",
            )
        with col2:
            reg_total_quantity = st.number_input(
                "Total Kuantitas Terjual", min_value=0, value=20, step=1, key="reg_qty",
            )
            reg_category = st.selectbox("Category", categories, key="reg_category")
            reg_sub_category = st.selectbox("Sub-Category", sub_categories, key="reg_subcategory")

        submitted_reg = st.form_submit_button("🔮 Prediksi Profit", use_container_width=True)

    if submitted_reg:
        payload = {
            "total_orders": reg_total_orders,
            "avg_discount": reg_avg_discount,
            "avg_sales": reg_avg_sales,
            "total_quantity_sold": reg_total_quantity,
            "category": reg_category,
            "sub_category": reg_sub_category,
        }
        try:
            response = requests.post(f"{backend_url}/predict/regression", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.metric("Prediksi Rata-rata Profit", f"${result['predicted_avg_profit']:,.2f}")
            else:
                st.error(f"Backend mengembalikan error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Gagal menghubungi backend: {e}")

# ---------------------------------------------------------------------------
# TAB 2: Classification
# ---------------------------------------------------------------------------
with tab_classification:
    st.subheader("Klasifikasi Profitabilitas Produk")
    st.caption("Model akan mengklasifikasikan produk sebagai Profitable atau Not Profitable.")

    with st.form("form_classification"):
        col1, col2 = st.columns(2)
        with col1:
            clf_total_orders = st.number_input(
                "Total Orders", min_value=0, value=5, step=1, key="clf_orders",
            )
            clf_avg_discount = st.slider(
                "Rata-rata Diskon", min_value=0.0, max_value=1.0, value=0.15, step=0.01,
                key="clf_discount",
            )
            clf_avg_sales = st.number_input(
                "Rata-rata Sales", min_value=0.0, value=250.0, step=10.0, key="clf_sales",
            )
        with col2:
            clf_total_quantity = st.number_input(
                "Total Kuantitas Terjual", min_value=0, value=20, step=1, key="clf_qty",
            )
            clf_category = st.selectbox("Category", categories, key="clf_category")
            clf_sub_category = st.selectbox("Sub-Category", sub_categories, key="clf_subcategory")

        submitted_clf = st.form_submit_button("🔮 Klasifikasikan Produk", use_container_width=True)

    if submitted_clf:
        payload = {
            "total_orders": clf_total_orders,
            "avg_discount": clf_avg_discount,
            "avg_sales": clf_avg_sales,
            "total_quantity_sold": clf_total_quantity,
            "category": clf_category,
            "sub_category": clf_sub_category,
        }
        try:
            response = requests.post(f"{backend_url}/predict/classification", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                label = result["predicted_label"]
                prob = result["probability_profitable"]

                if label == "Profitable":
                    st.success(f"✅ Hasil: **{label}**")
                else:
                    st.warning(f"⚠️ Hasil: **{label}**")

                st.progress(prob)
                st.caption(f"Probabilitas Profitable: {prob:.1%}")
            else:
                st.error(f"Backend mengembalikan error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Gagal menghubungi backend: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "LnT Camp 2026 Final Project — Bridging the Gap: Empowering Future Talent "
    "through Machine Learning for Industry Innovation."
)
