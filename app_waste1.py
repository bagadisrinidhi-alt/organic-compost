import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear

# ------------------------------------------------------------
# 1️⃣  Load datasets directly (no upload)
# ------------------------------------------------------------
CROP_DATA_PATH = "updated_crop_dataset.csv"       # rename to your correct file
WASTE_DATA_PATH = "complete_data.csv"  # converted from Excel to CSV

# Load datasets
crops = pd.read_csv(CROP_DATA_PATH)
wastes = pd.read_csv(WASTE_DATA_PATH)

# Clean column names
crops.columns = [c.strip() for c in crops.columns]
wastes.columns = [c.strip() for c in wastes.columns]

# ------------------------------------------------------------
# 2️⃣  Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Smart Waste Recommendation", layout="wide")
st.title("🌱 Smart Organic Waste Recommendation System")

# User input: crop name
crop_name = st.text_input("Enter Crop Name (e.g., Banana, Maize, Wheat):")

if crop_name:
    # Find the crop in dataset
    row = crops[crops["Crop"].str.lower() == crop_name.lower()]

    if row.empty:
        st.error(f"❌ Crop '{crop_name}' not found in dataset.")
    else:
        crop_row = row.iloc[0]

        # ------------------------------------------------------------
        # 3️⃣  Extract crop nutrient requirements
        # ------------------------------------------------------------
        req_N = float(crop_row.get("Nitrogen (kg/acre)", 0.0))
        req_C = float(crop_row.get("Carbon (kg/acre)", 0.0))
        req_Minerals = float(crop_row.get("Minerals (kg/acre)", 0.0))
        req_Moist = float(crop_row.get("Moisture (liters/acre)", 0.0))

        st.subheader("🌾 Crop Nutrient Requirements")
        req_df = pd.DataFrame({
            "Nutrient": ["Nitrogen (kg/acre)", "Carbon (kg/acre)", "Minerals (kg/acre)", "Moisture (liters/acre)"],
            "Required": [req_N, req_C, req_Minerals, req_Moist]
        })
        st.dataframe(req_df, hide_index=True)

        # ------------------------------------------------------------
        # 4️⃣  Prepare Waste Dataset (Fix Series OR issue)
        # ------------------------------------------------------------
        wastes["waste_name"] = wastes.iloc[:, 0].astype(str)

        def get_column(df, possible_names):
            """Helper to safely get a numeric column from possible variants."""
            for name in possible_names:
                if name in df.columns:
                    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
            return pd.Series([0.0] * len(df))

        wastes["nitrogen_per_kg"] = get_column(wastes, ["NITROGEN", "Nitrogen"])
        wastes["carbon_per_kg"] = get_column(wastes, ["CARBON", "Carbon"])
        wastes["minerals_per_kg"] = get_column(wastes, ["MINERALS", "Minerals"])
        wastes["moisture_per_kg"] = get_column(wastes, ["MOISTURE", "Moisture"])

        # ------------------------------------------------------------
        # 5️⃣  Mixture Recommendation (Optimal Linear Combination)
        # ------------------------------------------------------------
        A = np.vstack([
            wastes["nitrogen_per_kg"].values,
            wastes["carbon_per_kg"].values,
        ])
        b = np.array([req_N, req_C], dtype=float)

        res = lsq_linear(A, b, bounds=(0, np.inf))
        x = res.x
        supplied = A.dot(x)

        mix_df = pd.DataFrame({
            "Waste": wastes["waste_name"],
            "Amount (kg)": x
        })
        mix_df = mix_df[mix_df["Amount (kg)"] > 1e-6].sort_values("Amount (kg)", ascending=False).reset_index(drop=True)

        st.subheader("♻️ Recommended Waste Mixture (Best Combination)")
        st.dataframe(mix_df.head(20), hide_index=True)

        st.success(
            f"✅ Mixture supplies approximately: Nitrogen = {supplied[0]:.2f} kg, Carbon = {supplied[1]:.2f} kg"
        )

        # ------------------------------------------------------------
        # 6️⃣  Single Waste Recommendation
        # ------------------------------------------------------------
        single_results = []
        for idx, w in wastes.iterrows():
            col = np.array([w["nitrogen_per_kg"], w["carbon_per_kg"]])
            possible = True
            required_ws = []
            for i in range(len(b)):
                if b[i] > 0:
                    if col[i] <= 0:
                        possible = False
                        break
                    else:
                        required_ws.append(b[i] / col[i])
            if not possible:
                continue
            Wj = max(required_ws) if required_ws else 0.0
            single_results.append({
                "Waste": w["waste_name"],
                "Required (kg)": float(Wj)
            })

        single_df = pd.DataFrame(single_results).sort_values("Required (kg)").reset_index(drop=True)
        st.subheader("🌿 Single Waste Recommendations (Sorted by Least Required)")
        st.dataframe(single_df.head(20), hide_index=True)

else:
    st.info("👆 Enter a crop name to get its nutrient and waste recommendations.")
