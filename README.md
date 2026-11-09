# 🌾 Farming with Foresight: Wheat Leaf Disease Prediction & Fertilizer Advisory
**Society Academia Industry (SAI-III) Internship Project**

An end-to-end computer vision and agronomic decision-support system designed to identify foliar wheat diseases with high accuracy, explain predictions using Grad-CAM heatmaps, and provide targeted fertilizer/fungicide remedies.

---

## 📌 Project Overview
* **Target Pathology Taxonomy (4 Core Classes)**:
  1. `Healthy` (Normal foliage)
  2. `Leaf Rust` (*Puccinia triticina*)
  3. `Loose Smut` (*Ustilago tritici*)
  4. `Crown Root Rot` (*Fusarium pseudograminearum / Bipolaris sorokiniana*)
* **Best Model**: `EfficientNet-B0` (Transfer Learning)
  * **Test Accuracy**: **96.79%**
  * **Macro F1-Score**: **96.47%**
* **Deployment**: Interactive Streamlit Web Interface (`app.py`)

---

## 📁 Repository Structure
```
├── app.py                                   # Streamlit web application
├── run_app.bat                              # One-click Windows launcher for app.py
├── best_wheat_disease_model.pth             # Trained EfficientNet-B0 weights & metadata
├── fertilizer_recommendations.json          # Agronomic remedies & chemical dosages
├── wheat_leaf_disease_pipeline.ipynb        # Complete Colab training & evaluation pipeline
├── wheat leaf disease prediciton and fertilizer recommendation.pdf  # Internship presentation
└── Samples/                                 # Test verification images
    ├── Sample 1 Loose Smut.png
    ├── Sample 2 Leaf Rust.png
    ├── Sample 3 Healthy.png
    ├── Sample 4 Healthy.png
    ├── Sample 5 Loose Smut.png
    └── Sample 6 Leaf Rust.png
```

---

## 🚀 Running the Streamlit Web Application

### Method 1: One-Click Launcher (Windows)
Double-click `run_app.bat`.

### Method 2: Command Line
Open PowerShell or Terminal in this directory and execute:
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 🛠️ Key Application Features
1. **Dual Input Mode**:
   - Choose directly from the sample test images in `Samples/`.
   - Or upload any custom wheat leaf image (`.jpg`, `.jpeg`, `.png`).
2. **Pathology Classification & Confidence**:
   - Displays clear status badges (`HEALTHY` vs. `DISEASE DETECTED`).
   - Displays prediction confidence percentage and class probability breakdown.
3. **Grad-CAM AI Focus**:
   - Optional toggle to inspect the exact leaf lesion regions triggering the classification.
4. **Rule-Based 3-Tier Fertilizer & Treatment Advisory**:
   - **Healthy**: Balanced NPK (Urea, SSP, MOP) split application to maintain natural immunity.
   - **Leaf Rust**: Propiconazole 25% EC, Tebuconazole 250 EC, and foliar potassium (pause excess nitrogen).
   - **Loose Smut**: Carboxin + Thiram (Vitavax Power) seed treatment, Tebuconazole (Raxil), and balanced NPK+S for unaffected tillers.
   - **Crown Root Rot**: Trichoderma viride bio-agent, Carbendazim + Mancozeb (Saaf), and Zinc Sulphate + SSP.

---

## 👥 Project Team
* **Aastik Das** (CSE) - 230101001
* **Jyoti Saxena** (CSE) - 230101064
* **Maniranjan** (CSE) - 230101072
* **Sankalp Kumar Sinha** (CSE) - 230101110
* **Sandeep Kumar** (MnC) - 230104016
* **Sarbajit Kundu** (MAE) - 230103038

