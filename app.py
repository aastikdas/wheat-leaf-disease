import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wheat Leaf Disease Diagnostic & Fertilizer Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Professional Styling (Fixed Navbar, Smooth Light Yellow Background, Clean Typography)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* 1. Base App & Smooth Light Yellow Background */
    html, body, .stApp {
        background-color: #FAF5E8 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        color: #2D3748 !important;
        font-size: 16px;
    }

    /* 2. Preserve Streamlit Icons (Material Symbols) */
    [data-testid*="Icon"], [data-testid*="icon"], .stIcon {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    }

    /* 3. Fixed Sticky Top Navbar */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 56px !important;
        z-index: 999995 !important;
    }
    
    .fixed-navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 56px;
        background-color: #1C3B1A;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 2rem;
        z-index: 999990;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }
    .navbar-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 0.2px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .navbar-title span.accent {
        color: #F6E05E;
    }

    /* 4. Main Content Constrain (Prevents wide empty voids) */
    .block-container {
        padding-top: 82px !important;
        padding-bottom: 50px !important;
        max-width: 960px !important;
        margin: 0 auto !important;
    }

    /* 5. Clean White Cards on Light Yellow BG */
    .app-card {
        background-color: #FFFFFF;
        border: 1px solid #EADBBA;
        border-radius: 10px;
        padding: 22px 26px;
        margin-bottom: 22px;
        box-shadow: 0 2px 10px rgba(120, 100, 40, 0.05);
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1C3B1A;
        margin-bottom: 12px;
    }

    /* 6. Supported Disease Badges */
    .pill-badge {
        display: inline-block;
        background-color: #F4EEDC;
        color: #3F3C1E;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 5px 13px;
        border-radius: 16px;
        margin-right: 8px;
        margin-bottom: 6px;
        border: 1px solid #DFCFA8;
    }

    /* 7. Output Line Styling */
    .report-line {
        font-size: 1.2rem;
        margin-bottom: 12px;
        color: #2D3748;
    }
    .report-disease {
        font-weight: 700;
        color: #1C3B1A;
    }
    .report-note {
        font-size: 1.0rem;
        color: #4A5568;
        margin-top: 8px;
        margin-bottom: 16px;
        font-style: italic;
    }
    .fert-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1C3B1A;
        margin-top: 16px;
        margin-bottom: 6px;
    }
    .fert-item {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #2D3748;
    }

    /* 8. Button Styling */
    div.stButton > button[kind="primary"] {
        background-color: #2D5A27 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 9px 24px !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #22451E !important;
    }
</style>

<!-- Fixed Navbar -->
<div class="fixed-navbar">
    <div class="navbar-title">
        🌾 WheatGuard <span class="accent">|</span> Wheat Leaf Disease Diagnostic & Fertilizer Advisory
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model Loading & Checkpoint Verification
# -----------------------------------------------------------------------------
MODEL_FILE = "best_wheat_disease_model.pth"
SAMPLES_DIR = "Samples"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, None, None
    checkpoint = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
    class_names = checkpoint.get('class_names', ['Crown Root Rot', 'Healthy', 'Leaf Rust', 'Loose Smut'])
    
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(class_names))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((checkpoint.get('input_size', 224), checkpoint.get('input_size', 224))),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint.get('norm_mean', [0.485, 0.456, 0.406]),
                             checkpoint.get('norm_std', [0.229, 0.224, 0.225]))
    ])
    return model, transform, class_names

model, transform, class_names = load_model()

# -----------------------------------------------------------------------------
# Grad-CAM Engine (Interpolated to match image size perfectly)
# -----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register()

    def _register(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return heatmap

    def clean(self):
        for h in self.hooks:
            h.remove()

# -----------------------------------------------------------------------------
# Rule-Based Fertilizer Advisory (Concise, 3 items each with dosage)
# -----------------------------------------------------------------------------
def get_recommendations(pred_class):
    if pred_class == "Healthy":
        return {
            "display_name": "Healthy Wheat Leaf",
            "note": "Crop is in good condition now, but maintain proper care to keep it healthy.",
            "fertilizers": [
                {
                    "name": "Urea (Nitrogen 46% N)",
                    "dosage": "100–120 kg/ha split across basal, CRI, and tillering stages"
                },
                {
                    "name": "Single Super Phosphate - SSP (16% P₂O₅)",
                    "dosage": "250–300 kg/ha applied as basal dose before sowing"
                },
                {
                    "name": "Muriate of Potash - MOP (60% K₂O)",
                    "dosage": "40–50 kg/ha for stalk strength and lodging resistance"
                }
            ]
        }
    elif pred_class == "Leaf Rust":
        return {
            "display_name": "Leaf Rust",
            "note": None,
            "fertilizers": [
                {
                    "name": "Tilt (Propiconazole 25% EC)",
                    "dosage": "1 ml per litre of water (500 ml/ha) foliar spray at first appearance"
                },
                {
                    "name": "Folicur (Tebuconazole 250 EC)",
                    "dosage": "1 ml per litre of water, repeat after 12–15 days if rust persists"
                },
                {
                    "name": "Potassium Nitrate (13:0:45)",
                    "dosage": "10–15 g per litre foliar spray to boost immunity and grain filling"
                }
            ]
        }
    elif pred_class == "Loose Smut":
        return {
            "display_name": "Loose Smut",
            "note": None,
            "fertilizers": [
                {
                    "name": "Vitavax Power (Carboxin 37.5% + Thiram 37.5%)",
                    "dosage": "2.5 g per kg seed (seed treatment before sowing)"
                },
                {
                    "name": "Raxil (Tebuconazole 2% DS)",
                    "dosage": "1.25 g per kg seed (dry seed dressing)"
                },
                {
                    "name": "Balanced NPK with Sulphur (20:20:0:13)",
                    "dosage": "100–120 kg/ha basal dose during sowing"
                }
            ]
        }
    else: # Crown Root Rot
        return {
            "display_name": "Crown Root Rot",
            "note": None,
            "fertilizers": [
                {
                    "name": "Trichoderma viride / harzianum (Bio-agent)",
                    "dosage": "4–5 g per kg seed or 2.5 kg/ha mixed with FYM/organic compost"
                },
                {
                    "name": "Saaf (Carbendazim 12% + Mancozeb 63% WP)",
                    "dosage": "2 g per litre of water (soil drenching around root zone)"
                },
                {
                    "name": "Zinc Sulphate + Single Super Phosphate (SSP)",
                    "dosage": "25 kg/ha Zinc Sulphate + 250 kg/ha SSP basal application"
                }
            ]
        }

# -----------------------------------------------------------------------------
# System Overview Card (Gives context so the page never looks empty)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-card">
    <div class="card-title">🌾 Crop Health Diagnostic Portal</div>
    <div style="font-size: 1.02rem; color: #4A5568; line-height: 1.6; margin-bottom: 14px;">
        Automated wheat leaf disease identification system powered by EfficientNet-B0 deep transfer learning.
        Upload a field photo or choose from verified test samples to diagnose leaf condition and receive targeted fertilizer recommendations.
    </div>
    <div>
        <span class="pill-badge">🌿 Crown Root Rot</span>
        <span class="pill-badge">🍃 Leaf Rust</span>
        <span class="pill-badge">🌾 Loose Smut</span>
        <span class="pill-badge">✅ Healthy / Normal</span>
        <span class="pill-badge" style="background-color: #E2E8F0; color: #2D3748; border-color: #CBD5E0;">🎯 Model: EfficientNet-B0 (96.8% Acc)</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(f"Model file `{MODEL_FILE}` not found. Please place `best_wheat_disease_model.pth` in the project root.")
    st.stop()

# -----------------------------------------------------------------------------
# Session State Management
# -----------------------------------------------------------------------------
if "current_image_key" not in st.session_state:
    st.session_state.current_image_key = None
if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

# -----------------------------------------------------------------------------
# Step 1: Input Selection
# -----------------------------------------------------------------------------
st.markdown('<h3 style="color: #1C3B1A; margin-top: 10px; margin-bottom: 15px;">1. Select Leaf Image(s)</h3>', unsafe_allow_html=True)

input_method = st.radio(
    "Choose input source:",
    ["Pick from Sample Images", "Upload My Own Leaf Image(s)"],
    horizontal=True
)

selected_images = []

if input_method == "Pick from Sample Images":
    if os.path.exists(SAMPLES_DIR) and len(os.listdir(SAMPLES_DIR)) > 0:
        sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        dropdown_options = ["Click here to select", "All Sample Images (Batch 6)"] + sample_files
        
        chosen_sample = st.selectbox(
            "Select a sample image from the dataset:",
            dropdown_options,
            index=0
        )
        
        if chosen_sample == "All Sample Images (Batch 6)":
            for sf in sample_files:
                sample_path = os.path.join(SAMPLES_DIR, sf)
                selected_images.append({
                    "image": Image.open(sample_path).convert('RGB'),
                    "label": sf
                })
        elif chosen_sample != "Click here to select":
            sample_path = os.path.join(SAMPLES_DIR, chosen_sample)
            selected_images.append({
                "image": Image.open(sample_path).convert('RGB'),
                "label": chosen_sample
            })
    else:
        st.warning(f"No sample images found in `{SAMPLES_DIR}` directory.")
else:
    uploaded_files = st.file_uploader(
        "Upload wheat leaf photo(s) (JPG or PNG) — you can select multiple files:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for uf in uploaded_files:
            try:
                selected_images.append({
                    "image": Image.open(uf).convert('RGB'),
                    "label": uf.name
                })
            except Exception as e:
                st.error(f"Error loading {uf.name}: {e}")

# Reset prediction if image selection changed
active_key = ",".join([item["label"] for item in selected_images]) if selected_images else None
if active_key != st.session_state.current_image_key:
    st.session_state.current_image_key = active_key
    st.session_state.run_prediction = False

# Action button and preview before diagnosis
if selected_images:
    st.markdown("<div style='margin-top: 12px; margin-bottom: 14px;'>", unsafe_allow_html=True)
    if len(selected_images) == 1:
        st.image(selected_images[0]["image"], width=220, caption=f"Selected: {selected_images[0]['label']}")
    else:
        st.write(f"**{len(selected_images)} images selected in order:**")
        cols = st.columns(min(len(selected_images), 6))
        for i, item in enumerate(selected_images):
            with cols[i % len(cols)]:
                st.image(item["image"], width=130, caption=f"#{i+1}: {item['label']}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    btn_text = "Diagnose Leaf" if len(selected_images) == 1 else f"Diagnose {len(selected_images)} Leaves (In Order)"
    if st.button(btn_text, type="primary"):
        st.session_state.run_prediction = True

# -----------------------------------------------------------------------------
# Step 2: Diagnostic & Fertilizer Output (Shown in order after clicking button)
# -----------------------------------------------------------------------------
if selected_images and st.session_state.run_prediction:
    st.markdown("---")
    st.markdown('<h3 style="color: #1C3B1A; margin-top: 10px; margin-bottom: 18px;">2. Diagnosis & Fertilizer Recommendation</h3>', unsafe_allow_html=True)

    for idx, item in enumerate(selected_images, start=1):
        leaf_img = item["image"]
        leaf_label = item["label"]
        input_tensor = transform(leaf_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx]) * 100.0

        rec_data = get_recommendations(pred_class)

        if len(selected_images) > 1:
            st.markdown(f'<div style="background-color: #F4EEDC; padding: 7px 14px; border-radius: 6px; font-weight: 700; color: #1C3B1A; margin-bottom: 15px; border-left: 4px solid #1C3B1A;">Image #{idx} of {len(selected_images)}: {leaf_label}</div>', unsafe_allow_html=True)

        col_img, col_info = st.columns([1, 1.3], gap="large")

        with col_img:
            # Compact image preview (not giant)
            st.image(leaf_img, width=300, caption=leaf_label)
            
            # Unique key for each image's checkbox
            cam_key = f"cam_toggle_{idx}_{abs(hash(leaf_label))}"
            show_cam = st.checkbox("Show AI Symptom Focus Heatmap (Grad-CAM)", value=False, key=cam_key)
            if show_cam:
                try:
                    cam = GradCAM(model, model.features[-1])
                    heatmap = cam.generate(input_tensor, pred_idx)
                    cam.clean()

                    rgb_resized = np.array(leaf_img.resize((224, 224))) / 255.0
                    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
                    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
                    overlay = np.clip(0.6 * rgb_resized + 0.4 * heatmap_colored, 0, 1)
                    st.image(overlay, width=300, caption=f"Grad-CAM Heatmap: {leaf_label}")
                except Exception as e:
                    st.caption(f"Heatmap preview unavailable: {e}")

        with col_info:
            # Single clean lines
            st.markdown(f'<div class="report-line"><b>Disease Name:</b> <span class="report-disease">{rec_data["display_name"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="report-line"><b>Prediction Confidence:</b> {confidence:.2f}%</div>', unsafe_allow_html=True)
            
            if rec_data.get("note"):
                st.markdown(f'<div class="report-note">{rec_data["note"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="fert-header">Recommended Fertilizers & Dosage:</div>', unsafe_allow_html=True)
            st.markdown('<ol style="margin-top: 6px; padding-left: 20px;">', unsafe_allow_html=True)
            for fert in rec_data["fertilizers"]:
                st.markdown(f'''
                <li style="margin-bottom: 12px;">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #2D3748;">{fert["name"]}</div>
                    <div style="font-size: 0.96rem; color: #4A5568; margin-top: 2px;"><b>Dosage:</b> {fert["dosage"]}</div>
                </li>
                ''', unsafe_allow_html=True)
            st.markdown('</ol>', unsafe_allow_html=True)

        # Divider between multiple results
        if idx < len(selected_images):
            st.markdown("<hr style='border: none; border-top: 1px dashed #DFCFA8; margin: 30px 0;'>", unsafe_allow_html=True)
