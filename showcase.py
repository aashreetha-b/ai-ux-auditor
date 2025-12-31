# showcase.py
# Single-file Context-Aware AI UX Auditor — demo-ready
# - Synthetic data
# - Heuristics
# - ViT embeddings (timm)
# - Tiny fusion regressor
# - Attention-rollout heatmap
# - Simple "AI redesign" mockup
# - PDF generator (ReportLab)
# - Streamlit UI with PDF download + Before→After
#
# Run: python -m venv venv && source venv/bin/activate
# pip install -r requirements.txt
# python showcase.py
#
# The script will try to launch Streamlit. If Streamlit fails, you can call functions locally.

import os
import io
import math
import time
import random
import json
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt

# ML libs
import torch
import torch.nn as nn
from torchvision import transforms
import timm

# Streamlit + PDF
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# OpenCV for heuristics
import cv2

# ---------------- Configuration ----------------
DATA_DIR = "data_synth"
MODEL_PATH = "models/ux_fusion_demo.pth"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- Utilities ----------------
def save_image(img, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path, quality=90)

# ---------------- Synthetic dataset generator ----------------
def make_synthetic_ui(i, w=360, h=640):
    img = Image.new("RGB", (w, h), (255,255,255))
    draw = ImageDraw.Draw(img)
    # header
    header_h = int(h * random.uniform(0.08, 0.13))
    header_color = tuple(map(int, np.random.randint(220,245,3)))
    draw.rectangle([0,0,w,header_h], fill=header_color)
    draw.text((18, header_h//4), f"App {i+1}", fill=(30,30,30))

    y = header_h + 12
    # cards & text
    while y < h - 140:
        block_h = random.choice([40,50,70,90])
        pad = random.randint(12,36)
        if random.random() < 0.2:
            # colored hero
            draw.rectangle([pad, y, w-pad, y+block_h], fill=tuple(map(int, np.random.randint(140,220,3))))
        else:
            draw.rectangle([pad, y, w-pad, y+block_h], outline=(200,200,200))
        y += block_h + random.randint(8, 26)

    # CTA
    cta_h = int(h * 0.07)
    cta_w = int(w * random.uniform(0.6, 0.95))
    cta_left = (w - cta_w)//2
    cta_top = h - cta_h - random.randint(36,80)
    if random.random() < 0.25:
        cta_color = (200,200,205)  # low contrast
        text_color = (30,30,30)
    else:
        cta_color = (10,100,200)
        text_color = (255,255,255)
    draw.rectangle([cta_left, cta_top, cta_left+cta_w, cta_top+cta_h], fill=cta_color)
    draw.text((cta_left+14, cta_top+int(cta_h*0.18)), "Buy Now", fill=text_color)
    return img

def generate_synthetic(n=120):
    meta = []
    for i in range(n):
        img = make_synthetic_ui(i)
        p = os.path.join(DATA_DIR, f"ui_{i:04d}.png")
        save_image(img, p)
        meta.append(p)
    return meta

# ---------------- Heuristics ----------------
def srgb_to_linear(c):
    if c <= 0.03928:
        return c/12.92
    else:
        return ((c+0.055)/1.055)**2.4

def rel_lum(rgb):
    r,g,b = [x/255.0 for x in rgb]
    return 0.2126*srgb_to_linear(r)+0.7152*srgb_to_linear(g)+0.0722*srgb_to_linear(b)

def contrast_ratio(rgb1, rgb2):
    L1 = rel_lum(rgb1); L2 = rel_lum(rgb2)
    lighter = max(L1,L2); darker = min(L1,L2)
    return (lighter+0.05)/(darker+0.05)

def compute_whitespace(pil_img):
    arr = np.asarray(pil_img).astype(np.float32)/255.0
    return float(np.mean(np.all(arr>0.95,axis=2)))

def compute_contrast_estimate(pil_img):
    arr = np.asarray(pil_img).astype(np.float32)/255.0
    lum = 0.2126*arr[:,:,0]+0.7152*arr[:,:,1]+0.0722*arr[:,:,2]
    return float(lum.std())

def estimate_cta_bbox(pil_img):
    arr = np.asarray(pil_img.convert("RGB"))
    h,w = arr.shape[:2]
    bottom = arr[int(h*0.55):,:,:]
    gray = cv2.cvtColor(bottom, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50,150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area=0
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        area = ww*hh
        if area>500 and ww/hh>2 and hh>20:
            if area>best_area:
                best_area=area; best=(x,int(h*0.55)+y,ww,hh)
    return best

def heuristics_audit(pil_img):
    whitespace = compute_whitespace(pil_img)
    contrast = compute_contrast_estimate(pil_img)
    cta = estimate_cta_bbox(pil_img)
    cta_contrast = None
    if cta:
        x,y,ww,hh = cta
        arr = np.asarray(pil_img.convert("RGB"))
        cta_mean = arr[y:y+hh,x:x+ww].mean(axis=(0,1)).astype(int)
        bg_mean = arr.mean(axis=(0,1)).astype(int)
        cta_contrast = contrast_ratio(tuple(cta_mean.tolist()), tuple(bg_mean.tolist()))
    # composite score
    read = max(0.0, min(1.0, 0.6 - (arr.mean()/255.0 - 0.5)))
    vis = max(0.0,min(1.0,(min(10, cta_contrast or 3)/10*0.6 + (1 - abs(whitespace-0.15))*0.4)))
    spacing = 1.0 - max(0.0, min(1.0, abs(whitespace-0.15)/0.3))
    usability = float(np.clip((read*0.4 + vis*0.35 + spacing*0.25)*100.0, 0, 100))
    issues=[]
    if cta_contrast is None or cta_contrast<3.0:
        issues.append("CTA may have insufficient contrast.")
    if whitespace < 0.08:
        issues.append("Low whitespace — content feels cramped.")
    return {
        "whitespace": whitespace,
        "contrast": contrast,
        "cta_bbox": cta,
        "cta_contrast": cta_contrast,
        "readability_score": read,
        "visual_hierarchy_score": vis,
        "spacing_score": spacing,
        "usability_score": usability,
        "issues": issues
    }

# ---------------- Encoder (small wrapper) ----------------
class SimpleEncoder:
    def __init__(self, model_name="vit_base_patch16_224", device=DEVICE):
        self.device = device
        self.model_name = model_name
        try:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg').to(self.device).eval()
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            self.embed_dim = self.model.num_features
            self.ok = True
        except Exception as e:
            print("[ENC] timm load failed, falling back to random encoder:", e)
            self.model = None
            self.embed_dim = 512
            self.ok = False
    def encode(self, pil_img):
        if self.ok:
            x = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.model.forward_features(x) if hasattr(self.model, "forward_features") else self.model(x)
            return feats.cpu().numpy().reshape(-1)
        else:
            # deterministic pseudo-embedding from image statistics
            arr = np.asarray(pil_img.resize((64,64))).astype(np.float32)/255.0
            stats = np.concatenate([arr.mean(axis=(0,1)), arr.std(axis=(0,1)), np.histogram(arr.flatten(), bins=10, range=(0,1))[0]/(64*64)])
            v = np.zeros(self.embed_dim)
            v[:len(stats)] = stats
            return v

# ---------------- Fusion model (tiny, train quickly) ----------------
class UXFusionTiny(nn.Module):
    def __init__(self, emb_dim, heur_dim=4, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim + heur_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.out = nn.Linear(hidden//2, 1)
    def forward(self, emb, heur):
        x = torch.cat([emb, heur], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x).squeeze(1)

# ---------------- Attention rollup for ViT (best-effort) ----------------
def vit_attention_rollup_heatmap(model, pil_img):
    # This attempts to extract attention maps if timm ViT exposes them.
    # If not available, return None.
    try:
        # transform
        t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        x = t(pil_img).unsqueeze(0).to(DEVICE)
        attn_maps=[]
        hooks=[]
        def hook_fn(module, inp, out):
            # out maybe attn weights
            try:
                attn_maps.append(out.detach().cpu())
            except Exception:
                pass
        for name,m in model.named_modules():
            if name.endswith("attn") or "attn" in name:
                try:
                    hooks.append(m.register_forward_hook(hook_fn))
                except Exception:
                    pass
        # run forward features path
        if hasattr(model, "forward_features"):
            _ = model.forward_features(x)
        else:
            _ = model(x)
        for h in hooks: h.remove()
        if not attn_maps: return None
        A = torch.cat(attn_maps, dim=0)
        A = A.mean(dim=2)  # mean heads
        A = A.mean(dim=0)  # mean layers
        A = A[0]  # N,N
        cls_attn = A[0,1:]
        n = int(math.sqrt(cls_attn.shape[0]))
        hmap = cls_attn.reshape(n,n).numpy()
        hmap = (hmap - hmap.min())/(hmap.max()-hmap.min()+1e-9)
        return hmap
    except Exception:
        return None

# ---------------- LLM stub (deterministic, safe) ----------------
def llm_recommendations(audit: Dict[str,Any], top_k=4):
    # Deterministic template that reads audit fields and returns actionable recommendations
    issues = audit.get("issues", [])
    recs=[]
    if "CTA may have insufficient contrast." in issues:
        recs.append({"issue":"CTA contrast","suggestion":"Set CTA background to #0A64C8 and text to #FFFFFF; increase vertical padding to 14px.","impact":35,"confidence":"high"})
    if audit["whitespace"]<0.10:
        recs.append({"issue":"Whitespace","suggestion":"Increase vertical spacing between content blocks by 12px and adopt 16px baseline grid.","impact":20,"confidence":"high"})
    if not recs:
        recs.append({"issue":"Minor polish","suggestion":"Consider subtle increase in CTA prominence and consistent left padding (16px).","impact":10,"confidence":"medium"})
    return {"summary":"Automated recommendations generated locally.", "prioritized_recommendations":recs, "raw_text":"local_stub"}

# ---------------- Simple redesign generator ----------------
def generate_redesign(pil_img, audit):
    # Apply deterministic structural improvements to create an "after" mockup:
    img = pil_img.convert("RGB")
    # Add padding
    img = ImageOps.expand(img, border=24, fill=(250,250,250))
    # Slightly brighten and enhance color/contrast
    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.12)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    # Emphasize CTA region if found: draw border and shadow
    cta = audit.get("cta_bbox")
    if cta:
        x,y,ww,hh = cta
        draw = ImageDraw.Draw(img)
        # map coords due to padding: we added 24px
        pad=24
        rx,ry = x+pad, y+pad
        draw.rectangle([rx-6, ry-6, rx+ww+6, ry+hh+6], outline=(10,80,200), width=6)
    # slight smoothing to resemble a designed mockup
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img

# ---------------- PDF report generator (simple, polished) ----------------
def generate_pdf_report(audit, llm_parsed, screenshot_path, heatmap_array, output_path, author="Your Name"):
    c = canvas.Canvas(output_path, pagesize=A4)
    W, H = A4
    # Cover
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, H-80, "UX Audit Report")
    c.setFont("Helvetica", 11)
    c.drawString(40, H-100, f"Generated by {author} — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFont("Helvetica", 10)
    c.drawString(40, H-130, f"Usability Score: {audit.get('usability_score',0):.1f}/100")
    c.showPage()

    # Metrics + screenshot
    c.setFont("Helvetica-Bold", 14); c.drawString(40, H-40, "Executive Summary")
    c.setFont("Helvetica", 10)
    c.drawString(40, H-60, llm_parsed.get("summary",""))
    # add screenshot small
    try:
        c.drawImage(screenshot_path, 40, H-360, width=250, height=300)
    except Exception:
        pass
    # heuristics table
    c.setFont("Helvetica-Bold", 12); c.drawString(320, H-80, "Key Metrics")
    c.setFont("Helvetica", 10)
    lines = [
        f"Readability: {audit.get('readability_score',0):.2f}",
        f"Visual Hierarchy: {audit.get('visual_hierarchy_score',0):.2f}",
        f"Spacing Score: {audit.get('spacing_score',0):.2f}",
        f"Whitespace: {audit.get('whitespace',0):.3f}",
        f"CTA contrast: {audit.get('cta_contrast')}"
    ]
    y=H-100
    for ln in lines:
        c.drawString(320, y, ln); y-=14
    c.showPage()

    # Recommendations
    c.setFont("Helvetica-Bold", 14); c.drawString(40, H-40, "Prioritized Recommendations")
    c.setFont("Helvetica", 10)
    y=H-70
    for r in llm_parsed.get("prioritized_recommendations",[]):
        c.drawString(48, y, f"- {r['issue']}: {r['suggestion']} (impact {r.get('impact',0)})"); y-=16
        if y<60:
            c.showPage(); y=H-60
    c.showPage()

    # Append heatmap image if present
    if heatmap_array is not None:
        # create overlay png in-memory
        buf = _create_heatmap_overlay(screenshot_path, heatmap_array)
        try:
            c.drawImage(buf, 40, 200, width=500, height=600)
            c.showPage()
        except Exception:
            pass

    c.save()
    return output_path

def _create_heatmap_overlay(screenshot_path, heatmap):
    img = Image.open(screenshot_path).convert("RGBA")
    W,H = img.size
    h = heatmap.copy()
    if h.max()>0:
        h = (h - h.min())/(h.max()-h.min()+1e-9)
    cmap = plt.get_cmap("jet")
    hm = (cmap(h)[:,:,:3]*255).astype(np.uint8)
    hm_pil = Image.fromarray(hm).resize((W,H)).convert("RGBA")
    # apply alpha derived from intensity
    alpha = Image.fromarray((np.clip(np.kron(h, np.ones((int(H/h.shape[0]), int(W/h.shape[1])))),0,1)*160).astype(np.uint8)).resize((W,H))
    hm_pil.putalpha(alpha)
    combined = Image.alpha_composite(img, hm_pil)
    buf = io.BytesIO(); combined.save(buf, format="PNG"); buf.seek(0)
    return buf

# ---------------- Quick training pipeline (tiny) ----------------
def quick_train(meta_paths, encoder: SimpleEncoder, epochs=3):
    # create features and labels (heuristic score used as label)
    X_emb=[]; X_heu=[]; Y=[]
    for p in meta_paths:
        img = Image.open(p).convert("RGB")
        audit = heuristics_audit(img)
        emb = encoder.encode(img)
        heur = np.array([audit['contrast'], audit['whitespace'], audit['readability_score'], audit['visual_hierarchy_score']], dtype=np.float32)
        X_emb.append(emb)
        X_heu.append(heur)
        Y.append(audit['usability_score']/100.0)  # 0..1
    X_emb = torch.tensor(np.stack(X_emb), dtype=torch.float32)
    X_heu = torch.tensor(np.stack(X_heu), dtype=torch.float32)
    Y = torch.tensor(np.array(Y,dtype=np.float32))
    dataset = torch.utils.data.TensorDataset(X_emb, X_heu, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    model = UXFusionTiny(emb_dim=X_emb.shape[1], heur_dim=X_heu.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        model.train()
        tot=0.0
        for xb, xh, y in loader:
            xb=xb.to(DEVICE); xh=xh.to(DEVICE); y=y.to(DEVICE)
            pred = model(xb, xh)
            loss = ((pred-y)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs} loss={tot/len(loader.dataset):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    return model

# ---------------- Inference pipeline ----------------
def run_full_pipeline_on_image(img_path, encoder):
    img = Image.open(img_path).convert("RGB")
    audit = heuristics_audit(img)
    emb = encoder.encode(img)
    # load tiny model if present
    heur_vec = torch.tensor([[audit['contrast'], audit['whitespace'], audit['readability_score'], audit['visual_hierarchy_score']]], dtype=torch.float32)
    # Try to load model
    try:
        model = UXFusionTiny(emb_dim=encoder.embed_dim, heur_dim=4).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            emb_t = torch.tensor(emb).unsqueeze(0).to(DEVICE)
            pred = model(emb_t, heur_vec.to(DEVICE)).cpu().numpy().item()
            score = float(pred*100.0)
    except Exception:
        # fallback: heuristic score
        score = audit['usability_score']
    # heatmap
    heatmap = None
    if encoder.ok:
        try:
            heatmap = vit_attention_rollup_heatmap(encoder.model, img)
        except Exception:
            heatmap=None
    # LLM recommendations (local stub)
    llm = llm_recommendations(audit)
    return {"audit":audit, "score":score, "heatmap":heatmap, "llm":llm, "image_path":img_path}

# ---------------- Streamlit UI ----------------
def run_streamlit_app():
    st.set_page_config(page_title="AI UX Auditor", layout="wide")
    st.title("Context-Aware AI UX Auditor — Showcase")
    st.write("Upload a UI screenshot to run a quick audit (local demo). This demo runs a deterministic LLM-stub; no API keys needed.")

    # left column upload + controls
    col1, col2 = st.columns([1,1])
    with col1:
        uploaded = st.file_uploader("Upload screenshot", type=["png","jpg","jpeg"])
        if st.button("Generate synthetic dataset (120 images)"):
            with st.spinner("Generating synthetic images..."):
                meta = generate_synthetic(120)
            st.success(f"Saved {len(meta)} synthetic images to {DATA_DIR}")
        if st.button("Quick train (tiny model, few epochs)"):
            with st.spinner("Training..."):
                meta = [str(p) for p in Path(DATA_DIR).glob("*.png")]
                enc = SimpleEncoder()
                quick_train(meta[:80], enc, epochs=3)
            st.success("Trained tiny fusion model (saved)")

    with col2:
        st.markdown("### Demo actions")
        if st.button("Run Demo on example synthetic image"):
            # pick a random synth image (or generate if none)
            imgs = list(Path(DATA_DIR).glob("*.png"))
            if not imgs:
                generate_synthetic(40)
                imgs = list(Path(DATA_DIR).glob("*.png"))
            p = str(random.choice(imgs))
            st.info(f"Running pipeline on {p}")
            enc = SimpleEncoder()
            res = run_full_pipeline_on_image(p, enc)
            st.metric("Predicted UX Score", f"{res['score']:.1f}/100")
            st.write("Heuristic audit:", res['audit'])
            if res['heatmap'] is not None:
                hm = res['heatmap']
                # upscale for display
                hm_big = np.kron(hm, np.ones((int(300/hm.shape[0]), int(200/hm.shape[1]))))
                st.image(hm_big, caption="Attention heatmap (upscaled)")
            st.write("LLM Recommendations:", res['llm'])
            # generate redesigned mockup
            red = generate_redesign(Image.open(p), res['audit'])
            outp = "tmp/redesigned_demo.png"
            save_image(red, outp)
            st.image(outp, caption="AI Redesign (structural edits)")

    st.markdown("---")
    st.header("Upload and audit your own screenshot")
    uploaded2 = st.file_uploader("Upload screenshot to audit", key="u2", type=["png","jpg","jpeg"])
    if uploaded2:
        pth = f"tmp/uploaded_{int(time.time())}.png"
        with open(pth, "wb") as f: f.write(uploaded2.getvalue())
        enc = SimpleEncoder()
        res = run_full_pipeline_on_image(pth, enc)
        st.metric("Predicted UX Score", f"{res['score']:.1f}/100")
        st.write("Heuristic audit:", res['audit'])
        if res['heatmap'] is not None:
            hm = res['heatmap']; hm_big = np.kron(hm, np.ones((int(300/hm.shape[0]), int(200/hm.shape[1]))))
            st.image(hm_big, caption="Attention heatmap")
        st.write("LLM Recommendations:", res['llm'])
        # redesigned image
        red = generate_redesign(Image.open(pth), res['audit'])
        outp = "tmp/redesigned_user.png"
        save_image(red, outp)
        st.image(outp, caption="AI Redesigned mockup")
        # PDF generation
        if st.button("Generate PDF Audit Report"):
            out_pdf = f"reports/ux_audit_{int(time.time())}.pdf"
            generate_pdf_report(res['audit'], res['llm'], pth, res['heatmap'], out_pdf, author="Your Name")
            st.success("PDF generated!")
            with open(out_pdf, "rb") as f:
                st.download_button("Download PDF", f, file_name=os.path.basename(out_pdf), mime="application/pdf")

# ---------------- CLI entry ----------------
if __name__ == "__main__":
    run_streamlit_app()
