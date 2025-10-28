import os, io, time, tempfile
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pdf2image import convert_from_path
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

#  PAGE SETUP
st.set_page_config(page_title="AI Financial Analyzer", page_icon="📊", layout="wide")


# its style time!
st.markdown("""
<style>
:root{
  --blue:#00A9E0;
  --blue-dark:#0088B8;
  --yellow:#FFB81C;
  --text:#1A1A1A;
  --text-light:#4A4A4A;
  --bg:#FAFAFA;
  --white:#FFFFFF;
  --border:#E0E0E0;
  --shadow: 0 2px 8px rgba(0,169,224,0.15);
}

/* Force light mode */
html, body, [data-testid="stAppViewContainer"], 
[data-testid="stApp"], .main, .block-container{ 
  background: var(--bg) !important; 
  color: var(--text) !important;
}

[data-testid="stHeader"]{ 
  background: transparent !important; 
}

.block-container{ 
  padding-top: 1rem !important; 
  padding-bottom: 3rem !important; 
}

/* Fix all text colors for light mode */
.stMarkdown, .stMarkdown p, .stMarkdown span, 
div[data-testid="stMarkdownContainer"], 
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span,
.element-container, h1, h2, h3, h4, h5, h6, p, span, label,
div, section {
  color: var(--text) !important;
}

/* Separate header with logo */
.top-header{
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 16px 24px;
  background: var(--white);
  border-bottom: 3px solid transparent;
  border-image: linear-gradient(90deg, var(--blue), var(--yellow)) 1;
  margin-bottom: 24px;
}

.logo-container img{
  width: 120px;
  height: auto;
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}

.header-title{
  flex: 1;
}

.header-title h1{
  font-size: 28px;
  font-weight: 700;
  color: var(--text) !important;
  margin: 0;
  background: linear-gradient(135deg, var(--blue), var(--yellow));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-subtitle{
  font-size: 14px;
  color: var(--text-light) !important;
  margin: 4px 0 0 0;
}

/* Hero section with gradient */
.hero-section{
  background: linear-gradient(135deg, var(--blue) 0%, var(--yellow) 100%);
  padding: 32px;
  border-radius: 20px;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
  text-align: center;
}

.hero-title{
  font-size: 32px;
  font-weight: 700;
  color: white !important;
  margin: 0 0 8px 0;
  text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.hero-subtitle{
  font-size: 16px;
  color: white !important;
  opacity: 0.95;
  margin: 0;
}


/* File uploader light mode */
[data-testid="stFileUploaderDropzone"] {
  background: transparent !important;
  border: 2px dashed var(--blue) !important;
  border-radius: 14px !important;
  transition: all 0.3s ease;
  text-align: center !important;
  box-shadow: none !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
  background: #eaf6fb !important;
}

[data-testid="stFileUploaderDropzone"] * {
  color: var(--blue) !important;
}

[data-testid="stFileUploaderDropzone"] button {
  background: var(--blue) !important;
  color: white !important;
  border: none !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
}

[data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
  color: var(--blue) !important;
}
            
/* Radio button blue style */
div[role="radiogroup"] > label[data-baseweb="radio"] {
  color: #0077b6 !important;
}

div[role="radio"] > div:first-child {
  border: 2px solid #0077b6 !important;
}

div[role="radio"][aria-checked="true"] > div:first-child {
  background-color: #0077b6 !important;
  border-color: #0077b6 !important;
}

/* Progress section */
.progress-section{
  background: var(--white);
  border: 2px solid var(--border);
  border-radius: 16px;
  padding: 20px 24px;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
}

.file-progress{
  display: flex;
  align-items: center;
  gap: 16px;
}

.file-info{
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
}

.file-icon{
  font-size: 32px;
}

.file-details{
  flex: 1;
}

.file-name{
  font-weight: 700;
  color: var(--text) !important;
  margin: 0;
  font-size: 16px;
}

.file-size{
  font-size: 13px;
  color: var(--text-light) !important;
  margin: 2px 0 0 0;
}

.progress-bar-container{
  width: 220px;
  height: 10px;
  background: #E8E8E8;
  border-radius: 10px;
  overflow: hidden;
}

.progress-bar-fill{
  height: 100%;
  background: linear-gradient(90deg, var(--blue), var(--yellow));
  border-radius: 10px;
  transition: width 0.3s ease;
  box-shadow: 0 0 10px rgba(0,169,224,0.4);
}

.progress-text{
  font-size: 15px;
  font-weight: 700;
  color: var(--blue) !important;
  min-width: 50px;
  text-align: right;
}

/* Cards */
.card{
  background: var(--white);
  border: 2px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  box-shadow: var(--shadow);
  margin-bottom: 24px;
}

.card h4{
  margin: 0 0 20px 0;
  color: var(--text) !important;
  font-size: 22px;
  font-weight: 700;
}

/* Grid */
.grid-5{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 20px;
  margin: 20px 0;
}

.thumb-title{
  text-align: center;
  font-weight: 700;
  color: var(--text) !important;
  margin: 8px 0 6px;
  font-size: 16px;
}

.thumb-sub{
  text-align: center;
  color: var(--text-light) !important;
  font-size: 14px;
  margin: 0 0 12px;
}

/* Chips */
.chips{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-top: 10px;
}

.chip{
  font-size: 12px;
  padding: 6px 14px;
  border-radius: 20px;
  border: 2px solid var(--blue);
  background: var(--white);
  color: var(--blue) !important;
  font-weight: 600;
}

/* Images */
img:not(.logo-container img){
  border-radius: 12px;
  border: 2px solid var(--border);
}

/* Buttons */
.stButton > button{
  background: linear-gradient(135deg, var(--blue) 0%, var(--yellow) 100%) !important;
  border: 0 !important;
  color: white !important;
  font-weight: 700 !important;
  padding: 14px 32px !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 12px rgba(0,169,224,0.3) !important;
}

.stDownloadButton > button{
  background: var(--blue) !important;
  color: white !important;
  font-weight: 700 !important;
  border-radius: 12px !important;
  padding: 12px 28px !important;
}

/* Success message */
.stSuccess{
  background: #E8F5E9 !important;
  color: #2E7D32 !important;
  border-left: 4px solid #4CAF50 !important;
}

/* Slider */
.stSlider label, .stSlider p, div[data-testid="stSlider"] *{
  color: var(--text) !important;
}

/* Checkbox */
.stCheckbox label, .stCheckbox span{
  color: var(--text) !important;
  font-weight: 600 !important;
}

hr{
  border: 0;
  border-top: 2px solid var(--border);
  margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<img src="sukukLogo.png" style="width:65px; position:absolute; top:25px; left:30px;">', unsafe_allow_html=True)

# HERO SECTION 
st.markdown("""
<div class="hero-section">
  <h2 class="hero-title">AI Financial Document Analyzer</h2>
  <p class="hero-subtitle">Intelligent document processing powered by AI · Object detection · Visual analysis</p>
</div>
""", unsafe_allow_html=True)

# UPLOAD SECTION
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
c1, c2 = st.columns([0.35, 0.65])
with c1:
    st.markdown("**📁 Input Mode**")
    file_mode = st.radio("Choose", ["Single Image", "Full PDF"], horizontal=True, label_visibility="collapsed")
with c2:
    st.markdown("**📤 Upload Your Document**")
    uploaded = st.file_uploader("Image (jpg/png) or PDF", type=["jpg","jpeg","png","pdf"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

# MODEL
@st.cache_resource
def load_yolo():
    return YOLO(r"Modelling\\Financial_Page_Classification.v3-version3.yolov8\\best.pt")
yolo_model = load_yolo()

# HELPERS AND FUNCTIONS
POPPLER_PATH = r"C:\\poppler-25.07.0\\Library\\bin"

LABEL_COLORS = {
    "table": (0,169,224),
    "financial_sheets": (0,169,224),
    "notes": (255,184,28),
    "text": (100,100,100),
    "auditor_report": (255,184,28),
}

def draw_boxes(img_path, r):
    img = cv2.imread(img_path)
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        lbl = r.names[int(b.cls)]
        conf = float(b.conf)
        color = LABEL_COLORS.get(lbl, (100,100,100))
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 4)
        label_text = f"{lbl} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1-h-14), (x1+w+16, y1), color, -1)
        cv2.putText(img, label_text, (x1+8, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def classify_page_yolo(labels, confs):
    if not labels: 
        return "Other Page", 0.0
    
    # Convert to set of lowercase labels
    labels_set = {lbl.strip().lower() for lbl in labels}
    
    # Get max confidence for the classification
    lab2 = {}
    for l,c in zip(labels, confs):
        l = l.lower().strip()
        lab2[l] = max(lab2.get(l,0.0), float(c))
    
    # Classification logic
    if "auditor_report" in labels_set:
        return "Auditor Report Page", lab2.get("auditor_report", 0.0)
    
    if "financial_sheets" in labels_set or ("financial_sheets" in labels_set and "table" in labels_set):
        conf = max(lab2.get("financial_sheets", 0.0), lab2.get("table", 0.0))
        return "Financial Sheets Page", conf
    
    if "notes" in labels_set and "text" in labels_set:
        conf = min(lab2.get("notes", 0.0), lab2.get("text", 0.0))
        return "Text Notes Page", conf
    
    if "notes" in labels_set and "table" in labels_set:
        conf = min(lab2.get("notes", 0.0), lab2.get("table", 0.0))
        return "Tabular Notes Page", conf
    
    if "table" in labels_set:
        return "Financial Sheets Page", lab2.get("table", 0.0)
    
    return "Other Page", 0.0

def show_file_progress(filename, filesize, percent):
    """Show progress bar below upload section"""
    html = f"""
    <div class="progress-section">
      <div class="file-progress">
        <div class="file-info">
          <div class="file-icon">📄</div>
          <div class="file-details">
            <p class="file-name">{filename}</p>
            <p class="file-size">{filesize}</p>
          </div>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar-fill" style="width: {int(percent*100)}%"></div>
        </div>
        <span class="progress-text">{int(percent*100)}%</span>
      </div>
    </div>
    """
    return html

def read_image_or_pdf(file):
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, file.name)
    with open(p, "wb") as f:
        f.write(file.read())

    # If PDF, convert to images (internally)
    if file.name.lower().endswith(".pdf"):
        filesize = f"{file.size / 1024:.1f} KB" if file.size < 1024*1024 else f"{file.size / (1024*1024):.1f} MB"
        progress_placeholder = st.empty()
        
        pages = convert_from_path(p, dpi=200, fmt="png", poppler_path=POPPLER_PATH)
        total = len(pages)
        outs=[]
        
        for i,pg in enumerate(pages, start=1):
            percent = i / total
            progress_placeholder.markdown(show_file_progress(file.name, filesize, percent), unsafe_allow_html=True)
            ip = os.path.join(tmp, f"page_{i}.png")
            pg.save(ip, "PNG")
            outs.append(ip)
            time.sleep(0.02)
        
        progress_placeholder.markdown(show_file_progress(file.name, filesize, 1.0), unsafe_allow_html=True)
        st.success(f"✅ Successfully Uploaded {total} pages!")
        return outs
    return [p]

# MAIN FLOW
if uploaded:
    if file_mode == "Full PDF" and not uploaded.name.lower().endswith(".pdf"):
        st.error("⚠️ Please upload a PDF file for 'Full PDF' mode.")
        st.stop()

    imgs = read_image_or_pdf(uploaded)
    results_rows = []

    # Always run YOLO
    st.markdown('<div class="card"><h4>🔍 Detection Results</h4>', unsafe_allow_html=True)
    grid = []
    for pth in imgs:
        rs = yolo_model.predict(source=pth, save=False, show=False, conf=0.25, verbose=False)
        r = rs[0]
        labels = [r.names[int(b.cls)] for b in r.boxes]
        confs  = [float(b.conf) for b in r.boxes]
        page_cls, page_conf = classify_page_yolo(labels, confs)
        drawn = draw_boxes(pth, r)

        grid.append((drawn, page_cls, page_conf, labels))
        results_rows.append({"image": os.path.basename(pth), "classification": page_cls, "confidence": page_conf})

    for i in range(0, len(grid), 5):
        st.markdown('<div class="grid-5">', unsafe_allow_html=True)
        cols = st.columns(5)
        for col, (img, title, conf, labs) in zip(cols, grid[i:i+5]):
            col.markdown(f'<div class="thumb-title">{title}</div>', unsafe_allow_html=True)
            col.markdown(f'<div class="thumb-sub">Confidence: {int(conf*100)}%</div>', unsafe_allow_html=True)
            col.image(img, use_container_width=True)
            if labs:
                chips = "".join([f'<span class="chip">{l}</span>' for l in sorted(set(labs))])
                col.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Dynamic plots for PDF with Plotly
    if len(imgs) > 1 and results_rows:
        st.markdown('<div class="card"><h4>📊 Document Summary - Interactive Charts</h4>', unsafe_allow_html=True)
        df = pd.DataFrame(results_rows)
        cA, cB = st.columns(2)

        with cA:
            cnt = df["classification"].value_counts().sort_values(ascending=False)
            fig = px.bar(
                x=cnt.index, 
                y=cnt.values,
                labels={'x': 'Classification', 'y': 'Number of Pages'},
                title='Page Type Distribution',
                color=cnt.values,
                color_continuous_scale=['#00A9E0', '#FFB81C']
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='#FAFAFA',
                font=dict(color='#1A1A1A', size=12, family='Arial'),
                xaxis=dict(tickangle=-15),
                hoverlabel=dict(bgcolor="white", font_size=13)
            )
            fig.update_traces(marker=dict(line=dict(color='white', width=2)))
            st.plotly_chart(fig, use_container_width=True)

        with cB:
            fig2 = px.histogram(
                df, 
                x='confidence',
                nbins=10,
                title='Confidence Score Distribution',
                labels={'confidence': 'Confidence Score'},
                color_discrete_sequence=['#FFB81C']
            )
            fig2.update_layout(
                showlegend=False,
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='#FAFAFA',
                font=dict(color='#1A1A1A', size=12, family='Arial'),
                bargap=0.1,
                hoverlabel=dict(bgcolor="white", font_size=13)
            )
            fig2.update_traces(marker=dict(line=dict(color='white', width=2)))
            st.plotly_chart(fig2, use_container_width=True)

        st.download_button("⬇️ Download Results as CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="pdf_page_results.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)