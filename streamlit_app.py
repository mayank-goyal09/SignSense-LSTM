import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_landmarks
import time

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="SignSense — Real-Time Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# INJECT GOOGLE FONTS + FULL CUSTOM CSS
# ==========================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    /* ── GLOBAL RESET ── */
    *, *::before, *::after { box-sizing: border-box; }
    
    .stApp {
        background: #07070a !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Kill default Streamlit header bar */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0c14 0%, #0a0a12 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    [data-testid="stSidebar"] * {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* ── HIDE STREAMLIT DEFAULTS ── */
    #MainMenu, footer, .stDeployButton { display: none !important; }
    
    /* ── ANIMATIONS ── */
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 15px rgba(0, 255, 136, 0.15), inset 0 0 15px rgba(0, 255, 136, 0.03); }
        50% { box-shadow: 0 0 30px rgba(0, 255, 136, 0.3), inset 0 0 30px rgba(0, 255, 136, 0.06); }
    }
    @keyframes pulseRing {
        0% { transform: scale(0.95); opacity: 1; }
        100% { transform: scale(1.6); opacity: 0; }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes floatUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes borderGradient {
        0% { border-color: rgba(0, 255, 136, 0.3); }
        33% { border-color: rgba(0, 170, 255, 0.3); }
        66% { border-color: rgba(170, 0, 255, 0.3); }
        100% { border-color: rgba(0, 255, 136, 0.3); }
    }
    @keyframes breathe {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    /* ── HERO BANNER ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(0,255,136,0.05) 0%, rgba(0,170,255,0.05) 50%, rgba(170,0,255,0.03) 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 35px 40px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        animation: floatUp 0.8s ease-out;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: -200%;
        width: 200%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.02), transparent);
        animation: shimmer 6s infinite;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        font-size: 2.6rem;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00ff88, #00aaff, #aa00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px 0;
    }
    .hero-subtitle {
        font-family: 'Outfit', sans-serif;
        font-weight: 300;
        font-size: 1.05rem;
        color: rgba(255,255,255,0.45);
        letter-spacing: 0.5px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0,255,136,0.1);
        border: 1px solid rgba(0,255,136,0.2);
        color: #00ff88;
        padding: 4px 14px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 15px;
        animation: breathe 3s infinite;
    }
    
    /* ── GLASS PANELS ── */
    .glass-panel {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        animation: floatUp 0.9s ease-out;
    }
    .glass-panel-accent {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        animation: floatUp 1s ease-out, borderGradient 6s linear infinite;
    }
    
    .panel-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin-bottom: 18px;
    }
    
    /* ── PROBABILITY DEBUG BARS ── */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    .prob-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.5);
        width: 80px;
        text-align: right;
        flex-shrink: 0;
    }
    .prob-track {
        flex: 1;
        background: rgba(255,255,255,0.05);
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }
    .prob-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.4);
        width: 50px;
        flex-shrink: 0;
    }
    .prob-fill-top {
        background: linear-gradient(90deg, #00ff88, #00aaff);
        box-shadow: 0 0 8px rgba(0,255,136,0.3);
    }
    .prob-fill-mid {
        background: linear-gradient(90deg, #ffaa00, #ff6600);
    }
    .prob-fill-low {
        background: rgba(255,255,255,0.15);
    }
    
    /* ── CAMERA FEED FRAME — style the Streamlit image element directly ── */
    [data-testid="stImage"] {
        border: 2px solid rgba(0,255,136,0.15);
        border-radius: 16px;
        overflow: hidden;
        animation: pulseGlow 4s infinite ease-in-out;
    }
    [data-testid="stImage"] img {
        border-radius: 14px;
        display: block;
    }
    
    /* ── LIVE INDICATOR ── */
    .live-dot-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
    }
    .live-dot {
        width: 10px; height: 10px;
        background: #00ff88;
        border-radius: 50%;
        position: relative;
    }
    .live-dot::after {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: #00ff88;
        border-radius: 50%;
        animation: pulseRing 1.5s infinite;
    }
    .live-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #00ff88;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* ── DETECTION CARD ── */
    .detection-card {
        background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,170,255,0.05));
        border: 1px solid rgba(0,255,136,0.15);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .detection-card::before {
        content: '';
        position: absolute;
        top: 0; left: -200%;
        width: 200%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,255,136,0.03), transparent);
        animation: shimmer 4s infinite;
    }
    .detection-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.4);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .detection-word {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        color: #ffffff;
        letter-spacing: -1px;
        margin: 10px 0;
        text-shadow: 0 0 40px rgba(0,255,136,0.3);
    }
    .detection-confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #00ff88;
        font-weight: 600;
    }
    
    /* ── CONFIDENCE BAR ── */
    .conf-bar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 50px;
        height: 8px;
        margin-top: 15px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #00ff88, #00aaff);
        transition: width 0.4s ease;
        box-shadow: 0 0 12px rgba(0,255,136,0.4);
    }
    
    /* ── HISTORY LOG ── */
    .history-entry {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 12px 16px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        margin-bottom: 8px;
        animation: floatUp 0.5s ease-out;
        transition: all 0.3s ease;
    }
    .history-entry:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(0,255,136,0.15);
        transform: translateX(4px);
    }
    .history-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.25);
        min-width: 70px;
    }
    .history-word {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #ffffff;
    }
    .history-pip {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #00ff88;
        flex-shrink: 0;
    }
    
    /* ── STAT CHIPS ── */
    .stat-row {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    .stat-chip {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-chip:hover {
        border-color: rgba(0,255,136,0.2);
        background: rgba(0,255,136,0.03);
    }
    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        color: #ffffff;
    }
    .stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        color: rgba(255,255,255,0.3);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }
    
    /* ── IDLE STATE ── */
    .idle-state {
        text-align: center;
        padding: 60px 30px;
        color: rgba(255,255,255,0.15);
    }
    .idle-icon {
        font-size: 4rem;
        margin-bottom: 15px;
        animation: breathe 3s infinite;
    }
    .idle-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 400;
        font-size: 1rem;
        color: rgba(255,255,255,0.2);
    }
    
    /* ── BUTTONS ── */
    .stButton > button {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.5px !important;
        border: none !important;
    }
    
    /* Start button */
    .start-btn button {
        background: linear-gradient(135deg, #00ff88, #00cc6a) !important;
        color: #070a07 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0,255,136,0.25) !important;
    }
    .start-btn button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0,255,136,0.4) !important;
    }
    
    /* Stop button */
    .stop-btn button {
        background: rgba(255, 60, 60, 0.1) !important;
        color: #ff3c3c !important;
        border: 1px solid rgba(255, 60, 60, 0.2) !important;
    }
    .stop-btn button:hover {
        background: rgba(255, 60, 60, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #00ff88 !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* ── SIDEBAR BRANDING ── */
    .sidebar-brand {
        text-align: center;
        padding: 20px 10px 30px 10px;
    }
    .sidebar-logo {
        font-size: 3.5rem;
        margin-bottom: 10px;
    }
    .sidebar-name {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.6rem;
        background: linear-gradient(135deg, #00ff88, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sidebar-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: rgba(255,255,255,0.25);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .sidebar-divider {
        width: 40px;
        height: 2px;
        background: linear-gradient(90deg, #00ff88, transparent);
        margin: 25px auto;
        border-radius: 2px;
    }
    .sidebar-section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: rgba(255,255,255,0.25);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    /* ── VOCABULARY CHIP ── */
    .vocab-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .vocab-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 6px 14px;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.6);
        transition: all 0.3s ease;
    }
    .vocab-chip:hover {
        border-color: rgba(0,255,136,0.3);
        color: #00ff88;
        background: rgba(0,255,136,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD RESOURCES (Cached)
# ==========================================
@st.cache_resource
def get_model():
    return load_model('action.h5')

model = get_model()
actions = np.array(['hello', 'how_you', 'hi', 'whats_up', 'you_good'])

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'running' not in st.session_state:
    st.session_state.running = False

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🤟</div>
        <div class="sidebar-name">SignSense</div>
        <div class="sidebar-tag">LSTM Neural Engine</div>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">⚙️ Engine Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Confidence Threshold", 0.50, 1.0, 0.80, 0.05,
                          help="Minimum confidence required to register a gesture.")
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">📖 Vocabulary</div>', unsafe_allow_html=True)
    
    vocab_html = '<div class="vocab-grid">'
    for act in actions:
        vocab_html += f'<div class="vocab-chip">{act.replace("_", " ").title()}</div>'
    vocab_html += '</div>'
    st.markdown(vocab_html, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">🔬 Debug</div>', unsafe_allow_html=True)
    show_debug = st.toggle("Show Probability Bars", value=True,
                           help="Show real-time probability distribution for all gestures")
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    if st.button("🗑️  Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_detections = 0
        st.rerun()
    
    st.markdown("""
    <div style="position: fixed; bottom: 15px; left: 15px; font-family: 'JetBrains Mono', monospace;
                font-size: 0.6rem; color: rgba(255,255,255,0.15); letter-spacing: 1px;">
        v1.0 · LSTM · MediaPipe
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# HERO BANNER
# ==========================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">● LSTM Powered</div>
    <div class="hero-title">Real-Time Sign Language Translator</div>
    <div class="hero-subtitle">
        Translating hand gestures into words using deep learning — powered by MediaPipe landmarks and an LSTM neural network.
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN LAYOUT
# ==========================================
col_camera, col_insights = st.columns([3, 2], gap="large")

with col_camera:
    # Camera Feed Panel
    st.markdown("""
    <div class="glass-panel">
        <div class="panel-title">📹 Camera Feed</div>
    </div>
    """, unsafe_allow_html=True)
    
    frame_placeholder = st.empty()
    
    # Show idle state if not running
    if not st.session_state.running:
        frame_placeholder.markdown("""
        <div class="glass-panel" style="margin-top: -20px;">
            <div class="idle-state">
                <div class="idle-icon">✋</div>
                <div class="idle-text">Press "Start Engine" to begin real-time detection</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Buttons row
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        start_btn = st.button("▶  Start Engine", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with btn_col2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_btn = st.button("■  Stop Engine", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_insights:
    # Live Detection Card
    detection_placeholder = st.empty()
    detection_placeholder.markdown("""
    <div class="glass-panel-accent">
        <div class="panel-title">🎯 Current Detection</div>
        <div class="detection-card">
            <div class="detection-label">Awaiting Input</div>
            <div class="detection-word" style="color: rgba(255,255,255,0.1);">—</div>
            <div class="detection-confidence" style="color: rgba(255,255,255,0.1);">0.0%</div>
            <div class="conf-bar-track"><div class="conf-bar-fill" style="width: 0%;"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats
    stats_placeholder = st.empty()
    stats_placeholder.markdown(f"""
    <div class="stat-row">
        <div class="stat-chip">
            <div class="stat-value">{st.session_state.total_detections}</div>
            <div class="stat-label">Detections</div>
        </div>
        <div class="stat-chip">
            <div class="stat-value">{len(st.session_state.history)}</div>
            <div class="stat-label">Unique</div>
        </div>
        <div class="stat-chip">
            <div class="stat-value">{int(threshold * 100)}%</div>
            <div class="stat-label">Threshold</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability Debug Bars
    if show_debug:
        prob_placeholder = st.empty()
        prob_placeholder.markdown("""
        <div class="glass-panel">
            <div class="panel-title">📊 Class Probabilities</div>
            <div style="text-align: center; color: rgba(255,255,255,0.15); padding: 10px;
                        font-family: 'Outfit'; font-size: 0.85rem;">Waiting for engine...</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        prob_placeholder = None
    
    # Translation History
    st.markdown("""
    <div class="glass-panel">
        <div class="panel-title">🕒 Translation History</div>
    </div>
    """, unsafe_allow_html=True)
    history_placeholder = st.empty()
    
    if not st.session_state.history:
        history_placeholder.markdown("""
        <div style="text-align: center; padding: 30px; color: rgba(255,255,255,0.12);
                    font-family: 'Outfit', sans-serif; font-size: 0.9rem;">
            No translations yet...
        </div>
        """, unsafe_allow_html=True)
    else:
        hist_html = ""
        for item in st.session_state.history[:8]:
            hist_html += f"""
            <div class="history-entry">
                <div class="history-pip"></div>
                <div class="history-time">{item['time']}</div>
                <div class="history-word">{item['word']}</div>
            </div>
            """
        history_placeholder.markdown(hist_html, unsafe_allow_html=True)

# ==========================================
# ENGINE LOOP (Flicker-Free)
# ==========================================
if start_btn:
    st.session_state.running = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access your camera. Please check permissions.")
        st.stop()
    
    sequence = []
    last_action = ""
    last_detection_update = ""   # Track what's currently shown in detection card
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            keypoints = extract_landmarks(frame)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            # Only run prediction + update side panels every 3rd frame
            # but ALWAYS update the camera feed for smooth video
            run_prediction = (len(sequence) == 30) and (frame_count % 3 == 0)
            
            if run_prediction:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                action = actions[np.argmax(res)]
                confidence = float(res[np.argmax(res)])
                conf_pct = confidence * 100
                
                # Update probability debug bars
                if prob_placeholder is not None:
                    prob_bars_html = '<div class="glass-panel"><div class="panel-title">📊 Class Probabilities</div>'
                    sorted_indices = np.argsort(res)[::-1]
                    for rank, idx in enumerate(sorted_indices):
                        pct = float(res[idx]) * 100
                        name = actions[idx].replace('_', ' ').title()
                        if rank == 0:
                            fill_class = 'prob-fill-top'
                        elif pct > 10:
                            fill_class = 'prob-fill-mid'
                        else:
                            fill_class = 'prob-fill-low'
                        prob_bars_html += f'''
                        <div class="prob-row">
                            <div class="prob-label">{name}</div>
                            <div class="prob-track"><div class="prob-fill {fill_class}" style="width: {pct}%;"></div></div>
                            <div class="prob-value">{pct:.1f}%</div>
                        </div>'''
                    prob_bars_html += '</div>'
                    prob_placeholder.markdown(prob_bars_html, unsafe_allow_html=True)
                
                # Build a key to check if detection card actually needs updating
                detection_key = f"{action}_{confidence > threshold}"
                
                if confidence > threshold:
                    # Only re-render detection card if the gesture changed
                    if detection_key != last_detection_update:
                        last_detection_update = detection_key
                        detection_placeholder.markdown(f"""
                        <div class="glass-panel-accent">
                            <div class="panel-title">🎯 Current Detection</div>
                            <div class="live-dot-container">
                                <div class="live-dot"></div>
                                <div class="live-label">Live</div>
                            </div>
                            <div class="detection-card">
                                <div class="detection-label">Detected Gesture</div>
                                <div class="detection-word">{action.replace('_', ' ').upper()}</div>
                                <div class="detection-confidence">{conf_pct:.1f}% confidence</div>
                                <div class="conf-bar-track"><div class="conf-bar-fill" style="width: {conf_pct}%;"></div></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add to history only when gesture transitions
                    if action != last_action:
                        st.session_state.total_detections += 1
                        st.session_state.history.insert(0, {
                            'word': action.replace('_', ' ').upper(),
                            'time': time.strftime("%H:%M:%S")
                        })
                        st.session_state.history = st.session_state.history[:15]
                        last_action = action
                        
                        # Refresh history
                        hist_html = ""
                        for item in st.session_state.history[:8]:
                            hist_html += f"""
                            <div class="history-entry">
                                <div class="history-pip"></div>
                                <div class="history-time">{item['time']}</div>
                                <div class="history-word">{item['word']}</div>
                            </div>
                            """
                        history_placeholder.markdown(hist_html, unsafe_allow_html=True)
                        
                        # Refresh stats
                        stats_placeholder.markdown(f"""
                        <div class="stat-row">
                            <div class="stat-chip">
                                <div class="stat-value">{st.session_state.total_detections}</div>
                                <div class="stat-label">Detections</div>
                            </div>
                            <div class="stat-chip">
                                <div class="stat-value">{len(set(i['word'] for i in st.session_state.history))}</div>
                                <div class="stat-label">Unique</div>
                            </div>
                            <div class="stat-chip">
                                <div class="stat-value">{int(threshold * 100)}%</div>
                                <div class="stat-label">Threshold</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Only update "scanning" state if it changed
                    if detection_key != last_detection_update:
                        last_detection_update = detection_key
                        detection_placeholder.markdown(f"""
                        <div class="glass-panel-accent">
                            <div class="panel-title">🎯 Current Detection</div>
                            <div class="detection-card">
                                <div class="detection-label">Scanning...</div>
                                <div class="detection-word" style="font-size: 1.8rem; color: rgba(255,255,255,0.2);">{action.replace('_', ' ').upper()}</div>
                                <div class="detection-confidence" style="color: rgba(255,170,0,0.7);">{conf_pct:.1f}% — below threshold</div>
                                <div class="conf-bar-track"><div class="conf-bar-fill" style="width: {conf_pct}%; background: linear-gradient(90deg, #ffaa00, #ff6600);"></div></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # ALWAYS update camera — this is the ONLY thing that updates every frame
            # Use .image() directly on the placeholder — no HTML wrapper!
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            frame_count += 1
            
            # Throttle to ~30 fps to avoid flooding the websocket
            time.sleep(0.033)
            
        except Exception as e:
            st.error(f"Detection error: {e}")
            break
        
        if stop_btn:
            break
    
    cap.release()
    st.session_state.running = False
    st.toast("✅ Engine stopped successfully.", icon="🛑")
