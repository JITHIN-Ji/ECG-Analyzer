import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import butter, filtfilt, find_peaks
import math
import fitz  # PyMuPDF
import io
import base64
import json
import re
import sqlite3
from fpdf import FPDF
import tempfile
import google.generativeai as genai
import os

st.set_page_config(page_title="ECG Digitizer & Analyzer", layout="wide")
st.title("🫀 ECG Digitizer & Analyzer")

api_key = "YOUR_API_KEY_HERE"
if api_key:
    genai.configure(api_key=api_key)

@st.cache_resource
def get_connection():
    return sqlite3.connect("ecg_data.db", timeout=10, check_same_thread=False)

def init_db(conn):
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ecg_records (
                name TEXT,
                age TEXT,
                gender TEXT,
                heart_rate REAL,
                rhythm TEXT,
                rr_interval REAL,
                ibi REAL,
                qrs_duration REAL,
                p_wave_mv REAL,
                qrs_mv REAL,
                t_wave_mv REAL,
                qt_interval REAL,
                qtc_interval REAL,
                sdnn REAL,
                rmssd REAL,
                st_segment TEXT,
                pr_segment TEXT,
                UNIQUE(name, age)
            )
        """)

def pdf_to_image(pdf_file):
    pdf_bytes = pdf_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = pdf_document[0]
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    pdf_document.close()
    return image

def extract_patient_details(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = doc[0].get_text()
        name_match = re.search(r"Patient Name[:\-]?\s*(.+)", text, re.IGNORECASE)
        age_gender_match = re.search(r"Age\s*/\s*Gender[:\-]?\s*(\d+)[Yy]\s*/\s*(\w+)", text)
        name = name_match.group(1).strip() if name_match else "Unknown"
        if age_gender_match:
            age = age_gender_match.group(1).strip()
            gender = age_gender_match.group(2).strip().capitalize()
        else:
            age = "Unknown"
            gender = "Unknown"
        return name, age, gender
    except:
        return "Unknown", "Unknown", "Unknown"

def get_lead_ii_coordinates(image, api_key):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        prompt = """
        Provide JSON bounding box for Lead II ECG as:
        {"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}
        """
        image_part = {"mime_type": "image/png", "data": img_base64}
        response = model.generate_content([prompt, image_part])
        match = re.search(r'\{[^}]*\}', response.text)
        return json.loads(match.group()).values()
    except:
        return None

def estimate_pixels_per_mm(gray_img):
    edges = cv2.Canny(gray_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=20, maxLineGap=2)
    x_positions = [line[0][0] for line in lines if abs(line[0][0] - line[0][2]) < 2] if lines is not None else []
    pixel_diffs = np.diff(sorted(set(x_positions)))
    return np.median(pixel_diffs) if len(pixel_diffs) > 0 else 10.0

uploaded_file = st.file_uploader("Upload ECG Image or PDF", type=["png", "jpg", "jpeg", "pdf"])
if uploaded_file:
    conn = get_connection()
    init_db(conn)

    if uploaded_file.type == "application/pdf":
        name, age, gender = extract_patient_details(uploaded_file)
        uploaded_file.seek(0)
        image = pdf_to_image(uploaded_file)
    else:
        name, age, gender = "Unknown", "Unknown", "Unknown"
        image = Image.open(uploaded_file)

    st.markdown(f"**👤 Patient Name:** {name} &nbsp;&nbsp; **🎂 Age:** {age} &nbsp;&nbsp; **Gender:** {gender}")

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    st.image(image, caption="Original ECG", use_column_width=True)

    crop_coords = get_lead_ii_coordinates(image, api_key) if api_key else None
    if not crop_coords:
        st.warning("Using default coordinates for Lead II region.")
        crop_coords = (110, 658, 1422, 847)

    x1, y1, x2, y2 = map(int, crop_coords)
    crop_img = img_gray[y1:y2, x1:x2]
    st.image(crop_img, caption="Lead II Region", channels="GRAY")

    pixels_per_mm = estimate_pixels_per_mm(crop_img)
    mm_per_pixel = 1 / pixels_per_mm
    ms_per_sample = mm_per_pixel * 40
    FS = 1000 / ms_per_sample

    cur = conn.execute("SELECT * FROM ecg_records WHERE name=? AND age=?", (name, age))
    existing = cur.fetchone()

    if existing:
        st.info("Submitted Sucessfully")
        (name, age, gender, bpm, rhythm_class, rr_interval, ibi, qrs_duration,
         p_wave_mv, qrs_mv, t_wave_mv, qt_interval, qtc, sdnn, rmssd, st_seg, pr_seg) = existing
    else:
        _, bin_img = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        noisy = np.clip(np.float32(bin_img) + np.random.randn(*bin_img.shape) * 10, 0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(noisy, None, 3, 7, 21)
        signal = [np.where(denoised[:, col] == 0)[0][0] if len(np.where(denoised[:, col] == 0)[0]) else 0 for col in range(denoised.shape[1])]

        def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS, order=2):
            nyq = 0.5 * fs
            b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
            return filtfilt(b, a, data)

        filtered_signal = bandpass_filter(signal)
        threshold = np.mean(filtered_signal) + 0.5 * np.std(filtered_signal)
        r_peaks, _ = find_peaks(filtered_signal, height=threshold, distance=int(0.6 * FS))

        if len(r_peaks) <= 1:
            st.error("Not enough R-peaks detected.")
            st.stop()

        rr = np.diff(r_peaks) / FS
        rr_interval = np.mean(rr)
        bpm = 60 / rr_interval
        ibi = 60 / bpm
        sdnn = np.std(rr)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
        qrs_durations = [(np.argmax(np.diff(filtered_signal[max(r-30,0):min(r+30,len(filtered_signal))]) < -5) +
                          np.argmax(np.diff(filtered_signal[max(r-30,0):min(r+30,len(filtered_signal))][30:]) > 5)) / FS
                         for r in r_peaks]
        qrs_duration = np.mean(qrs_durations)
        p_wave_amp = np.ptp(filtered_signal[:50])
        qrs_amp = np.ptp(filtered_signal)
        t_wave_amp = np.ptp(filtered_signal[-100:])
        qt_interval = 0.4
        qtc = qt_interval / math.sqrt(rr_interval)
        rhythm_class = "Bradycardia (Slow)" if bpm < 60 else "Normal (Regular)" if bpm <= 100 else "Tachycardia (Fast)"

        p_wave_mv = (p_wave_amp / pixels_per_mm) / 10
        qrs_mv = (qrs_amp / pixels_per_mm) / 10
        t_wave_mv = (t_wave_amp / pixels_per_mm) / 10
        st_seg = "~0.12"
        pr_seg = "~0.06"

        with conn:
            conn.execute("""INSERT INTO ecg_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (name, age, gender, bpm, rhythm_class, rr_interval, ibi, qrs_duration,
                          p_wave_mv, qrs_mv, t_wave_mv, qt_interval, qtc, sdnn, rmssd, st_seg, pr_seg))

    st.subheader("📊 ECG Analysis Report")
    df_results = pd.DataFrame({
        "Parameter": [
            "Heart Rate (bpm)", "Rhythm", "RR Interval (s)", "IBI (s)",
            "QRS Duration (s)", "P-Wave Amplitude (mV)", "QRS Amplitude (mV)",
            "T-Wave Amplitude (mV)", "QT Interval (s)", "QTc Interval (s)",
            "SDNN (s)", "RMSSD (s)", "ST Segment (s)", "PR Segment (s)"
        ],
        "Value": [
            f"{bpm:.2f}", rhythm_class, f"{rr_interval:.3f}", f"{ibi:.3f}",
            f"{qrs_duration:.3f}", f"{p_wave_mv:.2f}", f"{qrs_mv:.2f}",
            f"{t_wave_mv:.2f}", f"{qt_interval:.3f}", f"{qtc:.3f}",
            f"{sdnn:.4f}", f"{rmssd:.4f}", st_seg, pr_seg
        ]
    })
    st.table(df_results)

    # fig, ax = plt.subplots(figsize=(12, 4))
    # if 'filtered_signal' in locals():
    #     ax.plot(filtered_signal, label='Filtered ECG')
    #     ax.plot(r_peaks, filtered_signal[r_peaks], 'ro', label='R-peaks')
    # ax.set_title("Lead II ECG with R-peaks")
    # ax.legend()
    # ax.grid(True)
    # st.pyplot(fig)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="ECG Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Name: {name}   Age: {age}   Gender: {gender}", ln=True)
    pdf.ln(5)
    for i in range(len(df_results)):
        param = df_results.iloc[i, 0]
        value = df_results.iloc[i, 1]
        pdf.cell(80, 10, txt=str(param), border=1)
        pdf.cell(50, 10, txt=str(value), border=1, ln=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        tmp_pdf.seek(0)
        st.download_button("Download PDF Report", data=tmp_pdf.read(), file_name="ecg_report.pdf")
