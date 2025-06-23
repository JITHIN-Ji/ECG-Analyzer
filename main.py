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
import google.generativeai as genai
from fpdf import FPDF
import tempfile
import sqlite3
import os
from datetime import datetime

DB_PATH = "ecg_results.db"
if not os.path.exists(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS ecg_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, age TEXT, gender TEXT,
            heart_rate REAL, rhythm TEXT, rr_interval REAL, ibi REAL,
            qrs_duration REAL, p_wave_amp REAL, qrs_amp REAL, t_wave_amp REAL,
            qt_interval REAL, qtc_interval REAL, sdnn REAL, rmssd REAL,
            st_segment REAL, pr_segment REAL

        )''')



st.set_page_config(page_title="ECG Digitizer & Analyzer", layout="wide")
st.title("🫀 ECG Digitizer & Analyzer")

# === Google Gemini API ===
api_key = "YOUR_API_KEY_HERE"  # Replace with your Gemini API key
if api_key:
    genai.configure(api_key=api_key)





# === Convert PDF to image ===
def pdf_to_image(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf_document[0]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        pdf_document.close()
        return image
    except Exception as e:
        st.error(f"PDF conversion error: {str(e)}")
        return None

# === Get Lead II using Gemini ===
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
        if match:
            coords = json.loads(match.group())
            return coords.get('x1'), coords.get('y1'), coords.get('x2'), coords.get('y2')
    except:
        pass
    return None

# === Estimate mm per pixel ===
# ... [existing imports remain unchanged]

# === Estimate pixels per mm instead of mm per pixel ===
def estimate_pixels_per_mm(gray_img):
    edges = cv2.Canny(gray_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=20, maxLineGap=2)
    x_positions = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 2:  # vertical line
                x_positions.append(x1)

    x_positions = sorted(set(x_positions))
    pixel_diffs = np.diff(x_positions)

    if len(pixel_diffs) == 0:
        
        return 10.0  # fallback

    return np.median(pixel_diffs)

# === File Upload ===
uploaded_file = st.file_uploader("Upload ECG Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

def extract_patient_info_from_pdf(pdf_file):
    pdf_file.seek(0)
    text = ""
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except:
        return "Unknown", "N/A", "N/A"

    name_match = re.search(r'Patient Name:\s*(.+)', text)
    age_gender_match = re.search(r'Age\s*/\s*Gender:\s*(\d+)\s*Y\s*/?\s*(Male|Female)', text, re.IGNORECASE)

    name = name_match.group(1).strip() if name_match else "Unknown"
    age = age_gender_match.group(1) if age_gender_match else "N/A"
    gender = age_gender_match.group(2) if age_gender_match else "N/A"
    return name, age, gender





def extract_heart_rate_from_pdf(pdf_io):
    pdf_io.seek(0)
    text = ""
    try:
        with fitz.open(stream=pdf_io, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except:
        return None

    # Look for formats like: AR: 72bpm
    match = re.search(r'AR\s*[:\-]?\s*(\d+)\s*bpm', text, re.IGNORECASE)
    return int(match.group(1)) if match else None






if uploaded_file:
    # Read the uploaded file into memory once
    file_bytes = uploaded_file.read()

    # Convert to image (if PDF)
    image = pdf_to_image(io.BytesIO(file_bytes)) if uploaded_file.type == "application/pdf" else Image.open(uploaded_file)

    # Extract name/age/gender
    name, age, gender = extract_patient_info_from_pdf(io.BytesIO(file_bytes)) if uploaded_file.type == "application/pdf" else ("Unknown", "N/A", "N/A")

    st.subheader("👤 Patient Details")
    st.markdown(f"**Name:** {name}  \n**Age:** {age}  \n**Gender:** {gender}")

    # Check if user already exists
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT * FROM ecg_reports WHERE name = ? AND age = ?",
            (name, age)
        )
        previous_entries = cursor.fetchall()

    if previous_entries:
        st.success("✅ Report already exists. ")
        prev_df = pd.DataFrame(previous_entries, columns=[
            "ID", "Name", "Age", "Gender", "Heart Rate", "Rhythm", "RR Interval", "IBI",
            "QRS Duration", "P-Wave Amplitude", "QRS Amplitude", "T-Wave Duration",
            "QT Interval", "QTc Interval", "SDNN", "RMSSD", "ST Segment", "PR Segment"
        ])
        st.subheader("📋 Stored ECG Report")
        st.dataframe(prev_df)
        st.stop()  # ⛔ stop further code execution

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    st.image(image, caption="Original ECG", use_column_width=True)

    crop_coords = get_lead_ii_coordinates(image, api_key) if api_key else None
    if not crop_coords:
        st.warning("Using manual bounding box for Lead II")
        crop_coords = (110, 658, 1422, 847)

    x1, y1, x2, y2 = crop_coords
    crop_img = img_gray[y1:y2, x1:x2]
    # st.image(crop_img, caption="Lead II Region", channels="GRAY")

    # Estimate FS using real paper speed (25mm/s = 40 ms/mm)
    pixels_per_mm = estimate_pixels_per_mm(crop_img)
    mm_per_pixel = 1 / pixels_per_mm
    ms_per_mm = 40
    ms_per_sample = mm_per_pixel * ms_per_mm
    FS = 1000 / ms_per_sample

    # ==== Everything else below remains unchanged ====

    # Preprocessing
    ret, bin_img = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    noisy = np.float32(bin_img) + np.random.randn(*bin_img.shape) * 10
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(noisy, None, 3, 7, 21)
    

    for i in range(5, denoised.shape[0] - 5):
        for j in range(5, denoised.shape[1] - 5):
            if denoised[i, j] == 0:
                window = denoised[i - 5:i + 5, j - 5:j + 5]
                if np.sum(window == 0) < 5:
                    denoised[i, j] = 255

    signal = []
    for col in range(denoised.shape[1]):
        col_vals = denoised[:, col]
        black_pixels = np.where(col_vals == 0)[0]
        signal.append(black_pixels[0] if len(black_pixels) else signal[-1] if signal else 0)

    def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=FS, order=2):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return filtfilt(b, a, data)

    filtered_signal = bandpass_filter(signal)
    threshold = np.mean(filtered_signal) + 0.5 * np.std(filtered_signal)
    r_peaks, _ = find_peaks(filtered_signal, height=threshold, distance=int(0.6 * FS))

    if len(r_peaks) > 1:
        rr = np.diff(r_peaks) / FS
        bpm = 60 / np.mean(rr)
        # === Compare PDF Heart Rate with Extracted Heart Rate ===
        pdf_stream = io.BytesIO(file_bytes)
        pdf_hr = extract_heart_rate_from_pdf(pdf_stream)

        # One single validation message
        if pdf_hr is None or abs(pdf_hr - bpm) > 7:
            st.error("❌ Unable To Read File! Please re-upload.")
            st.stop()

        ibi = 1 / (bpm / 60)
        sdnn = np.std(rr)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
        qrs_durations = [(np.argmax(np.diff(filtered_signal[max(r-30,0):min(r+30,len(filtered_signal))]) < -5) +
                          np.argmax(np.diff(filtered_signal[max(r-30,0):min(r+30,len(filtered_signal))][30:]) > 5)) / FS
                         for r in r_peaks]
        qrs_duration = np.mean(qrs_durations)
        # Amplitude conversion from pixels to mV
        mv_per_mm = 0.1  # ECG paper scale
        mv_per_pixel = mv_per_mm / pixels_per_mm

        

        p_wave_amp = np.ptp(filtered_signal[:50]) * mv_per_pixel
        qrs_amp = np.ptp(filtered_signal) * mv_per_pixel
# Assumed ECG paper scale: 10 mm/mV, and ~1 pixel per mm
    # --- Estimate PR Segment ---
        estimated_p_duration = 0.08  # 80 ms
        pr_segments = []

        for r in r_peaks:
            p_end = r - int(0.12 * FS)  # assume P-wave ends ~120ms before R
            qrs_start = r - np.argmax(np.diff(filtered_signal[max(r-30, 0):r]) < -0.5)
            if qrs_start > p_end:
                pr_segments.append((qrs_start - p_end) / FS)

        pr_segment = np.mean(pr_segments) if pr_segments else 0.06  # fallback

        # --- Estimate ST Segment ---
        st_segments = []

        for r in r_peaks:
            qrs_end = r + np.argmax(np.diff(filtered_signal[r:min(r+30, len(filtered_signal))]) > 0.5)
            t_region = filtered_signal[qrs_end + 5: qrs_end + int(0.4 * FS)]
            t_thresh = 0.2 * np.max(t_region) if len(t_region) else 0
            t_above = np.where(t_region > t_thresh)[0]
            if len(t_above) > 0:
                t_start = qrs_end + 5 + t_above[0]
                st_segments.append((t_start - qrs_end) / FS)

        st_segment = np.mean(st_segments) if st_segments else 0.08
        # === Classify ST Segment ===
        # === Classify ST Segment ===
        # --- Estimate PR Interval ---
        # pr_intervals = []

        # for r in r_peaks:
        #     p_start = r - int(0.16 * FS)  # assume P wave starts ~160ms before R
        #     qrs_start = r - int(0.10 * FS)  # assume QRS starts ~100ms before R
        #     if 0 < p_start < qrs_start:
        #         pr_intervals.append((qrs_start - p_start) / FS)

        # pr_interval = np.mean(pr_intervals) if pr_intervals else 0.14


        



                

        # Estimate T-wave duration in the last portion of the signal
        t_wave_region = filtered_signal[-int(FS):]  # Last 1 second window

        # Normalize for thresholding
        t_wave_region = t_wave_region - np.mean(t_wave_region)
        t_wave_threshold = 0.2 * np.max(t_wave_region)  # 20% of peak height

        # Detect where the T-wave starts and ends
        above_thresh = np.where(t_wave_region > t_wave_threshold)[0]

        if len(above_thresh) > 0:
            t_start = above_thresh[0]
            t_end = above_thresh[-1]
            t_wave_duration = (t_end - t_start) / FS  # duration in seconds
        else:
            t_wave_duration = 0.0

        qt_interval = 0.4
        qtc = qt_interval / math.sqrt(np.mean(rr))
        rhythm_class = (
            "Bradycardia (Slow)" if bpm < 60 else
            "Normal (Regular)" if bpm <= 100 else
            "Tachycardia (Fast)"
        )

        # === Compare PDF Heart Rate with Extracted Heart Rate ===
        


        df_results = pd.DataFrame({
        "Parameter": [
            "Heart Rate (bpm)", "Rhythm", "RR Interval (s)", "IBI (s)",
            "QRS Duration (s)", "P-Wave Amplitude", "QRS Amplitude",
            "T-Wave Duration (s)", "QT Interval (s)", "QTc Interval (s)",
            "SDNN (s)", "RMSSD (s)", "ST Segment (s)", "PR Segment (s)"
        ],
        "Value": [
            f"{bpm:.2f}", rhythm_class, f"{np.mean(rr):.3f}", f"{ibi:.3f}",
            f"{qrs_duration:.3f}", f"{p_wave_amp:.2f}", f"{qrs_amp:.2f}",
            f"{t_wave_duration:.3f}", f"{qt_interval:.3f}", f"{qtc:.3f}",
            f"{sdnn:.4f}", f"{rmssd:.4f}", f"{st_segment:.3f}", f"{pr_segment:.3f}"

        ]
    })

# 👇 Show styled table
        st.subheader("📋 ECG Parameters for This Upload")
        st.dataframe(df_results.style.format(precision=3))



        

        

        # fig, ax = plt.subplots(figsize=(12, 4))
        # ax.plot(filtered_signal, label='Filtered ECG')
        # ax.plot(r_peaks, filtered_signal[r_peaks], 'ro', label='R-peaks')
        # ax.set_title("Lead II ECG with R-peaks")
        # ax.legend()
        # ax.grid(True)
        # st.pyplot(fig)

                # ✅ SAVE TO DATABASE
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                INSERT INTO ecg_reports (
                    name, age, gender, heart_rate, rhythm, rr_interval, ibi,
                    qrs_duration, p_wave_amp, qrs_amp, t_wave_amp,
                    qt_interval, qtc_interval, sdnn, rmssd,
                    st_segment, pr_segment

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, age, gender, bpm, rhythm_class, np.mean(rr), ibi,
                qrs_duration, p_wave_amp, qrs_amp, t_wave_duration,
                qt_interval, qtc, sdnn, rmssd, st_segment, pr_segment
            ))

        st.success("✅ ECG analysis has been saved  successfully!")


        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # === Title ===
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt="ECG Analysis Report", ln=True, align='C')
        pdf.ln(10)

        # === Patient Info ===
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, txt=f"Name: {name}", ln=True)
        pdf.cell(0, 10, txt=f"Age: {age}     Gender: {gender}", ln=True)
        pdf.cell(0, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(8)

        # === Table Header ===
        pdf.set_fill_color(200, 220, 255)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(90, 10, "Parameter", border=1, fill=True)
        pdf.cell(90, 10, "Value", border=1, ln=True, fill=True)

        # === Table Rows ===
        pdf.set_font("Arial", '', 12)
        for i in range(len(df_results)):
            param = df_results.iloc[i, 0]
            value = df_results.iloc[i, 1]
            pdf.cell(90, 10, str(param), border=1)
            pdf.cell(90, 10, str(value), border=1, ln=True)

        # === Save & Download ===
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            tmp_pdf.seek(0)
            st.download_button("📄 Download ECG Report", data=tmp_pdf.read(), file_name="ecg_report.pdf")
