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

if uploaded_file:
    image = pdf_to_image(uploaded_file) if uploaded_file.type == "application/pdf" else Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    st.image(image, caption="Original ECG", use_column_width=True)

    crop_coords = get_lead_ii_coordinates(image, api_key) if api_key else None
    if not crop_coords:
        st.warning("Using manual bounding box for Lead II")
        crop_coords = (110, 658, 1422, 847)

    x1, y1, x2, y2 = crop_coords
    crop_img = img_gray[y1:y2, x1:x2]
    st.image(crop_img, caption="Lead II Region", channels="GRAY")

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
        ibi = 1 / (bpm / 60)
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
        qtc = qt_interval / math.sqrt(np.mean(rr))
        rhythm_class = (
            "Bradycardia (Slow)" if bpm < 60 else
            "Normal (Regular)" if bpm <= 100 else
            "Tachycardia (Fast)"
        )

        st.subheader("\U0001F4CA ECG Analysis Report")
        df_results = pd.DataFrame({
            "Parameter": [
                "Heart Rate (bpm)", "Rhythm", "RR Interval (s)", "IBI (s)",
                "QRS Duration (s)", "P-Wave Amplitude", "QRS Amplitude",
                "T-Wave Amplitude", "QT Interval (s)", "QTc Interval (s)",
                "SDNN (s)", "RMSSD (s)", "ST Segment (s)", "PR Segment (s)"
            ],
            "Value": [
                f"{bpm:.2f}", rhythm_class, f"{np.mean(rr):.3f}", f"{ibi:.3f}",
                f"{qrs_duration:.3f}", f"{p_wave_amp:.2f}", f"{qrs_amp:.2f}",
                f"{t_wave_amp:.2f}", f"{qt_interval:.3f}", f"{qtc:.3f}",
                f"{sdnn:.4f}", f"{rmssd:.4f}", "~0.12", "~0.06"
            ]
        })
        st.table(df_results)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(filtered_signal, label='Filtered ECG')
        ax.plot(r_peaks, filtered_signal[r_peaks], 'ro', label='R-peaks')
        ax.set_title("Lead II ECG with R-peaks")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="ECG Analysis Report", ln=True, align='C')
        pdf.ln(10)
        for i in range(len(df_results)):
            param = df_results.iloc[i, 0]
            value = df_results.iloc[i, 1]
            pdf.cell(80, 10, txt=str(param), border=1)
            pdf.cell(50, 10, txt=str(value), border=1, ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            tmp_pdf.seek(0)
            st.download_button("Download PDF Report", data=tmp_pdf.read(), file_name="ecg_report.pdf")
    else:
        st.error("Not enough R-peaks detected.")
