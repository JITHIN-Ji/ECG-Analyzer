🫀 Bharat Cardio: ECG Digitizer & Analyzer
A Streamlit-based web application to analyze ECG reports from uploaded images or PDFs. It digitizes the ECG signal, calculates key cardiovascular metrics, and classifies the ECG (e.g., Normal, MI, Abnormal Heartbeat) using AI models.

🚀 Features
✅ Upload ECG in image (PNG/JPG) or PDF format

📄 Extract patient details (name, age, gender) from PDF

⚙️ Digitize Lead II ECG waveform from the image

📈 Compute key ECG parameters:

Heart Rate, RR Interval, Rhythm

QRS Duration, P-Wave & QRS Amplitudes

T-Wave Duration, QT/QTc Interval

ST & PR Segments

HRV Metrics: SDNN, RMSSD

🧠 Classify ECG with Google Gemini AI API

📦 Store results in SQLite database

📤 Download formatted PDF report

📦 Requirements
Install required packages using:

bash
Copy
Edit
pip install -r requirements.txt
Example packages:

streamlit

opencv-python

numpy

pandas

scipy

matplotlib

Pillow

PyMuPDF

fpdf

google-generativeai

📂 File Structure
bash
Copy
Edit
project/
│
├── app.py                      # Main Streamlit app
├── helper.py                  # Contains analyze_ecg_classification function
├── ecg_results.db             # SQLite database (auto-created)
├── requirements.txt
└── README.md                  # You're here
🧠 ECG Classification
ECG classification is handled by analyze_ecg_classification() using Google Gemini API. Detected classes may include:

✅ Normal

🚨 MI (Myocardial Infarction)

⚠️ History of MI

💓 Abnormal heartbeat

📤 Output
On-screen result: Table with ECG metrics

Database storage: ecg_reports SQLite table

PDF report: Downloadable summary with patient info and ECG parameters

🛠 How It Works
Upload a scanned ECG report (image or PDF)

Extract & preprocess the ECG signal from Lead II region

Estimate sampling rate using grid spacing

Digitize waveform to get pixel-level signal

Detect R-peaks, calculate bpm, intervals, amplitudes

Classify ECG via Gemini AI

Store and export data & report

🗃 Database Schema
Table: ecg_reports

Column	Type
id	INTEGER
name	TEXT
age	TEXT
gender	TEXT
heart_rate	REAL
rhythm	TEXT
rr_interval	REAL
ibi	REAL
qrs_duration	REAL
p_wave_amp	REAL
qrs_amp	REAL
t_wave_amp	REAL
qt_interval	REAL
qtc_interval	REAL
sdnn	REAL
rmssd	REAL
st_segment	REAL
pr_segment	REAL
classification	TEXT

📝 How to Run
bash
Copy
Edit
streamlit run app.py
