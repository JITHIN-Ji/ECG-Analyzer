import os
import tempfile
import google.generativeai as genai

# === Add your Gemini API keys here ===
GEMINI_API_KEYS = [
    "AIzaSyBWyEsq0clnAfP5HO6idRAvkgD-hODy4DI",   # Primary key
    "AIzaSyCQ9CMngLk00FjdKQ6VsxJPuAF3qiEtwPQ"    # Secondary key
]

def analyze_ecg_classification(uploaded_file_bytes, file_type="pdf"):
    """
    Analyzes ECG file bytes and returns classification using Gemini API with fallback to secondary API key.
    
    Args:
        uploaded_file_bytes: File bytes from streamlit uploaded file
        file_type: Type of file ("pdf" or "image")
    
    Returns:
        str: Classification result or None if all keys fail
    """
    prompt = """
    Analyze the provided ECG report. Based on your analysis, classify the ECG into ONLY ONE of the following four categories:

    1. Normal: All parameters are within normal limits.
    2. MI: Clear evidence of an acute Myocardial Infarction (e.g., ST-segment elevation).
    3. History of MI: Evidence of a past/old/evolved infarct (e.g., pathological Q-waves without acute ST changes).
    4. Abnormal heartbeat: An arrhythmia or conduction block is the primary finding (e.g., Atrial Fibrillation, AV Block, Tachycardia).

    Your entire response must be ONLY the name of one of the four categories listed above. Do not add any explanation, punctuation, or other text.
    """

    for api_key in GEMINI_API_KEYS:
        try:
            genai.configure(api_key=api_key)

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                tmp_file.write(uploaded_file_bytes)
                tmp_file_path = tmp_file.name

            uploaded_file = genai.upload_file(path=tmp_file_path, display_name="ECG Report")

            model = genai.GenerativeModel(
                model_name="gemini-2.5-pro",
                generation_config={"temperature": 0}
            )

            response = model.generate_content([prompt, uploaded_file])

            genai.delete_file(uploaded_file.name)
            os.unlink(tmp_file_path)

            return response.text.strip()

        except Exception:
            try:
                if 'uploaded_file' in locals():
                    genai.delete_file(uploaded_file.name)
                if 'tmp_file_path' in locals():
                    os.unlink(tmp_file_path)
            except:
                pass
            continue

    return None
