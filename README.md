# MedAssist AI — Medical Diagnosis Chatbot Starter

A Streamlit prototype based on the uploaded hackathon problem statement:
- Text questions and medical image uploads
- Gemini-generated medical information and cautious next steps
- Optional YOLO specialist slots for eye, chest/lung, and bone images
- Language selection for responses
- Safety warnings and model-status display

## Important limitation
This ZIP does NOT include trained medical YOLO weights or a validated dataset. Until you
train and validate a suitable model and add its weights, the detector is inactive. Gemini
text/image explanations are not a clinical diagnosis. Do not use this app for medical
decisions or emergencies.

## Install and run on Windows + VS Code

1. Extract the ZIP to a folder such as `C:\\Users\\YOUR_NAME\\Downloads\\medical_diagnosis_chatbot_starter`.
2. Open VS Code → **File → Open Folder** → choose the extracted folder.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   ```
   If unavailable, try `python -m venv .venv`.
5. Activate it:
   ```powershell
   .\\.venv\\Scripts\\Activate.ps1
   ```
   If PowerShell blocks activation, run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\\.venv\\Scripts\\Activate.ps1
   ```
6. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
7. Create a new file named `.env` in the same folder as `app.py`, and add:
   ```env
   GEMINI_API_KEY=PASTE_YOUR_REAL_KEY_HERE
   GEMINI_MODEL=gemini-2.5-flash
   ```
   Get the key from Google AI Studio. Keep it private; never upload `.env` to GitHub.
8. Run:
   ```powershell
   python -m streamlit run app.py
   ```
9. Open the `Local URL` shown in the terminal, usually `http://localhost:8501`.

## Use
1. Choose the specialist and response language in the sidebar.
2. Type a question and/or upload a PNG/JPG/JPEG image.
3. Click **Analyze / Ask MedAssist**.
4. Read the response and limitations.

## Enable optional YOLO inference
Place properly trained, validated specialist weights in:
- `models/eye_best.pt`
- `models/chest_best.pt`
- `models/bone_best.pt`

Do not rename unrelated pretrained weights and claim they detect medical conditions. Document dataset license, class definitions, train/validation/test split, class-wise precision/recall, sensitivity/specificity, and known limitations before demonstrating model performance.

## Troubleshooting
- Python command missing: install Python 3.11/3.12 and enable Add Python to PATH; reopen VS Code.
- Missing module: activate `.venv`, then run `python -m pip install -r requirements.txt`.
- Gemini key error: check `.env` is beside `app.py`, then restart Streamlit.
- Gemini 429/503: check model access/quota and retry later.
- YOLO weights not configured: expected until you add trained weights.
