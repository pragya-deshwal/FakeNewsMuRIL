# Multilingual Fake News Detection (NLP + Transformers)
## Detect fake news across Indian languages using MuRIL (Google Transformer Model) + ML models.

## Key Highlights
- 70,000+ news samples (Hindi, Marathi, Gujarati, Telugu)
- Transformer-based model (MuRIL)
- Achieved 99.4% accuracy
- Multimodal approach using CLIP (image + text)
- End-to-end ML pipeline (training → evaluation → inference)

## Architecture
Dataset (Multilingual) -> Preprocessing -> MuRIL / TF-IDF / CLIP -> Model Training -> Evaluation -> Prediction (Real / Fake)

## Results
| Model         | Accuracy |
| ------------- | -------- |
| MuRIL         | 99.4%    |
| Random Forest | 99.49%   |
| CLIP          | 96.03%   |

## Tech Stack
- Python
- Hugging Face Transformers
- PyTorch
- scikit-learn
- Pandas

## How to Run
- git clone https://github.com/pragya-deshwal/FakeNewsMuRIL.git
- cd FakeNewsMuRIL
- pip install -r requirements.txt
- python app.py

## Key Learnings
- Transformer models outperform traditional ML for multilingual tasks
- Data preprocessing significantly impacts performance
- Multimodal models (CLIP) improve robustness

## Future Improvements
- Add more Indian languages
- Deploy as web app / API
- Integrate real-time fact-checking

## Project Report
https://drive.google.com/file/d/108Vf8KF7OhCPw7ZGH1lg3pHvGbnCdbI8/view?usp=drive_link

## Results

### CLIP Confusion Matrix
![CLIP Confusion Matrix](images/CLIP%20Confusion%20Matrix.png)

### CLIP Graph
![CLIP Graph](images/CLIP%20Graph.png)

### English Dataset Confusion Matrix
![English Dataset](images/Confusion%20Matrix%20for%20English%20Dataset.png)

### MuRiL Confusion Matrix
![MuRiL Confusion](images/MuRiL%20Confusion%20Matrix.png)

### MuRiL Graph
![MuRiL Graph](images/MuRiL%20Graph.png)

