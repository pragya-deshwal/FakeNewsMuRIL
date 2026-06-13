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
