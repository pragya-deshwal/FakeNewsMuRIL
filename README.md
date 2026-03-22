# FakeNewsMuRIL

A Fake News Detection system for Indian Languages using Google's MuRIL (Multilingual Representations for Indian Languages) transformer model.

## Overview

This project fine-tunes the `google/muril-base-cased` model to classify news articles as **Real** or **Fake** across multiple Indian languages. The model has been trained on a combined dataset of news articles in Gujarati, Hindi, Marathi, and Telugu.

## Supported Languages

- Gujarati
- Hindi
- Marathi
- Telugu

## Project Structure

```
FakeNewsMuRIL/
├── app.py              # Main training and inference script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## External Resources

Due to large file sizes, the following resources are hosted externally:

### Pre-trained Model

The fine-tuned MuRIL model is available on Google Drive:

**Download Link:** [final_muril_model](https://drive.google.com/file/d/1idz4R1NysXNSTR66wZoJDPwUrrzEdUCr/view?usp=sharing)

After downloading, extract the folder to your project root directory:
```
FakeNewsMuRIL/
└── final_muril_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.txt
```

### Datasets

The training datasets are available on Google Drive:

**Download Link:** [Datasets](https://drive.google.com/file/d/1idz4R1NysXNSTR66wZoJDPwUrrzEdUCr/view?usp=sharing) 

The dataset contains the following files:
- `gujarati_news.csv` (~124 MB)
- `hindi_news.csv` (~193 MB)
- `marathi_news.csv` (~139 MB)
- `telugu_news.csv` (~136 MB)

Each CSV file contains two columns:
- `text` - The news article content in the respective language
- `label` - Binary classification label (0 = Real News, 1 = Fake News)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/FakeNewsMuRIL.git
cd FakeNewsMuRIL
```

### Step 2: Install Dependencies

```bash
pip install torch transformers datasets pandas scikit-learn numpy
```

### Step 3: Download External Resources

1. Download the pre-trained model from the link above
2. Extract `final_muril_model/` folder to the project root
3. (Optional) Download datasets if you want to retrain the model

## Usage

### Option 1: Using the Pre-trained Model (Inference Only)

If you only want to use the model for predictions, download the `final_muril_model/` folder and use the following code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model
model_path = "final_muril_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_fake_news(text):
    """
    Predict whether a news article is real or fake.
    
    Args:
        text (str): News article text in any supported language
        
    Returns:
        tuple: (label, confidence) where label is 'Real' or 'Fake'
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    label = "Fake" if pred == 1 else "Real"
    confidence = probs[0][pred].item()
    return label, confidence

# Example predictions
examples = [
    "મોદી સરકાર નવી યોજના લાવી છે",        # Gujarati
    "सरकार ने नई योजना शुरू की",            # Hindi
    "सरकारने योजना सुरू केली आहे",          # Marathi
    "ప్రభుత్వం కొత్త పథకాన్ని ప్రారంభించింది"  # Telugu
]

for text in examples:
    label, confidence = predict_fake_news(text)
    print(f"Text: {text}")
    print(f"Prediction: {label} (Confidence: {confidence*100:.2f}%)")
    print()
```

### Option 2: Training from Scratch

If you want to train the model yourself:

1. Download all dataset files and place them in the project root
2. Update the `DATA_PATH` variable in `app.py` to your project path
3. Run the training script:

```bash
python app.py
```

The training script will:
1. Load and combine all language datasets
2. Split data into 80% training and 20% testing (stratified)
3. Tokenize the text using MuRIL tokenizer
4. Fine-tune the model for 3 epochs
5. Save the best model to `final_muril_model/`

### Training Configuration

The model is trained with the following hyperparameters:

- Base Model: `google/muril-base-cased`
- Batch Size: 2 (per device)
- Number of Epochs: 3
- Max Sequence Length: 128 tokens
- Evaluation Strategy: Every 1000 steps
- Optimizer: AdamW (default)
- Learning Rate: 5e-5 (default)
- Train/Test Split: 80/20 (stratified)

### Resuming Training

The script supports resuming training from checkpoints. If training is interrupted, the script will automatically resume from the last saved checkpoint when run again.

## Model Architecture

This project uses **MuRIL** (Multilingual Representations for Indian Languages), a BERT-based transformer model developed by Google Research.

### Why MuRIL?

- Pre-trained specifically on Indian languages
- Supports 17 Indian languages including transliterated text
- Outperforms multilingual BERT on Indian language tasks
- Handles code-mixing (multiple languages in same text)

### Supported Languages by MuRIL

Assamese, Bengali, English, Gujarati, Hindi, Kannada, Kashmiri, Malayalam, Marathi, Nepali, Oriya, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, and Urdu.

### Model Output

The model outputs a binary classification:
- Class 0: Real News
- Class 1: Fake News

Along with confidence scores for each prediction.


## File Descriptions

### app.py

The main Python script that handles:
- Data loading and preprocessing
- Model initialization and tokenization
- Training with Hugging Face Trainer API
- Model saving and inference

### Excluded Files (Not in Repository)

The following files are excluded from the repository due to their large size:

- `checkpoints/` - Training checkpoint files (regenerated during training)
- `muril_offline/` - Offline copy of base MuRIL model (downloaded automatically)
- `muril_output/` - Additional training outputs (regenerated during training)
- `*.csv` - Dataset files (available via external download)
- `final_muril_model/` - Trained model (available via external download)

## Requirements

```
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
pandas>=1.3.0
scikit-learn>=0.24.0
numpy>=1.21.0
```

## Troubleshooting

### Out of Memory Error

If you encounter memory issues during training:
- Reduce `per_device_train_batch_size` in `app.py`
- Use gradient accumulation
- Enable mixed precision training

### Model Download Issues

If MuRIL model download fails:
1. Check your internet connection
2. Download manually from [Hugging Face](https://huggingface.co/google/muril-base-cased)
3. Place files in `muril_offline/` folder

### Dataset Encoding Issues

All CSV files should be encoded in UTF-8. If you face encoding errors:
```python
df = pd.read_csv("file.csv", encoding="utf-8-sig")
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Google Research](https://github.com/google-research/bert) for developing MuRIL
- [Hugging Face](https://huggingface.co/) for the Transformers library
- The open-source community for dataset contributions

## References

- MuRIL Paper: [MuRIL: Multilingual Representations for Indian Languages](https://arxiv.org/abs/2103.10730)
- Hugging Face Model: [google/muril-base-cased](https://huggingface.co/google/muril-base-cased)
