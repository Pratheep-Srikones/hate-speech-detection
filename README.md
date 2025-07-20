# Hate Speech Detection using Machine Learning

A machine learning-based classifier that detects hate speech, offensive language, and neutral content in tweets. Built with scikit-learn, NLTK, and FastAPI, this project demonstrates effective preprocessing, class balancing, and Random Forest classification to achieve high accuracy.

## Project Structure

```
pratheep-srikones-hate-speech-detection/
├── hate-speech.csv                 # Raw dataset
├── hate_speech_detection.ipynb     # Development notebook
├── main.py                         # FastAPI app for inference
├── requirements.txt                # Project dependencies
├── utils.py                        # Preprocessing and helper functions
└── models/
    ├── rf_model.pkl                # Trained Random Forest model
    └── tfidf_vectorizer.pkl        # Trained TF-IDF vectorizer
```

## Features

* Text preprocessing and cleaning with NLTK
* Class balancing using synonym-based augmentation
* TF-IDF vectorization (unigrams + bigrams)
* Random Forest classifier for multiclass classification
* FastAPI-based server for easy model deployment and inference

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/pratheep-srikones-hate-speech-detection.git
   cd pratheep-srikones-hate-speech-detection
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```

## Usage

Once the server is running, use a tool like `curl`, Postman, or a browser to send a request:

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "Please be kind to others"}'
```

Response:

```json
{
  "prediction": "Neutral"
}
```

## Model Performance

* **Accuracy**: 89%
* **Precision / Recall / F1**:

  * Hate Speech (Class 0): 88 / 85 / 87
  * Offensive (Class 1): 88 / 86 / 87
  * Neutral (Class 2): 90 / 94 / 92

## References

* [NLTK Documentation](https://www.nltk.org/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [NLPAug - Data Augmentation for NLP](https://github.com/makcedward/nlpaug)
* [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
* [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Author

Developed by Pratheep Srikones

---

For feedback, suggestions, or collaborations, feel free to reach out!
