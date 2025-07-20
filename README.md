# Hate Speech Detection using Machine Learning

A machine learning-based classifier that detects hate speech, offensive language, and neutral content in tweets. Built with scikit-learn and NLTK, this project demonstrates effective preprocessing, class balancing, and Random Forest classification to achieve high accuracy.

## Project Structure

```
pratheep-srikones-hate-speech-detection/
├── hate-speech.csv                 # Raw dataset
├── hate_speech_detection.ipynb     # Development notebook
├── main.py                         # CLI for inference
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
* Evaluation with precision, recall, F1-score, and confusion matrix

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

4. Run the script:

   ```bash
   python main.py "Your tweet text here"
   ```

## Usage

Use the CLI to predict the class of a tweet:

```bash
$ python main.py "please don't kill animals"
Prediction: Neutral
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



Developed by Pratheep Srikones

---

For feedback, suggestions, or collaborations, feel free to reach out!
