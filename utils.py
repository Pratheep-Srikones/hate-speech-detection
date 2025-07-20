import re 
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'https?://\S+|www\.\S+', '', tweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)


    tweet = emoji.replace_emoji(tweet, replace='')

    tweet = re.sub(r'[^a-z\s]', '', tweet)

    tokens = nltk.word_tokenize(tweet)

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]

    return ' '.join(tokens)

