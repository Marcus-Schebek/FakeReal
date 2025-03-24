import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from spellchecker import SpellChecker
import matplotlib.pyplot as plt
from streamviz import gauge

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Data Cleaning
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)  # Remove everything except letters
    text = ' '.join(text.split())  # Remove extra whitespace
    text = text.lower()  # Convert to lowercase
    return text

# Lemmatization
lemmatizer = WordNetLemmatizer()
def tokenize_and_lemmatize(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    lem = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return lem

# Spell Checking
def correct_text(text):
    spell = SpellChecker(language='pt')
    words = text.split()
    # Regex patterns for protected elements
    # Regex patterns for protected elements
    protected_patterns = [
        r'\b[A-Z]{2,}\b',          # Acronyms
        r'\d+',                     # Numbers
        r'[A-Z][a-z]+',             # Proper nouns
        r'[.,:;!?]+',               # Punctuation
        r'[()\[\]{}"\'<>]+',        # Symbols
        r'\b\w+-\w+\b',             # Hyphenated words
        r'\d+/\d+(?:/\d+)?'         # Dates (25/01 or 25/01/2023)
    ]
    protected_regex = re.compile('|'.join(protected_patterns))
    
    # Enhanced tokenizer pattern
    token_pattern = r'''
        \w+-\w+|                    # Hyphenated words
        \w+(?:'\w+)?|               # Words with apostrophes
        [.,:;!?()\[\]{}"\'<>]|      # Punctuation/symbols
        \s                           # Whitespace
    '''
    tokens = re.findall(token_pattern, text, re.VERBOSE)
    corrected_tokens = []
    
    for token in tokens:
        # Skip whitespace and protected elements
        if token.isspace() or protected_regex.fullmatch(token):
            corrected_tokens.append(token)
            continue
            
        # Handle hyphenated words specially
        if '-' in token:
            parts = token.split('-')
            corrected_parts = []
            for part in parts:
                if part in spell or len(part) < 4:
                    corrected_parts.append(part)
                else:
                    correction = spell.correction(part)
                    if correction and sum(a == b for a, b in zip(part.lower(), correction.lower())) / len(part) > 0.6:
                        corrected_parts.append(correction)
                    else:
                        corrected_parts.append(part)
            corrected_token = '-'.join(corrected_parts)
            corrected_tokens.append(corrected_token)
            continue
            
        # Check if word is in dictionary
        if token in spell:
            corrected_tokens.append(token)
            continue
            
        # Get correction with confidence check
        correction = spell.correction(token)
        
        # Only apply correction if confident
        if (correction and 
            len(token) > 3 and
            sum(a == b for a, b in zip(token.lower(), correction.lower())) / len(token) > 0.5):
            
            # Preserve original capitalization
            if token[0].isupper():
                correction = correction.capitalize()
            corrected_tokens.append(correction)
        else:
            corrected_tokens.append(token)
    
    # Reconstruct text with original formatting
    return ''.join(corrected_tokens)

# Calculate difference between original and corrected text
def calculate_difference(original_text, corrected_text):
    original_words = original_text.split()
    corrected_words = corrected_text.split()
    difference = sum(1 for p1, p2 in zip(original_words, corrected_words) if p1 != p2)
    difference_percentage = (difference / len(original_words)) * 100
    return difference_percentage

# Classify text quality based on difference percentage
def classify_text_quality(difference_percentage):
    if difference_percentage < 2:
        return "Excellent"
    elif difference_percentage < 5:
        return "Good"
    elif difference_percentage < 10:
        return "Fair"
    else:
        return "Poor"

# Predict Fake News Probability
def predict_fake_news_probability(text, model, vectorizer, difference_percentage):
    cleaned_text = clean_text(text)
    text_features = vectorizer.transform([cleaned_text])
    probability = model.predict_proba(text_features)
    weight = 1 + (difference_percentage / 100)  # Weight based on text quality
    fake_news_probability = probability[0][0] * weight
    if fake_news_probability > 1.0:
        fake_news_probability = 1.0
    return fake_news_probability

# Load data and train model
def load_data_and_train_model():
    df = pd.read_csv('db/out_df.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['check_id'], test_size=0.2, random_state=42)
    tfidf_vec = TfidfVectorizer()
    xtrain_tfidf = tfidf_vec.fit_transform(X_train)
    xtest_tfidf = tfidf_vec.transform(X_test)
    return tfidf_vec, xtrain_tfidf, y_train

# Function to create donut charts
def create_donut_chart(percentage, title, colors):
    fig, ax = plt.subplots(figsize=(4, 4))  # Smaller size to fit side by side
    ax.pie([percentage, 100 - percentage], colors=colors, startangle=90, wedgeprops=dict(width=0.3))
    ax.text(0, 0, f"{percentage:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold')
    plt.title(title, fontsize=12, pad=10)
    return fig
# Function to create a gauge chart
def create_gauge_chart(value, title, low_color, mid_color, high_color, low_range, mid_range, theme):
    """
    Creates a gauge chart using the streamviz library.

    Args:
        value (float): The value to display on the gauge.
        title (str): The title of the gauge.
        low_color (str): Color for the low range (e.g., "#FF1708" for red).
        mid_color (str): Color for the mid range (e.g., "#FF9400" for orange).
        high_color (str): Color for the high range (e.g., "#1B8720" for green).
        low_range (float): The threshold for the low range (e.g., 0.29).
        mid_range (float): The threshold for the mid range (e.g., 0.69).

    Returns:
        None (displays the gauge directly in Streamlit).
    """
    gauge(
        gVal=value,
        gTitle=title,
        gMode='gauge+number',
        gSize="MED",
        gcLow=low_color,
        gcMid=mid_color,
        gcHigh=high_color,
        grLow=low_range,
        grMid=mid_range,
        sFix="%",
        gTheme=theme
    )