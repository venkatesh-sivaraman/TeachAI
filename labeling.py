from nltk.stem import WordNetLemmatizer
import nltk

exclude_words = [
    "top", # TODO can we improve region detection using these words?
    "bottom",
    "left",
    "right",
    "middle",
    "frame",
    "picture",
    "photo",
    "region",
    "part",
    "section",
    "area"
]

lemmatizer = WordNetLemmatizer()

def find_nouns(phrase):
    tokens = nltk.word_tokenize(phrase)
    pos = nltk.pos_tag(tokens)
    return set([lemmatizer.lemmatize(word) for word, p in pos if p.startswith("NN") and
                word.lower() not in exclude_words])
