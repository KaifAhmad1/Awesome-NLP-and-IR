## NLP 
### **NLP Preprocessing**

 - **Case Folding:**   Case folding/lowercasing is a preprocessing technique in Natural Language Processing (NLP) that standardizes the text by converting all characters to a single case, typically lowercase. This step is essential for various NLP tasks as it ensures uniformity and consistency in text data, thereby enhancing the performance of downstream applications.
   - **For Example**
      - `Artificial Intelligence` becomes `artificial intelligence`
      - `Data Science` becomes `data science`
        
    ``` Python
         text = "Machine Learning is FUN!"
         lowercased_text = text.lower()
         print(lowercased_text)  # Output: "machine learning is fun!" 
    ```
- **Contraction Mapping:** Contraction mapping refers to the process of expanding contractions, which are shortened forms of words or phrases, into their complete versions. For example:
   - **For Example**
     - `I'm` becomes `I am`
     - `can't` becomes `cannot`
     - `they're` becomes `they are`
  ``` Python
  import re
  contractions_dict = {
    "i'm": "I am", "can't": "cannot", "won't": "will not", "isn't": "is not",
    "it's": "it is", "don't": "do not", "didn't": "did not", "couldn't": "could not",
    "they're": "they are", "we're": "we are", "you're": "you are"
  }
  # Function to expand contractions
  def expand_contractions(text, contractions_dict=contractions_dict):
      """
      Expand contractions in text using a provided dictionary.
      """
      contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
      return contractions_re.sub(lambda match: contractions_dict[match.group(0)], text)
  # Input
  text = "I'm learning NLP. It's fascinating! Don't you think so?"
  expanded_text = expand_contractions(text)
  print(expanded_text)  
  ```
  ```
  Output: "I am learning NLP. It is fascinating! Do not you think so?"  
  ```
- **Correcting Spelling Errors:** Correcting spelling errors in Natural Language Processing (NLP) is a common task aimed at improving text quality and accuracy. This process involves identifying and replacing misspelled words with their correct counterparts using various techniques such as:
- 1. **Dictionary-based approaches:** Utilizing a dictionary to look up correct spellings and suggest replacements for misspelled words.
- 2. **Edit distance algorithms:** Calculating the distance between words using metrics like `Levenshtein / Minimum Edit` distance to find the most likely correct spelling.
- 3. **Rule-based methods:** Applying spelling correction rules and heuristics to identify and correct common spelling mistakes.
``` Python 
from textblob import TextBlob
# Input sentence with mistake
sentence = "I havv a pbroblem wit my speling."
# Create a TextBlob object
blob = TextBlob(sentence)

corrected_sentence = str(blob.correct())
print("Original sentence:", sentence)
print("Corrected sentence:", corrected_sentence)

```
```
Output:
Original sentence: I havv a pbroblem wit my speling.
Corrected sentence: I have a problem with my spelling.
```

- **Deduplication / Duplicate Removal:**  Deduplication in the context of Natural Language Processing (NLP) involves identifying and removing duplicate entries in a dataset. This process is crucial for ensuring data quality and accuracy, especially when dealing with large text corpora.
- 1. **Using Pandas for Deduplication:**
 ``` Python
import pandas as pd
dataframe = {
    'text': [
        "This is a sample sentence.",
        "This is a sample sentence.",
        "Another example sentence.",
        "This is a different sentence.",
        "Another example sentence."
    ]
}

data = pd.DataFrame(dataframe)
print("Original DataFrame:")
print(data)
data_deduplicated = data.drop_duplicates(subset='text')
print("\nDeduplicated DataFrame:")
print(data_deduplicated)

```
```
Output:
Original DataFrame:
                            text
0     This is a sample sentence.
1     This is a sample sentence.
2      Another example sentence.
3  This is a different sentence.
4      Another example sentence.

Deduplicated DataFrame:
                            text
0     This is a sample sentence.
2      Another example sentence.
3  This is a different sentence.
 ```
- 2. **Using Fuzzy Matching for Deduplication:** Fuzzy matching in NLP refers to the process of finding strings that are approximately equal to a given pattern. It is particularly useful in scenarios where exact matches are not possible due to typographical errors, variations in spelling, or other inconsistencies in the text data. Fuzzy matching is widely used in applications like data deduplication, record linkage, and spell-checking.
     - The `fuzzywuzzy` library in Python is commonly used for fuzzy matching. It uses the Levenshtein Distance to calculate the differences between sequences.
 ``` Python 
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Input Strings
string1 = "This is a sample sentence."
string2 = "This is a smaple sentnce."

# Calculate the similarity ratio
similarity_ratio = fuzz.ratio(string1, string2)
print(f"Similarity ratio between '{string1}' and '{string2}': {similarity_ratio}")
choices = ["This is a sample sentence.", "This is a simple sentence.", "Totally different sentence."]

# Find the best match
best_match = process.extractOne(string2, choices)
print(f"Best match for '{string2}': {best_match}")

```
```
Output:
Similarity ratio between 'This is a sample sentence.' and 'This is a smaple sentnce.': 94
Best match for 'This is a smaple sentnce.': ('This is a sample sentence.', 94)
 ```
- **Expanding Abbreviations and Acronyms:**  Expanding abbreviations and acronyms is an important task in Natural Language Processing (NLP) to enhance the understanding and processing of text. Here are some key methods and approaches used to achieve this:
- 1. **Dictionary-Based Methods:** These methods involve using precompiled lists of abbreviations and their expansions. The dictionary can be curated manually or generated from various sources such as online databases or domain-specific corpora.
- 2. **Rule-Based Methods:** These methods use linguistic rules and patterns to identify and expand abbreviations. For example, context-specific rules can be applied based on the position of the abbreviation in the text or its surrounding words.
- 3. **Statistical Methods:** These methods rely on statistical models and machine learning algorithms to predict expansions based on large corpora. Techniques include: N-gram models, Hidden Markov Models (HMMs) and Conditional Random Fields(CRFs)
      - **Simple Dictionary-based Implementation**
``` Python
abbreviation_dict = {
    "NLP": "Natural Language Processing",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "GPU": "Graphics Processing Unit",
    "USA": "United States of America"
}

def expand_abbreviations(text, abbreviation_dict):
    words = text.split()
    expanded_words = []
    for word in words:
        clean_word = word.strip('.,!?')
        if clean_word in abbreviation_dict:
            # Append the expanded form from the dictionary
            expanded_words.append(abbreviation_dict[clean_word])
        else:
            expanded_words.append(word)
    return ' '.join(expanded_words)

# Input
text = "NLP and AI are subsets of ML."
expanded_text = expand_abbreviations(text, abbreviation_dict)
print(expanded_text)
```
```
Output:
Natural Language Processing and Artificial Intelligence are subsets of Machine Learning
```
 
- **Stemming:** Stemming in Natural Language Processing (NLP) refers to the process of reducing words to their base or root form, known as the `stem`, by removing prefixes and suffixes. The stem may not always be a valid word in the language, but it represents the core meaning of the word, thereby helping to group similar words. Types of Stemming Algorithms: 
 - 1. **Porter Stemmer**
 - 2. **Snowball Stemmer**
 - 3. **Lancaster Stemmer**
      -  Stemming is preferred over lemmatization for its computational efficiency and speed, making it suitable for tasks like information retrieval, search systems, text classification, and text mining.

``` Python 
from nltk.stem import PorterStemmer, SnowballStemmer

# Porter Stemmer
porter = PorterStemmer()
word_porter = 'Dying'
stem_porter = porter.stem(word_porter)
print(f"Original: {word_porter}, Stemmed: {stem_porter}") 

# Snowball Stemmer
snowball = SnowballStemmer('english')
word_snowball = 'Continuing'
stem_snowball = snowball.stem(word_snowball)
print(f"Original: {word_snowball}, Stemmed: {stem_snowball}")
```
``` 
Original: Dying, Stemmed: dy
Original: Continuing, Stemmed: continu
```
- **Lemmatization:** Lemmatization is another crucial text normalization technique in Natural Language Processing (NLP) that involves reducing words to their base or dictionary form, known as the "lemma." Unlike stemming, which simply chops off affixes to obtain the root form, lemmatization considers the context of the word and ensures that the resulting base form is a valid word in the language. Lemmatization algorithms rely on linguistic rules and lexical resources to map inflected words to their base or dictionary forms.
- 1. **Rule-Based Lemmatization:** Rule-based lemmatization algorithms rely on linguistic rules and patterns to derive lemmas from inflected forms. These rules are often derived from linguistic knowledge and may vary based on language and context.
- 2. **Lexical Resource-based Lemmatization:** WordNet is a lexical database of the English language that includes information about word meanings, relationships between words, and word forms. Lemmatization algorithms leveraging WordNet typically use its morphological information and lexical relationships to derive lemmas.
      - Lemmatization provides more accurate, context-aware, and consistent base forms of words compared to stemming, making it essential for applications like virtual assistant and question-answering systems, text normalization, machine translation, and text summarization tasks requiring readable and meaningful text.
``` Python 
import nltk
nltk.download('wordnet')
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Lemmatization function
def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)
# Input 
sentence = "The striped bats are hanging on their feet for best"
lemmatized_sentence = lemmatize_sentence(sentence)
print(lemmatized_sentence)
```
```
Output:
The striped bat are hanging on their foot for best
```
- **Noise Removing:** Noise removal in NLP involves eliminating irrelevant or unwanted elements, such as HTML tags, special characters, punctuation, stop words, and numerical values, from text data. This process aims to clean and standardize the data, making it more suitable for analysis or model training. The goal of noise removal is to clean the text data by stripping away these elements while preserving the meaningful content. This typically involves a series of preprocessing steps, which may include:
- 1. **Stripping HTML Tags:** Removing HTML markup from text obtained from web sources.
 ``` Python 
from bs4 import BeautifulSoup

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text
# Input
html_text = "<p>This is <b>HTML</b> text.</p>"
clean_text = strip_html_tags(html_text)
print(clean_text)  # Output: This is HTML text.
  ```
```
This is HTML text.
```
- 2. **Removing Special Characters:** Eliminating non-alphanumeric characters, punctuation marks, and symbols.
``` Python 
import re
def remove_special_characters(text):
    # Remove non-alphanumeric characters and symbols
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# Input:
text_with_special_characters = "This is a text with special characters: @#$%^&*()_+"
clean_text = remove_special_characters(text_with_special_characters)
print(clean_text)
```
```
This is a text with special characters
```
- 3. **Removing Stop Words:** Eliminating non-alphanumeric characters, punctuation marks, and symbols.
 ``` Python 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_text = ' '.join([word for word in words if word.lower() not in stop_words])
    return filtered_text

# Input
text_with_stop_words = "This is a simple example to demonstrate removing stop words."
clean_text = remove_stop_words(text_with_stop_words)
print(clean_text)  
 ```
```
Output: This simple example demonstrate removing stop words.
```
- 4. **Removing Numerical Values:**  Eliminating digits and numbers that may not be relevant to the analysis.
 ``` Python 
import re
def remove_numerical_values(text):
    clean_text = re.sub(r'\d+', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# Input
text_with_numbers = "There are 123 apples and 45 oranges in 2024."
clean_text = remove_numerical_values(text_with_numbers)
print(clean_text)  
 ```
- 5. **Handling Emojis and Emoticons:** Removing or replacing emojis and emoticons with descriptive text.
``` Python 
import re
import emoji

def remove_emojis_and_emoticons(text):
    text = emoji.replace_emoji(text, replace='')
    # Define a regex pattern for common emoticons
    emoticon_pattern = r'[:;=8][\-o\*\']?[)\]\(\[dDpP/:\}\{@\|\\]'
    text = re.sub(emoticon_pattern, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# Input:
text_with_emojis_and_emoticons = "Hello 😊! This is an example text with emoticons :) and emojis 🚀."
clean_text = remove_emojis_and_emoticons(text_with_emojis_and_emoticons)
print(clean_text)
```  
```
Hello ! This is an example text with emoticons and emojis .
```
- 6. **Removing Non-Linguistic Symbols:**  Eliminating symbols or characters that do not convey linguistic meaning, such as currency symbols, mathematical operators, or trademark symbols.
 ``` Python 
import re
import string

def remove_non_linguistic_symbols(text):
    symbols_pattern = r'[^\w\s]'
    clean_text = re.sub(symbols_pattern, '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text
# Input
text_with_symbols = "Price: $100. Use coupon code SAVE20%! Email: example@example.com ©2024."
clean_text = remove_non_linguistic_symbols(text_with_symbols)
print(clean_text) 
  ```
```
Price 100 Use coupon code SAVE20 Email exampleexamplecom 2024
```

- **Tokenization:** Tokenization is the process of splitting text into smaller, meaningful units called tokens. These tokens can represent words, subwords, or characters and are the foundational elements used in NLP tasks like text analysis, machine translation, and sentiment analysis. Types of Tokenization below:
- 1. **Word Tokenization:** Word tokenization involves breaking text into individual words. It's the most intuitive form of tokenization but can be challenging for languages without clear word boundaries or texts with contractions and special characters.
      - Example: `Tokenization is fun!` is tokenized into [`Tokenization`, `is`, `fun`, `!`].
        
- 2. **Subword Tokenization:** Subword tokenization divides text into smaller units than words, which can help handle out-of-vocabulary words and morphological variations. Popular subword tokenization techniques include:
     -  **Byte Pair Encoding (BPE)**: Byte Pair Encoding (BPE) is a subword tokenization method that iteratively merges the most frequent pair of bytes or characters in a corpus to form subword units. It helps reduce the vocabulary size and handle out-of-vocabulary words effectively.
           - BPE efficiently reduces vocabulary size and handles out-of-vocabulary words with simplicity, making it ideal for machine translation, text generation, language modelling, and speech recognition.  Some prominent models that leverage Byte Pair Encoding (BPE) include GPT, GPT-2, RoBERTa, BART, and DeBERTa
 ```
       Steps
- Start with a vocabulary of all unique characters in the text.
- Count the frequency of each adjacent character pair in the text.
- Find and merge the most frequent pair into a new token.
- Replace all instances of this pair in the text with the new token.
- Repeat steps 2-4 for a predefined number of merges or until the desired vocabulary size is achieved.
 ```

   -  **WordPiece Tokenization**: WordPiece is a subword tokenization method originally developed for speech recognition and later adopted by NLP models like BERT. It breaks down words into smaller, more frequent subword units to handle the problem of out-of-vocabulary words and improve model performance by capturing word morphology and semantics more effectively.
         - WordPiece offers advantages in handling rare words efficiently by breaking them down into smaller, meaningful subwords, thus addressing the Out Of Vocabulary (OOV) problem common in word-based tokenization. Its use cases span across various NLP models like BERT, DistilBERT, and Electra, enhancing their ability to understand and process texts more accurately by leveraging subword units that retain linguistic meaning
```
Steps
-  Start with an initial vocabulary, typically consisting of all individual characters and some predefined words from the training corpus.
- Compute the frequency of all substrings (subword units) in the corpus. This includes both individual characters and longer subwords.
- Iteratively merge the most frequent substring pairs to form new subwords. This process continues until the vocabulary reaches a predefined size.
-  Once the vocabulary is built, tokenize new text by matching the longest possible subwords from the vocabulary.
```

   -  **Unigram Tokenization**: Unigram Tokenization is a subword tokenization method that treats each character as a token. It's a straightforward approach where the text is split into its constituent characters, without considering any linguistic rules or context.
         - Unigram Tokenization offers simplicity via straightforward character-level tokenization, making it language agnostic and effective for languages with complex morphology like Japanese or Turkish; it's also useful for text normalization tasks such as sentiment analysis or text classification, prioritizing individual character preservation.
 ```
Steps
- Tokenization: Break down the text into individual characters. Each character becomes a separate token.
- Vocabulary Construction: Build a vocabulary containing all unique characters present in the text.
 ```

   -  **SentencePiece Tokenization**: SentencePiece is an unsupervised text tokenizer and detokenizer that creates subword units without relying on predefined word boundaries, making it language-agnostic and suitable for various languages, including those with complex word formation rules. It supports BPE and Unigram models, includes text normalization, and effectively handles out-of-vocabulary words.
         - SentencePiece is flexible and language-agnostic, reducing out-of-vocabulary issues by generating subword units, making it ideal for machine translation, text generation, speech recognition, and pretrained language models like BERT, T5, and GPT.
```
Steps
- Data Preparation: Collect and preprocess the text corpus.
- Normalization: Standardize the text.
 Model Training:
- BPE:
   Calculate frequencies of adjacent token pairs.
   Merge the most frequent pair into a new token.
   Repeat until the desired vocabulary size is reached.
- Unigram:
   Start with many subwords.
   Assign probabilities and prune the least probable subwords.
   Repeat until the desired vocabulary size is reached.
- Save Model: Output the trained model and vocabulary.
- Tokenization: Use the model to tokenize new text into subwords.
```
[More detailed video explanation by Huggingface](https://huggingface.co/docs/transformers/en/tokenizer_summary)

### Statical NLP 
- **Naive Bayes**: Naive Bayes presents a straightforward yet effective classification approach rooted in `Bayes theorem`, assuming `independence` among features. Here's a simplified rundown:
  - Bayes theorem is a cornerstone of probability theory, revealing the probability of an event based on prior conditions. It's expressed as:
     -    `P(A|B) = (P(B|A) * P(A)) / P(B)`
 - Where, 
     - `P(A|B)` is the probability of event A occurring given that event B has occurred.
     - `P(B|A)` is the probability of event B occurring given that event A has occurred.
     - `P(A)` and `P(B)` are the probabilities of events A and B occurring independently of each other.
 - **Naive Bayes Assumption:** Naive Bayes assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
 - **Types of Naive Bayes:**
     - **Gaussian Naive Bayes:** Assumes features follow a Gaussian (normal) distribution.
     - **Multinomial Naive Bayes:** Suitable for classification with discrete features (e.g., word counts for text).
     - **Bernoulli Naive Bayes:** Assumes features are binary (e.g., presence or absence of a feature).

``` Python 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
# Input Messages
messages = [
    "Hey, want to buy some cheap Viagra?",
    "Meeting for lunch today?",
    "You've won a free vacation! Claim now!",
    "Don't forget to submit your assignment.",
    "URGENT: Your account needs verification.",
]

# Labels for examples 
labels = [1, 0, 1, 0, 1]

# Training data
training_messages = [
    "Buy Viagra for cheap!",
    "Lunch meeting at 12 pm.",
    "Claim your free vacation now!",
    "Submit your assignment by Friday.",
    "Your account requires immediate verification.",
]
training_labels = [1, 0, 1, 0, 1]

# Define a pipeline
model = make_pipeline(
    CountVectorizer(),  
    MultinomialNB(),   
)

# Train the model
model.fit(training_messages, training_labels)

# Inference on example
predictions = model.predict(messages)
for message, label in zip(messages, predictions):
    print(f"Message: {message} | Predicted Label: {'Spam' if label == 1 else 'Not Spam'}")
```
```
Message: Hey, want to buy some cheap Viagra? | Predicted Label: Spam
Message: Meeting for lunch today? | Predicted Label: Not Spam
Message: You've won a free vacation! Claim now! | Predicted Label: Spam
Message: Don't forget to submit your assignment. | Predicted Label: Not Spam
Message: URGENT: Your account needs verification. | Predicted Label: Spam
```

- **N-gram language model:** An n-gram is a sequence of `n` items from a given sample of text or speech. The `items` are typically words or characters, and the sequence can be as short or as long as needed:
    - Unigram (n=1): Single word sequences.
    - Bigram (n=2): Pairs of words.
    - Four-gram (n=4) and higher: Longer sequences.
  - For example, with the sentence `I love natural language processing`:
     - Unigrams: [`I`, `love`, `natural`, `language`, `processing`]
     - Bigrams: [`I love`, `love natural`, `natural language`, `language processing`]
     - Trigrams: [`I love natural`, `love natural language`, `natural language processing`]
-  N-gram models predict the likelihood of a word given the preceding `n-1` words. The core idea is to estimate the probability of the next word in a sequence, given the previous words. Using the chain rule of probability:
<pre>
P(w_n | w_1, w_2, ..., w_{n-1}) = C(w_1, w_2, ..., w_n) / C(w_1, w_2, ..., w_{n-1})
</pre>
- `P(w_n | w_1, w_2, ..., w_{n-1})` represents the probability of the word `w_n` occurring after the sequence of words `w_1, w_2, ..., w_{n-1}`
- `C(w_1, w_2, ..., w_n)` is the count of the n-gram `(w_1, w_2, ..., w_n)` in the training corpus.
- `C(w_1, w_2, ..., w_{n-1})` is the count of the (n-1)-gram `(w_1, w_2, ..., w_{n-1})` in the training corpus.

``` Python
import nltk
nltk.download('punkt')
from nltk.util import ngrams
from collections import defaultdict

class NgramLanguageModel:
    def __init__(self, n):
        # Initialize model with n-gram size and dictionaries for storing n-grams and their contexts
        self.n = n
        self.ngrams = defaultdict(int)
        self.context = defaultdict(int)

    def train(self, text):
        # Tokenize  text and generate n-grams
        tokens = nltk.word_tokenize(text)
        n_grams = ngrams(tokens, self.n)
        # Count occurrences of n-grams and their contexts
        for gram in n_grams:
            self.ngrams[gram] += 1
            self.context[gram[:-1]] += 1

    def predict_next_word(self, context):
        # Find candidates for the next word based on the given context
        candidates = {gram[-1]: count / self.context[context] for gram, count in self.ngrams.items() if gram[:-1] == context}
        # Return word with the highest probability, or None if no candidates exist
        return max(candidates, key=candidates.get) if candidates else None

# Input
text = "I love natural language processing"
model = NgramLanguageModel(n=2)
model.train(text)
next_word = model.predict_next_word(("natural",))
print(next_word)
```
```
language
```
- **Markov Chain:** A Markov Chain is a way to model a system where the probability of moving to the next state depends only on the current state.
  - Components of Markov Chain:
      - 1. **States:** Different conditions or situations the system can be in. For example, the weather can be either Sunny or Rainy.
             - A set of states `S = {s1, s2, …, sn}`.

      - 2. **Transition Probabilities:** The chances of moving from one state to another.
             - A transition probability matrix `P = [pij]`, where `pij = P(Xt+1 = sj | Xt = si)`.

- **Hidden Markov Model (HMM):** A Hidden Markov Model (HMM) is a statistical model where the system being modelled is assumed to follow a Markov process with hidden states. In contrast to Markov Chains, in HMMs, the state is not directly visible, but output dependent on the state is visible.
- Components:
   - **Initial State Distribution:** Probabilities of starting in each hidden state.
   - **Hidden States:** A finite set of states that are not directly observable.
   - **Observable States:** A finite set of states that can be directly observed.
   - **Transition Probabilities:** Probabilities of transitioning between hidden states.
   - **Emission Probabilities:** Probabilities of an observable state being generated from a hidden state.
- An HMM can be characterized by:
   - Initial state probabilities `π = [πi]`, where `πi = P(X1 = si)`.
   - A set of hidden states `S = {s1, s2, …, sn}`.
   - A set of observable states `O = {o1, o2, …, om}`.
   - Transition probabilities `A = [aij]`, where `aij = P(Xt+1 = sj | Xt = si)`.
   - Emission probabilities `B = [bjk]`, where `bjk = P(Yt = ok | Xt = sj)`.
- **Key Algorithms:**
  - Forward Algorithm: Computes the likelihood of observing a sequence of symbols.
  - Viterbi Algorithm: Finds the most likely sequence of hidden states based on observations.
  - Baum-Welch Algorithm: Trains Hidden Markov Models (HMMs) by estimating transition and emission probabilities from observed data.
 - **Applications:** Applications of Markov Chains and HMM are Part of Speech tagging, Speech Recognition, Name Entity Recognition etc. 

``` Python 
import numpy as np
class HiddenMarkovModel:
    def __init__(self, initial_prob, transition_prob, emission_prob):
        self.initial_prob = initial_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob

    def generate_sequence(self, length):
        hidden_states = [np.random.choice(len(self.initial_prob), p=self.initial_prob)]
        observable_states = [np.random.choice(len(self.emission_prob[0]), p=self.emission_prob[hidden_states[-1]])]

        for _ in range(1, length):
            hidden_states.append(np.random.choice(len(self.transition_prob[0]), p=self.transition_prob[hidden_states[-1]]))
            observable_states.append(np.random.choice(len(self.emission_prob[0]), p=self.emission_prob[hidden_states[-1]]))
        return hidden_states, observable_states

# Input
initial_prob = [0.6, 0.4]
transition_prob = [[0.7, 0.3], [0.4, 0.6]]
emission_prob = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]
hmm = HiddenMarkovModel(initial_prob, transition_prob, emission_prob)
hidden_states, observable_states = hmm.generate_sequence(5)
print("Hidden States:", hidden_states)
print("Observable States:", observable_states)
```
- **Conditional Random Fields(CRFs):** Conditional Random Fields (CRFs) are a type of machine learning model used for tasks where we need to predict a sequence of labels for a given sequence of input data. They're particularly handy in scenarios like analyzing text, where understanding the structure of the data is crucial.
   - Imagine you have a sentence, and you want to tag each word with its part of speech (POS). For example, in the sentence `The cat sat on the mat`, you'd want to label `The` as a determiner (DT), `cat` as a noun (NN), and so on. CRFs help you do this efficiently by considering not only the individual words but also the relationships between them.
     - Model label dependencies as `P(Y|X)`, with `Y` as output labels and `X` as input data.
     - Utilize feature functions to learn these relationships during training.
     - Predict label sequences for new data during inference.

``` Python 
!pip install sklearn-crfsuite
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Training data
X_train = [
   [('The', 'DT'), ('cat', 'NN'), ('sat', 'VBD')],
   [('A', 'DT'), ('dog', 'NN'), ('barked', 'VBD')]
]
y_train = [['DT', 'NN', 'VBD'], ['DT', 'NN', 'VBD']]

# Test data
X_test = [[('The', 'DT'), ('dog', 'NN'), ('barked', 'VBD')]]
y_test = [['DT', 'NN', 'VBD']]

# Create a CRF model
crf = sklearn_crfsuite.CRF(
   algorithm='lbfgs',
   c1=0.1,
   c2=0.1,
   max_iterations=100,
   all_possible_transitions=True
)

# Train the model
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
print(y_pred)
```
```
[['DT', 'NN', 'VBD']]
```

### Representation Learning in NLP
- Representation learning in the context of Natural Language Processing (NLP) is the process of automatically discovering and encoding the features of text data into numerical vectors that capture the semantic and syntactic properties of the text. These representations make it easier for machine learning models to process and understand the text for various tasks such as classification, translation, and sentiment analysis.
- **Encoding:** In NLP, encoding is the process of converting text into a different format for processing. For example, converting characters into numerical codes (like ASCII or UTF-8). This is crucial for machines to read and process text data. An example is encoding the word `hello` into its ASCII values: `104, 101, 108, 108, 111`.
- **Embedding:** In NLP, embedding is the process of mapping words or phrases into dense vectors in a lower-dimensional space. For instance, Word2Vec transforms the word `king` into a vector like `[0.25, 0.8, -0.5, ...]`, capturing its semantic meaning. Embeddings allow models to understand and work with the semantic relationships between words, enhancing tasks like text classification, sentiment analysis, and machine translation.

| Feature             | Encoding                                                | Embedding                                               |
| :------------------ | :------------------------------------------------------ | :------------------------------------------------------ |
| **Definition**      | Convert text to numerical forms (e.g., one-hot, TF-IDF). | Represent words in continuous vector space.            |
| **Purpose**         | Prepare text for ML models.                              | Capture semantic meanings in text.                     |
| **Example**         | One-hot, TF-IDF.                                         | Word2Vec, GloVe, BERT.                                 |
| **Dimensionality**  | Fixed (vocabulary size).                                 | Variable (semantic dimensionality).                    |
| **Data Type**       | Discrete (words/characters).                            | Continuous (semantic vectors).                         |
| **Transformation**  | Tokenization, vectorization.                             | Trained on large text corpora with NNs.                |
| **Representation**  | Numeric/sparse vectors.                                  | Dense vectors capture semantic meanings.               |
| **Application**     | Classification, sentiment analysis.                     | Semantic search, language modeling.                    |
| **Complexity**      | Simple algorithms.                                      | Complex models, resource-intensive.                    |
| **Information Preservation** | Preserves basic structure.                          | Retains semantic and contextual info.                  |
| **Examples**        | Bag-of-Words, TF-IDF, Integer Encoding.                                   | Word2Vec, GloVe,FasText. ELMo.                                 |
| **Usage in NLP**    | Preprocessing text data.                                | Essential for understanding text semantics.            |
| **Representation Type** | Results in sparse vectors.                           | Dense vectors capture semantic meanings and context.    |

- **One Hot Encoding:** One hot encoding is a technique used to represent categorical variables as binary vectors. Each unique category is represented by a binary vector where only one element is 1 and all others are 0.
   - Consider a dataset containing information about countries and their official languages:
      - **Countries**: USA, France, Germany, Japan, India
      - **Official Languages**: English, French, German, Japanese, Hindi
    - **Step 1:** We identify the unique categories in the `Official Language` column: English, French, German, Japanese, and Hindi.
    - **Step 2:** Create Binary Vectors
         - For each unique category, we create a binary vector:
             - English:    [1, 0, 0, 0, 0]
             - French:     [0, 1, 0, 0, 0]
             - German:     [0, 0, 1, 0, 0]
             - Japanese:   [0, 0, 0, 1, 0]
             - Hindi: [0, 0, 0, 0, 1]
  
   - **Step 3:** Assign Values
        - Now, we assign these binary vectors to each country based on their official language:
             - USA:        [1, 0, 0, 0, 0]
             - France:     [0, 1, 0, 0, 0]
             - Germany:    [0, 0, 1, 0, 0]
             - Japan:      [0, 0, 0, 1, 0]
             - India:     [0, 0, 0, 0, 1]
- One hot encoding is a useful technique for converting categorical data into a format that is suitable for machine learning algorithms. It ensures that each category is represented uniquely without introducing any ordinal relationships between categories.
   - Advantages:
      - 1. Simple to implement.
      - 2. Preserves all categorical data.
   - Disadvantages:
      - 1. Increases dimensionality.
      - 2. Higher computational load.
      - 3. Ignores ordinal relationships.
      - 4. May introduce multicollinearity.

``` Python 
def one_hot_encoding(categories, data):
    category_to_vector = {category: [0] * len(categories) for category in categories}
    
    for i, category in enumerate(categories):
        category_to_vector[category][i] = 1
    
    encoded_data = [category_to_vector[datum] for datum in data]
    return encoded_data

# Input 
countries = ['USA', 'France', 'Germany', 'Japan', 'India']
languages = ['English', 'French', 'German', 'Japanese', 'Hindi']
official_languages = ['English', 'French', 'German', 'Japanese', 'Hindi']

binary_vectors = one_hot_encoding(languages, official_languages)
encoded_countries = one_hot_encoding(languages, official_languages)
# Results
for country, encoded_vector in zip(countries, encoded_countries):
    print(f"{country}: {encoded_vector}")
```
```
USA: [1, 0, 0, 0, 0]
France: [0, 1, 0, 0, 0]
Germany: [0, 0, 1, 0, 0]
Japan: [0, 0, 0, 1, 0]
India: [0, 0, 0, 0, 1]
```
- **Integer Encoding:** Integer encoding is a technique used to represent categorical variables as integer values. It assigns a unique integer to each category. For instance, in a dataset of countries and their official languages:
   - **Steps:**
      1. **Assign integers to each unique category:**
         - English: 0
         - French: 1
         - German: 2
         - Japanese: 3
         - Hindi: 4

     2. **Map countries to their corresponding integer values:**
         - USA: 0
         - France: 1
         - Germany: 2
         - Japan: 3
         - India: 4
   - Advantages:
      - 1. Simple to implement.
      - 2. Memory efficiency as compared to one hot encoding uses less memory 
   - Disadvantages:
      - 1. Ordinal Misinterpretation
      - 2. Loss of Information
      - 3. Not Suitable for High Cardinality

``` Python 
from collections import defaultdict
def integer_encoding(data):
   categories = sorted(set(data))
   category_to_int = {cat: i for i, cat in enumerate(categories)}
   return [category_to_int[d] for d in data]

# Input 
languages = ['English', 'French', 'German', 'Japanese', 'Hindi']
encoded = integer_encoding(languages)
print(encoded)
```
```
[0, 1, 2, 4, 3]
```

- **Bag of words:** 



     
## Information Retrieval 

## Vector Search 

## LLMs 

## RAG 
