## Preprocessing 

--- 
### Case Folding 
   Case folding/lowercasing is a preprocessing technique in Natural Language Processing (NLP) that standardizes the text by converting all characters to a single case, typically lowercase. This step is essential for various NLP tasks as it ensures uniformity and consistency in text data, thereby enhancing the performance of downstream applications.
   - **For Example**
      - `Artificial Intelligence` becomes `artificial intelligence`
      - `Data Science` becomes `data science`

     
``` Python
text = "Machine Learning is FUN!"
lowercased_text = text.lower()
print(lowercased_text)  # Output: "machine learning is fun!" 
 ```
    
 ---
### Contraction Mapping
  Contraction mapping refers to the process of expanding contractions, which are shortened forms of words or phrases, into their complete versions. For example:
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

  ---
### Correcting Spelling Errors

  Correcting spelling errors in Natural Language Processing (NLP) is a common task aimed at improving text quality and accuracy. This process involves identifying and replacing misspelled words with their correct counterparts using various techniques such as:
- #### 1. **Dictionary-based approaches:**
  Utilizing a dictionary to look up correct spellings and suggest replacements for misspelled words.
- #### 2. **Edit distance algorithms:**
  Calculating the distance between words using metrics like `Levenshtein / Minimum Edit` distance to find the most likely correct spelling.
- #### 3. **Rule-based methods:**
  Applying spelling correction rules and heuristics to identify and correct common spelling mistakes.
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

--- 

### Deduplication / Duplicate Removal:
  Deduplication in the context of Natural Language Processing (NLP) involves identifying and removing duplicate entries in a dataset. This process is crucial for ensuring data quality and accuracy, especially when dealing with large text corpora.
- #### 1. **Using Pandas for Deduplication:**
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

--- 


#### 2. Using Fuzzy Matching for Deduplication
  Fuzzy matching in NLP refers to the process of finding strings that are approximately equal to a given pattern. It is particularly useful in scenarios where exact matches are not possible due to typographical errors, variations in spelling, or other inconsistencies in the text data. Fuzzy matching is widely used in applications like data deduplication, record linkage, and spell-checking.
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

--- 


### Expanding Abbreviations and Acronyms
  Expanding abbreviations and acronyms is an important task in Natural Language Processing (NLP) to enhance the understanding and processing of text. Here are some key methods and approaches used to achieve this:
- #### 1. **Dictionary-Based Methods:**
  These methods involve using precompiled lists of abbreviations and their expansions. The dictionary can be curated manually or generated from various sources such as online databases or domain-specific corpora.
- #### 2. **Rule-Based Methods:**
  These methods use linguistic rules and patterns to identify and expand abbreviations. For example, context-specific rules can be applied based on the position of the abbreviation in the text or its surrounding words.
- #### 3. **Statistical Methods:**
  These methods rely on statistical models and machine learning algorithms to predict expansions based on large corpora. Techniques include: N-gram models, Hidden Markov Models (HMMs) and Conditional Random Fields(CRFs)
  - ####  **Simple Dictionary-based Implementation**
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

 --- 
 
### Stemming
  Stemming in Natural Language Processing (NLP) refers to the process of reducing words to their base or root form, known as the `stem`, by removing prefixes and suffixes. The stem may not always be a valid word in the language, but it represents the core meaning of the word, thereby helping to group similar words. Types of Stemming Algorithms: 
 - #### 1. **Porter Stemmer**
 - #### 2. **Snowball Stemmer**
 - #### 3. **Lancaster Stemmer**
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

--- 


### Lemmatization
  Lemmatization is another crucial text normalization technique in Natural Language Processing (NLP) that involves reducing words to their base or dictionary form, known as the "lemma." Unlike stemming, which simply chops off affixes to obtain the root form, lemmatization considers the context of the word and ensures that the resulting base form is a valid word in the language. Lemmatization algorithms rely on linguistic rules and lexical resources to map inflected words to their base or dictionary forms.
- #### 1. **Rule-Based Lemmatization:**
  Rule-based lemmatization algorithms rely on linguistic rules and patterns to derive lemmas from inflected forms. These rules are often derived from linguistic knowledge and may vary based on language and context.
- #### 2. **Lexical Resource-based Lemmatization:**
  WordNet is a lexical database of the English language that includes information about word meanings, relationships between words, and word forms. Lemmatization algorithms leveraging WordNet typically use its morphological information and lexical relationships to derive lemmas.
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

--- 


### Noise Removing
  Noise removal in NLP involves eliminating irrelevant or unwanted elements, such as HTML tags, special characters, punctuation, stop words, and numerical values, from text data. This process aims to clean and standardize the data, making it more suitable for analysis or model training. The goal of noise removal is to clean the text data by stripping away these elements while preserving the meaningful content. This typically involves a series of preprocessing steps, which may include:
- #### 1. **Stripping HTML Tags:**
  Removing HTML markup from text obtained from web sources.
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
#### 2. Removing Special Characters
  Eliminating non-alphanumeric characters, punctuation marks, and symbols.
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
#### 3. Removing Stop Words
  Eliminating non-alphanumeric characters, punctuation marks, and symbols.
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
#### 4. Removing Numerical Values
  Eliminating digits and numbers that may not be relevant to the analysis.
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
#### 5. Handling Emojis and Emoticons
  Removing or replacing emojis and emoticons with descriptive text.
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
#### 6. Removing Non-Linguistic Symbols
  Eliminating symbols or characters that do not convey linguistic meaning, such as currency symbols, mathematical operators, or trademark symbols.
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

--- 

### Tokenization

Tokenization is the process of splitting text into smaller, meaningful units called tokens. These tokens can represent words, subwords, or characters and are foundational elements used in various NLP tasks like text analysis, machine translation, and sentiment analysis. The primary types of tokenization include:

---

#### 1. Word Tokenization

Word tokenization involves breaking text into individual words. It's the most intuitive form of tokenization but can be challenging for languages without clear word boundaries or texts with contractions and special characters.

- **Example:** 
  - `Tokenization is fun!` is tokenized into [`Tokenization`, `is`, `fun`, `!`].

- **Advantages:**
  - Simple and intuitive.
  - Works well for languages with clear word boundaries.
  
- **Disadvantages:**
  - Struggles with out-of-vocabulary words.
  - Difficulty handling contractions and special characters.

---

#### 2. Subword Tokenization

Subword tokenization divides text into smaller units than words, which helps handle out-of-vocabulary words and morphological variations. Popular subword tokenization techniques include Byte Pair Encoding (BPE), WordPiece, Unigram, and SentencePiece.

---

##### **Byte Pair Encoding (BPE)**

Byte Pair Encoding (BPE) is a subword tokenization method that iteratively merges the most frequent pair of bytes or characters in a corpus to form subword units.

- **Advantages:**
  - Reduces vocabulary size.
  - Handles out-of-vocabulary words effectively.
  - Simple and efficient for various NLP tasks.
  
- **Disadvantages:**
  - May not capture all linguistic nuances.
  - Can produce subwords that are not linguistically meaningful.

- **Models Using BPE:**
  - GPT, GPT-2, RoBERTa, BART, DeBERTa

- **Steps:**
  1. **Initialize Vocabulary:** Start with a vocabulary of all unique characters in the text.
  2. **Count Frequencies:** Count the frequency of each adjacent character pair in the text.
  3. **Merge Pairs:** Find and merge the most frequent pair into a new token.
  4. **Replace Pairs:** Replace all instances of this pair in the text with the new token.
  5. **Repeat:** Repeat steps 2-4 for a predefined number of merges or until the desired vocabulary size is achieved.

---

##### **WordPiece Tokenization**

WordPiece is a subword tokenization method originally developed for speech recognition and later adopted by NLP models like BERT. It breaks down words into smaller, more frequent subword units.

- **Advantages:**
  - Efficiently handles rare words.
  - Addresses the Out-Of-Vocabulary (OOV) problem.
  
- **Disadvantages:**
  - May result in subwords that are not intuitively meaningful.
  - Requires extensive training data to build an effective vocabulary.

- **Models Using WordPiece:**
  - BERT, DistilBERT, Electra

- **Steps:**
  1. **Initialize Vocabulary:** Start with an initial vocabulary, typically consisting of all individual characters and some predefined words from the training corpus.
  2. **Compute Frequencies:** Compute the frequency of all substrings (subword units) in the corpus.
  3. **Merge Substrings:** Iteratively merge the most frequent substring pairs to form new subwords until the vocabulary reaches a predefined size.
  4. **Tokenize Text:** Tokenize new text by matching the longest possible subwords from the vocabulary.

---

##### **Unigram Tokenization**

Unigram Tokenization is a subword tokenization method that treats each character as a token.

- **Advantages:**
  - Simple and language-agnostic.
  - Effective for languages with complex morphology like Japanese or Turkish.
  
- **Disadvantages:**
  - May not capture linguistic meaning effectively.
  - Can result in a large number of tokens for longer texts.

- **Steps:**
  1. **Tokenization:** Break down the text into individual characters. Each character becomes a separate token.
  2. **Vocabulary Construction:** Build a vocabulary containing all unique characters present in the text.

---

##### **SentencePiece Tokenization**

SentencePiece is an unsupervised text tokenizer and detokenizer that creates subword units without relying on predefined word boundaries.

- **Advantages:**
  - Flexible and language-agnostic.
  - Reduces out-of-vocabulary issues.
  
- **Disadvantages:**
  - Computationally intensive during training.
  - May generate subwords that are not intuitively meaningful.

- **Models Using SentencePiece:**
  - BERT, T5, GPT

- **Steps:**
  1. **Data Preparation:** Collect and preprocess the text corpus.
  2. **Normalization:** Standardize the text.
  3. **Model Training:**
     - **BPE:**
       1. Calculate frequencies of adjacent token pairs.
       2. Merge the most frequent pair into a new token.
       3. Repeat until the desired vocabulary size is reached.
     - **Unigram:**
       1. Start with many subwords.
       2. Assign probabilities and prune the least probable subwords.
       3. Repeat until the desired vocabulary size is reached.
  4. **Save Model:** Output the trained model and vocabulary.
  5. **Tokenization:** Use the model to tokenize new text into subwords.

---

#### Comparison Table

| Tokenization Type     | Advantages                                        | Disadvantages                                     | Example Models                      |
|-----------------------|---------------------------------------------------|---------------------------------------------------|-------------------------------------|
| **Word Tokenization** | Simple, intuitive                                 | Struggles with out-of-vocabulary words, contractions, and special characters | -                                   |
| **Byte Pair Encoding**| Reduces vocabulary size, handles out-of-vocabulary words effectively | May not capture all linguistic nuances            | GPT, GPT-2, RoBERTa, BART, DeBERTa  |
| **WordPiece**         | Efficiently handles rare words, addresses OOV problem | May result in non-intuitive subwords, requires extensive training data | BERT, DistilBERT, Electra           |
| **Unigram**           | Simple, language-agnostic, effective for complex morphology | May not capture linguistic meaning, large number of tokens for longer texts | -                                   |
| **SentencePiece**     | Flexible, language-agnostic, reduces OOV issues   | Computationally intensive during training, may generate non-intuitive subwords | BERT, T5, GPT                       |

---

For a more detailed explanation, check out this [Huggingface video](https://huggingface.co/docs/transformers/en/tokenizer_summary).


--- 

##  Statical NLP 


- ### **Naive Bayes**:
  Naive Bayes presents a straightforward yet effective classification approach rooted in `Bayes theorem`, assuming `independence` among features. Here's a simplified rundown:
  - Bayes theorem is a cornerstone of probability theory, revealing the probability of an event based on prior conditions. It's expressed as:
    $$P(A|B) = (P(B|A) * P(A)) / P(B)$$
 - Where, 
     - $P(A|B)$ is the probability of event $A$ occurring given that event $B$ has occurred.
     - $P(B|A)$ is the probability of event $B$ occurring given that event $A$ has occurred.
     - $P(A)$ and $P(B)$ are the probabilities of events $A$ and $B$ occurring independently of each other.
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

--- 

### N-gram language model
  An n-gram is a sequence of $n$ items from a given sample of text or speech. The `items` are typically words or characters, and the sequence can be as short or as long as needed:
    - Unigram $(n=1)$: Single word sequences.
    - Bigram $(n=2)$: Pairs of words.
    - Four-gram $(n=4)$ and higher: Longer sequences.
  - For example, with the sentence `I love natural language processing`:
     - Unigrams: [`I`, `love`, `natural`, `language`, `processing`]
     - Bigrams: [`I love`, `love natural`, `natural language`, `language processing`]
     - Trigrams: [`I love natural`, `love natural language`, `natural language processing`]
-  N-gram models predict the likelihood of a word given the preceding $n-1$ words. The core idea is to estimate the probability of the next word in a sequence, given the previous words. Using the chain rule of probability:
$$P(w_n | w_1, w_2, ..., w_{n-1}) = C(w_1, w_2, ..., w_n) / C(w_1, w_2, ..., w_{n-1})$$
- Where,
  - $P(w_n | w_1, w_2, ..., w_{n-1})$ represents the probability of the word $w_n$ occurring after the sequence of words $w_1, w_2, ..., w_{n-1}$
  - $C(w_1, w_2, ..., w_n)$ is the count of the n-gram $(w_1, w_2, ..., w_n)$ in the training corpus.
  - $C(w_1, w_2, ..., w_{n-1})$ is the count of the $(n-1)$-gram $(w_1, w_2, ..., w_{n-1})$ in the training corpus.

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

--- 

### Markov Chain
  A Markov Chain is a way to model a system where the probability of moving to the next state depends only on the current state.
  - Components of Markov Chain:
      - #### 1. **States:**
        Different conditions or situations the system can be in. For example, the weather can be either Sunny or Rainy.
        - A set of states $S = {s1, s2, …, sn}$.

      - #### 2. **Transition Probabilities:**
        The chances of moving from one state to another.
        - A transition probability matrix $P = [pij]$, where $pij = P(Xt+1 = sj | Xt = si)$.

- **Hidden Markov Model (HMM):**
  A Hidden Markov Model (HMM) is a statistical model where the system being modelled is assumed to follow a Markov process with hidden states. In contrast to Markov Chains, in HMMs, the state is not directly visible, but output dependent on the state is visible.
  - #### **Components:**
     - **Initial State Distribution:** Probabilities of starting in each hidden state.
     - **Hidden States:** A finite set of states that are not directly observable.
     - **Observable States:** A finite set of states that can be directly observed.
     - **Transition Probabilities:** Probabilities of transitioning between hidden states.
     - **Emission Probabilities:** Probabilities of an observable state being generated from a hidden state.
 - #### **An HMM can be characterized by:**
     - Initial state probabilities $π = [πi]$, where $πi = P(X1 = si)$.
     - A set of hidden states $S = {s1, s2, …, sn}$.
     - A set of observable states $O = {o1, o2, …, om}$.
     - Transition probabilities $A = [aij]$, where $aij = P(Xt+1 = sj | Xt = si)$.
     - Emission probabilities $B = [bjk]$, where $bjk = P(Yt = ok | Xt = sj)$.
- #### **Key Algorithms:**
    - **Forward Algorithm:** Computes the likelihood of observing a sequence of symbols.
    - **Viterbi Algorithm:** Finds the most likely sequence of hidden states based on observations.
    - **Baum-Welch Algorithm:** Trains Hidden Markov Models (HMMs) by estimating transition and emission probabilities from observed data.
- #### **Applications:**
  Applications of Markov Chains and HMM are Part of Speech tagging, Speech Recognition, Name Entity Recognition etc. 

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

--- 

### Conditional Random Fields (CRFs)

Conditional Random Fields (CRFs) are a type of probabilistic graphical model, specifically used for structured prediction problems. They are particularly effective in applications where context and sequential information are crucial, such as natural language processing tasks like named entity recognition (NER), part-of-speech tagging, and information extraction.

#### Key Concepts:

- **Graphical Model:** CRFs are undirected graphical models used to calculate the conditional probability of a label sequence given an observation sequence. They can be visualized as a graph where nodes represent random variables (labels and observations) and edges represent dependencies between them.

- **Sequence Modeling:** Unlike simpler models like Naive Bayes or HMMs, CRFs consider the context of the entire sequence when making predictions. This allows CRFs to model the dependencies between neighboring labels more effectively.

#### Mathematical Formulation:

Given an input sequence of observations $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ and a corresponding sequence of labels $\mathbf{y} = (y_1, y_2, \ldots, y_T)$, the CRF model calculates the conditional probability $P(\mathbf{y}|\mathbf{x})$.

The probability of a particular label sequence given the observations is defined as:

$$P(\mathbf{y}|\mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\left( \sum_{t=1}^{T} \sum_{k=1}^{K} \lambda_k f_k(y_t, y_{t-1}, \mathbf{x}, t) \right)$$

where:
- $\mathbf{y}$ is the sequence of labels.
- $\mathbf{x}$ is the sequence of observations.
- $f_k$ are the feature functions.
- $\lambda_k$ are the weights associated with the feature functions.
- $Z(\mathbf{x})$ is the normalization factor (partition function) ensuring that the probabilities sum to 1.

#### Feature Functions:

Feature functions $f_k(y_t, y_{t-1}, \mathbf{x}, t)$ can depend on the current and previous labels, the entire sequence of observations, and the current position in the sequence. These functions help capture the characteristics of the data and dependencies between labels.

#### Training CRFs:

Training CRFs involves optimizing the weights $\lambda_k$ to maximize the likelihood of the training data. This is typically done using gradient-based optimization methods. The objective is to find the set of weights that best explain the observed data by maximizing the conditional log-likelihood:

$$\mathcal{L}(\lambda) = \sum_{i=1}^{N} \log P(\mathbf{y}^{(i)}|\mathbf{x}^{(i)}; \lambda) - \frac{1}{2\sigma^2} \sum_{k} \lambda_k^2$$

where $\sigma^2$ is a regularization parameter to prevent overfitting.

#### Inference:

The goal of inference in CRFs is to find the most likely sequence of labels $\mathbf{y}$ given an observation sequence $\mathbf{x}$. This is typically done using algorithms such as the Viterbi algorithm or belief propagation.

#### Applications:

- **Named Entity Recognition (NER):** Identifying entities like names, organizations, and locations in text.
- **Part-of-Speech Tagging:** Assigning parts of speech to each word in a sentence.
- **Chunking:** Dividing a text into syntactically correlated parts like noun or verb phrases.
- **Information Extraction:** Extracting structured information from unstructured text.

#### Example:

Here's an example of implementing CRFs for sequence labeling using the `sklearn-crfsuite` library in Python:

```python
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Example data
train_sents = [[{'word': 'John', 'pos': 'NNP'}, {'word': 'is', 'pos': 'VBZ'}, {'word': 'from', 'pos': 'IN'}, {'word': 'New', 'pos': 'NNP'}, {'word': 'York', 'pos': 'NNP'}]]
train_labels = [['B-PER', 'O', 'O', 'B-LOC', 'I-LOC']]

def word2features(sent, i):
    word = sent[i]['word']
    postag = sent[i]['pos']
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'postag': postag,
    }
    if i > 0:
        word1 = sent[i-1]['word']
        postag1 = sent[i-1]['pos']
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1]['word']
        postag1 = sent[i+1]['pos']
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:postag': postag1,
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for label in sent]

X_train = [sent2features(s) for s in train_sents]
y_train = train_labels

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

test_sents = [[{'word': 'Jane', 'pos': 'NNP'}, {'word': 'is', 'pos': 'VBZ'}, {'word': 'from', 'pos': 'IN'}, {'word': 'Los', 'pos': 'NNP'}, {'word': 'Angeles', 'pos': 'NNP'}]]
X_test = [sent2features(s) for s in test_sents]
y_pred = crf.predict(X_test)

print(y_pred)
```
---

## Representation Learning in NLP

Representation learning in the context of NLP is the process of automatically discovering and encoding the features of text data into numerical vectors that capture the semantic and syntactic properties of the text. These representations make it easier for machine learning models to process and understand the text for various tasks such as classification, translation, and sentiment analysis.

### Encoding/Sparse Vectors

In NLP, encoding is the process of converting text into a different format for processing. For example, converting characters into numerical codes (like ASCII or UTF-8). This is crucial for machines to read and process text data. An example is encoding the word `hello` into its ASCII values: `104, 101, 108, 108, 111`.

---

#### One Hot Encoding

One hot encoding is a technique used to represent categorical variables as binary vectors. Each unique category is represented by a binary vector where only one element is 1 and all others are 0.

- Consider a dataset containing information about countries and their official languages:
  - **Countries**: USA, France, Germany, Japan, India
  - **Official Languages**: English, French, German, Japanese, Hindi

  **Step 1:** Identify the unique categories in the `Official Language` column: English, French, German, Japanese, and Hindi.

  **Step 2:** Create Binary Vectors
  - For each unique category, create a binary vector:
    - English: `[1, 0, 0, 0, 0]`
    - French: `[0, 1, 0, 0, 0]`
    - German: `[0, 0, 1, 0, 0]`
    - Japanese: `[0, 0, 0, 1, 0]`
    - Hindi: `[0, 0, 0, 0, 1]`
  
  **Step 3:** Assign Values
  - Assign these binary vectors to each country based on their official language:
    - USA: `[1, 0, 0, 0, 0]`
    - France: `[0, 1, 0, 0, 0]`
    - Germany: `[0, 0, 1, 0, 0]`
    - Japan: `[0, 0, 0, 1, 0]`
    - India: `[0, 0, 0, 0, 1]`

One hot encoding is a useful technique for converting categorical data into a format that is suitable for machine learning algorithms. It ensures that each category is represented uniquely without introducing any ordinal relationships between categories.

- **Advantages**:
  1. Simple to implement.
  2. Preserves all categorical data.

- **Limitations**:
  1. Increases dimensionality.
  2. Higher computational load.
  3. Ignores ordinal relationships.
  4. May introduce multicollinearity.

```python
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

#### Integer Encoding

Integer encoding is a technique used to represent categorical variables as integer values. It assigns a unique integer to each category. For instance, in a dataset of countries and their official languages:

- **Steps:**
  1. **Assign integers to each unique category:**
     - English: $0$
     - French: $1$
     - German: $2$
     - Japanese: $3$
     - Hindi: $4$

  2. **Map countries to their corresponding integer values:**
     - USA: $0$
     - France: $1$
     - Germany: $2$
     - Japan: $3$
     - India: $4$

- **Advantages:**
  1. Simple to implement.
  2. Memory efficient as compared to one-hot encoding, which uses more memory.

- **Limitations:**
  1. Ordinal misinterpretation.
  2. Loss of information.
  3. Not suitable for high cardinality.

```python
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

--- 
#### Bag of Words

The Bag of Words (BoW) model in NLP converts text into numerical vectors by creating a vocabulary of unique words from a corpus and representing each document by a vector of word frequencies. This method is simple and effective for tasks like text classification and clustering, though it ignores grammar, word order, and context, leading to potential loss of information and high-dimensional, sparse vectors. Despite its limitations, BoW is popular due to its ease of use and effectiveness.

- **Process Steps:**
  - **Corpus Collection:**
    Gathers a comprehensive set of text documents to form the corpus, laying the groundwork for analysis and modeling.
    - I love reading books. Books are great.
    - Books are a wonderful source of knowledge.
    - I have a great love for reading books.
    - Reading books can be very enlightening. Books are amazing.
    
  - **Preprocessing:**
    Executes meticulous text preprocessing tasks, including lowercasing, punctuation removal, and stop word elimination, to maintain standardized data quality.
    - i love reading books books are great
    - books are a wonderful source of knowledge
    - i have a great love for reading books
    - reading books can be very enlightening books are amazing
    
  - **Vocabulary Building:**
    Extracts unique words from the corpus, forming the foundational vocabulary that encompasses diverse linguistic elements.
    - Vocabulary: [`i`, `love`, `reading`, `books`, `are`, `great`, `a`, `wonderful`, `source`, `of`, `knowledge`, `have`, `for`, `can`, `be`, `very`, `enlightening`, `amazing`]
    
  - **Vectorization:**
    Transforms each document into a numerical vector representation based on the established vocabulary. Ensures vector length matches vocabulary size, with elements representing word frequencies, succinctly capturing the document's textual essence.
    - Sentence 1: i  love  reading  books  books  are  great
      Vector: $[1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]$
    - Sentence 2: books  are  a  wonderful  source  of  knowledge
      Vector: $[0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]$
    - Sentence 3: i  have  a  great  love  for  reading  books
      Vector: $[1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]$
    - Sentence 4: reading  books  can  be  very  enlightening  books  are  amazing
      Vector: $[0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]$

- **Advantages:**
  1. Simple to implement.
  2. Efficient conversion of text to numerical data.
  3. Effective for basic text classification and clustering.

- **Limitations:**
  1. Loss of context.
  2. High dimensionality and sparse vectors.
  3. Sparse data representations may pose challenges for some algorithms.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love reading books. Books are great.",
    "Books are a wonderful source of knowledge.",
    "I have a great love for reading books.",
    "Reading books can be very enlightening. Books are amazing."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vocabulary = vectorizer.get_feature_names_out()

for i, document in enumerate(corpus):
    print(f"Sentence {i + 1}: {document}")
    print(f"Vector: {X[i].toarray().tolist()[0]}")
    print()

print("Vocabulary:", vocabulary)
```

--- 


#### TF-IDF

TF-IDF is a numerical statistic used in information retrieval and text mining. It reflects the importance of a word in a document relative to a collection of documents (corpus). TF-IDF is often used as a weighting factor in search engine algorithms and text analysis.

- #### **Components of TF-IDF:**

  - #### **Term Frequency (TF):** \
    Measures how frequently a term occurs in a document.
    - Term Frequency is calculated as:
      $$\text{TF}(t, d) = \frac{f(t, d)}{\sum_{t' \in d} f(t', d)}$$
    - where:
      - $f(t,d)$ is the raw count of term $t$ in document $d$.
      - The denominator is the total number of terms in document $d$.
    - Example:
      - If the term `data` appears $3$ times in a document with $100$ words, the term frequency TF for `data` would be: $TF(data, d) = 3 / 100 = 0.03$

  - #### **Inverse Document Frequency (IDF):**
    Measures how frequently a term occurs in a document.
    - Inverse Document Frequency is calculated as:
      $$\text{IDF}(t, D) = \log \left( \frac{N}{| \{ d \in D : t \in d \} |} \right)$$
    - where:
      - $N$ is the total number of documents.
      - $| \{ d \in D : t \in d \} |$ is the number of documents containing the term $t$.
    - Example:
      - If the corpus contains $10,000$ documents, and the term `data` appears in $100$ of these documents, the inverse document frequency IDF for `data` would be:
      - $IDF(data, D) = \log(10000 / 100) = \log(100) = 2$

  - #### **Calculating TF-IDF:**
    - The TF-IDF score for a term $t$ in a document $d$ is given by:
      $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
    - Example:
      - Using the previous values:
      - $TF(data, d) = 0.03$
      - $IDF(data, D) = 2$
      - The TF-IDF score for `data` in the document would be:
      - $TF-IDF(data, d, D) = 0.03 * 2 = 0.06$

- Advantages:
  - 1. Simple and easy to understand
  - 2. Effective in identifying relevant terms and their weights
  - 3. Distinguishes between common and rare terms
  - 4. Language-independent

- Limitations:
  - 1. Doesn't consider semantics or term context
  - 2. May struggle with very long documents due to term frequency saturation. In lengthy documents, even insignificant terms can surface frequently, resulting in elevated and saturated term frequencies. Consequently, TF-IDF may encounter challenges in effectively distinguishing and assigning significant weights to important terms.
  - 3. Ignores term dependencies and phrase
  - 4. Needs large document collections for reliable IDF

``` python
from collections import Counter

def calculate_tf_idf(corpus):
    tfidf_scores = {}
    doc_count = len(corpus)

    for doc_index, document in enumerate(corpus, 1):
        term_counts = Counter(document.split())
        total_terms = sum(term_counts.values())

        doc_scores = {}
        for term, frequency in term_counts.items():
            tf = frequency / total_terms
            doc_freq = sum(1 for doc in corpus if term in doc)
            idf = 1 + (doc_count / doc_freq) if doc_freq > 0 else 1  # Here +1 is smoothing factor to avoid division by zero 
            doc_scores[term] = tf * idf

        tfidf_scores[doc_index] = doc_scores
    return tfidf_scores

# Input
corpus = [
    "This is document 1. It contains some terms.",
    "Document 2 has different terms than document 1.",
    "Document 3 is another example document with some common terms."
]
tfidf_scores = calculate_tf_idf(corpus)
for doc_index, doc_scores in tfidf_scores.items():
    print(f"Document {doc_index}:")
    for term, score in doc_scores.items():
        print(f"{term}: {score:.3f}")

```

--- 

#### BM25 (Best Matching 25)
  BM25 (Best Matching 25) is a ranking function used in information retrieval to estimate the relevance of documents to a given search query. It is an extension of the TF-IDF weighting scheme, designed to address some of its limitations while retaining its simplicity and effectiveness.
   - #### **Components of BM25:**
      - #### **Term Frequency (TF):** Measures how frequently a term occurs in a document, similar to TF in TF-IDF.
         - TF in BM25 is adjusted to handle document length normalization and term saturation. It is calculated as:
           $$\text{TF}(t, d) = \frac{f(t, d) \cdot (k + 1)}{f(t, d) + k \cdot (1 - b + b \cdot (|d| / \text{avgdl}))}$$
         - Where
           - $f(t,d)$ is the raw count of term $t$ in document $d$.
           - $|d|$ is the length of document $d$.
           - $\text{avgdl}$ is the average document length in the corpus.
           - $k$ and $b$ are tuning parameters, typically set to $1.5$ and $0.75$ respectively.
     
      - #### **Inverse Document Frequency (IDF):**
        Measures how frequently a term occurs in the entire document collection, similar to IDF in TF-IDF.
         - IDF in BM25 is calculated as:
           $$\text{IDF}(t, D) = \log \left( \frac{N - n(t) + 0.5}{n(t) + 0.5} \right)$$
         - Where
            - $N$ is the total number of documents in the collection.
            - $n(t)$ is the number of documents containing term $t$.
      - **Document length Normalization:** Adjusts the TF component based on the length of the document. This ensures that longer documents do not have an unfair advantage over shorter ones.
   
- #### **BM25 Score Calculation:**
  The BM25 score for a term t in a document d given a query q is calculated as:
      $$\text{BM25}(t, d, q) = \text{IDF}(t, D) \cdot \frac{f(t, d) \cdot (k + 1)}{f(t, d) + k \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$
    - Where:
       - $IDF(t, D)$ is the inverse document frequency for term $t$.
       - $f(t, d)$ is the term frequency for term $t$ in document $d$.
       - $|d|$ is the length of document $d$.
       - $\text{avgdl}$ is the average document length in the corpus.
       - $k$ and $b$ are tuning parameters.

   - Advantages:
      - 1. Effective in ranking documents based on relevance to a query.
      - 2. Accounts for document length normalization, addressing a limitation of TF-IDF.
      - 3. Suitable for large document collections.
      - 4. Robust and widely used in practice.

   - Limitations:
      - 1. Like TF-IDF, BM25 does not consider semantic relationships between terms.
      - 2. Tuning parameters (k and b) need to be carefully selected, although default values often work reasonably well.
      - 3. May still struggle with certain types of queries or document collections, such as those containing very short documents or highly specialized domains.

``` Python 
from collections import Counter
import math

# BM25 parameters
k = 1.5
b = 0.75

def calculate_bm25(corpus, k=1.5, b=0.75):
    bm25_scores = {}
    doc_count = len(corpus)
    avgdl = sum(len(doc.split()) for doc in corpus) / doc_count
    
    # Precompute document frequencies for each term
    doc_freq = {}
    for document in corpus:
        unique_terms = set(document.split())
        for term in unique_terms:
            if term not in doc_freq:
                doc_freq[term] = 0
            doc_freq[term] += 1
    
    # Compute BM25 scores
    for doc_index, document in enumerate(corpus, 1):
        term_counts = Counter(document.split())
        doc_length = sum(term_counts.values())

        doc_scores = {}
        for term, frequency in term_counts.items():
            tf = frequency
            df = doc_freq.get(term, 0)
            idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
            score = idf * ((tf * (k + 1)) / (tf + k * (1 - b + b * (doc_length / avgdl))))
            doc_scores[term] = score

        bm25_scores[doc_index] = doc_scores
    return bm25_scores

# Input
corpus = [
    "This is document 1. It contains some terms.",
    "Document 2 has different terms than document 1.",
    "Document 3 is another example document with some common terms."
]

bm25_scores = calculate_bm25(corpus, k, b)
for doc_index, doc_scores in bm25_scores.items():
    print(f"Document {doc_index}:")
    for term, score in doc_scores.items():
        print(f"{term}: {score:.3f}")
```
```
Document 1:
This: 1.016
is: 0.487
document: 0.138
1.: 0.487
It: 1.016
contains: 1.016
some: 0.487
terms.: 0.487
Document 2:
Document: 0.487
2: 1.016
has: 1.016
different: 1.016
terms: 1.016
than: 1.016
document: 0.138
1.: 0.487
Document 3:
Document: 0.440
3: 0.917
is: 0.440
another: 0.917
example: 0.917
document: 0.125
with: 0.917
some: 0.440
common: 0.917
terms.: 0.440
```

--- 
### Embedding\Dense Vectors

In NLP, embedding is the process of mapping words or phrases into dense vectors in a high-dimensional space. For instance, Word2Vec transforms the word `king` into a vector like $[0.25, 0.8, -0.5, \ldots]$, capturing its semantic meaning. Embeddings allow models to understand and work with the semantic relationships between words, enhancing tasks like text classification, sentiment analysis, and machine translation.

#### Embeddings in Representation Learning

In NLP, an `embedding` is a way of representing words, phrases, or even entire documents as continuous, dense vectors of numbers. These vectors capture the semantic meaning of the text so that words or phrases with similar meanings are represented by similar vectors.

- **Example:**
    - Consider the words `king`, `queen`, `man`, and `woman`. In a well-trained embedding space, these words might be represented by the following vectors (illustrative examples):
        - king = $[0.25, 0.75, 0.10, 0.60]$
        - queen = $[0.20, 0.80, 0.15, 0.65]$
        - man = $[0.30, 0.60, 0.05, 0.50]$
        - woman = $[0.25, 0.70, 0.10, 0.55]$
    - In this vector space, the embeddings for `king` and `queen` are closer to each other than to `man` and `woman`, capturing the relationship between royalty. Similarly, the difference between `king` and `man` is similar to the difference between `queen` and `woman`, capturing the gender relationship.

#### Word2Vec

Word2Vec is a popular technique in Natural Language Processing (NLP) that transforms words into numerical vectors, capturing their meanings and relationships. Developed by Tomas Mikolov and his team at Google in 2013, Word2Vec comes in two main models: Continuous Bag-of-Words (CBOW) and Skip-Gram.

- #### How Word2Vec Works:
    - **Continuous Bag-of-Words (CBOW):**
        The CBOW model predicts a target word based on its surrounding context words. Here’s how it works:
        - **Context Window:** Assume a context window of size $m$. For a given target word $w_t$, the context words are:
            $$w_{t-m}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+m}$$
        - **Input Representation:** Each word $w$ in the vocabulary $V$ is represented as a one-hot vector $x \in \mathbb{R}^{|V|}$, where only the index corresponding to $w$ is $1$ and all other indices are $0$.
        - **Projection Layer:** The input one-hot vectors are mapped to a continuous vector space using a weight matrix $W \in \mathbb{R}^{|V| \times d}$, where $d$ is the dimensionality of the word vectors (embeddings). The context word vectors are averaged:
            $$v_c = \frac{1}{2m} \sum_{i=-m, i \neq 0}^{m} W \cdot x_{t+i}$$
        - **Output Layer:** The averaged context vector $v_c$ is then multiplied by another weight matrix $W' \in \mathbb{R}^{d \times |V|}$ and passed through a softmax function to produce the probability distribution over the vocabulary:
            $$p(w_t | w_{t-m}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+m}) = \frac{\exp(v_{w_t} \cdot v_c)}{\sum_{w \in V} \exp(v_w \cdot v_c)}$$

    - **Skip-Gram:**
        The Skip-Gram model, on the other hand, predicts context words given the target word. The steps are:
        - **Input Representation:** For a target word $w_t$, represented as a one-hot vector $x_t$.
        - **Projection Layer:** The one-hot vector is projected into the embedding space using the weight matrix $W$:
            $$v_t = W \cdot x_t$$
        - **Output Layer:** For each context word $w_{t+i}$ (within the window of size $m$), the model predicts the probability distribution using the weight matrix $W'$ and softmax:
            $$p(w_{t+i} | w_t) = \frac{\exp(v_{w_{t+i}} \cdot v_t)}{\sum_{w \in V} \exp(v_w \cdot v_t)}$$

    - **Negative Sampling:**
        Negative sampling simplifies the training process by approximating the softmax function. Instead of computing the gradient over the entire vocabulary, negative sampling updates the weights for a small number of `negative` words (words not in the context). For each context word $w_O$ and target word $w_I$:
        $$\log \sigma(v_{w_O} \cdot v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_n \sim P_n(w)} [\log \sigma(-v_{w_n} \cdot v_{w_I})]$$
        where:
        - $\sigma$ is the sigmoid function
        - $k$ is the number of negative samples
        - $P_n(w)$ is the noise distribution

    - **Hierarchical Softmax:**
        Hierarchical softmax reduces computational complexity by representing the vocabulary as a binary tree. Each word is a leaf node, and predicting a word involves traversing from the root to the leaf node. The probability of a word is the product of the probabilities of the decisions made at each node in the path.

- **Advantages:**
    1. Efficient training on large datasets.
    2. Captures semantic similarities.
    3. Enables easy comparison of words.
    4. Handles large datasets.
    5. Flexible for task-specific fine-tuning.

- **Disadvantages\Limitations:**
    1. Ignores word order beyond a fixed window.
    2. Out-of-vocabulary words are not represented.
    3. Large embedding matrices can be memory-intensive.
    4. Static embeddings that don't adapt to different contexts within a document.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.layers import Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np

# Input text
text_data = [
    "Natural language processing and machine learning are fascinating fields.",
    "Word2Vec is an excellent tool for word embedding."
]

# Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)
vocab_size = len(tokenizer.word_index) + 1

# Generate skip-grams
skip_grams = [skipgrams(seq, vocab_size, window_size=2) for seq in sequences]

# Build the model
embedding_dim = 100
input_target = tf.keras.Input((1,))
input_context = tf.keras.Input((1,))
embedding = Embedding(vocab_size, embedding_dim, input_length=1)
target = embedding(input_target)
target = Flatten()(target)
context = embedding(input_context)
context = Flatten()(context)
dot_product = Dot(axes=1)([target, context])
output = Dense(1, activation='sigmoid')(dot_product)
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam')

# Prepare training data
pairs, labels = [], []
for skip_gram in skip_grams:
    new_pairs, new_labels = skip_gram
    pairs += new_pairs
    labels += new_labels

pairs = np.array(pairs)
labels = np.array(labels)

# Train the model
model.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=5)

# Extract embeddings
embedding_layer = model.get_layer('embedding')
word_embeddings = embedding_layer.get_weights()[0]
word_index = tokenizer.word_index

# Create a dictionary to map words to their embeddings
word_embedding_dict = {word: word_embeddings[idx] for word, idx in word_index.items()}

# Get the embedding vector for a word
word = 'word2vec'
embedding_vector = word_embedding_dict.get(word)
print(f"Vector for '{word}': {embedding_vector}")
```


--- 


#### GloVe: Global Vectors for Word Representation
   GloVe (Global Vectors for Word Representation) is an advanced technique in Natural Language Processing (NLP) that transforms words into numerical vectors by leveraging global word-word co-occurrence statistics from a corpus. Developed by Christopher D. Manning at Stanford University, GloVe provides rich semantic representations of words by capturing their contextual relationships.
  - #### **How GloVe Works:**
      - #### **Word Co-occurrence Matrix:**
          - **Context Window**: Define a context window of size $m$. For a given target word $w_i$, consider the words within this window as context words.
          - **Co-occurrence Matrix**: Construct a co-occurrence matrix $X$ where each element $X_ij$ represents the number of times word $j$ appears in the context of word $i$ across the entire corpus.

      - ####  **Probability and Ratios:**  To extract meaningful relationships from the co-occurrence matrix, GloVe focuses on the probabilities and ratios of word co-occurrences.
        - **Probability of Co-occurrence**:
           $$P_ij = X_ij / ∑_k X_ik$$
          - Here, $P_ij$ denotes the probability that word $j$ appears in the context of word $i$.

        - **Probability Ratio**:
           $$P_ik / P_jk = (X_ik / ∑_k X_ik) / (X_jk / ∑_k X_jk)$$
          - This ratio captures the relationship between words $i$ and $j$ for a common context word $k$.

  -  **GloVe Model Formulation:**
      - **Objective Function**: GloVe aims to learn word vectors $w_i$ and context word vectors $w~_j$ such that their dot product approximates the logarithm of their co-occurrence probability:
        $$w_i^T * w~_j + b_i + b~_j ≈ log(X_ij)$$
      - Where
        - $w_i$ and $w~_j$ are the word and context word vectors.
        - $b_i$ and $b~_j$ are bias terms.
      - The goal is to minimize the following weighted least squares loss:
        $$J = ∑_{i,j=1}^V f(X_ij) * (w_i^T * w~_j + b_i + b~_j - log(X_ij))^2$$
      - **Weighting Function**: The weighting function $f(X_ij)$ controls the influence of each co-occurrence pair, reducing the impact of very frequent or very rare co-occurrences:
        $$f(X_ij) = {(X_ij / x_max)^α if X_ij < x_max1 otherwise}$$
      - Where
        - $x_max$ and $α$ are hyperparameters (typically $α = 0.75$ and $x_max = 100$).

  -  **Training Process:**
      - **Initialization**:
         - Initialize word vectors $w_i$ and context vectors $w~_j$ randomly.
         - Initialize biases $b_i$ and $b~_j$.
      - **Optimization:**
         - Use stochastic gradient descent (SGD) or an adaptive optimization algorithm like AdaGrad to minimize the loss function.
         - Iteratively update vectors and biases based on the gradient of the loss function.
   - Advantages:
       - 1. Captures both global and local context of words.
       - 2. Efficient in handling large corpora.
       - 3. Produces meaningful embeddings that capture semantic relationships.

   - Limitations:
      - 1. Computationally expensive due to the need to compute the co-occurrence matrix.
      - 2. Static embeddings which do not adapt to different contexts within a document.
      - 3. Large memory requirement for storing the co-occurrence matrix.

``` Python 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dot, Add, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adagrad

# Sample text data
text_data = [
    "Natural language processing and machine learning are fascinating fields.",
    "GloVe is an excellent tool for word embedding."
]

# Tokenize text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
index_to_word = {v: k for k, v in word_index.items()}

# Create co-occurrence matrix (simplified example)
co_occurrence_matrix = np.zeros((vocab_size, vocab_size))

# Fill the co-occurrence matrix based on a context window (simplified for illustration)
window_size = 2
for sequence in sequences:
    for i, word_id in enumerate(sequence):
        start = max(0, i - window_size)
        end = min(len(sequence), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                co_occurrence_matrix[word_id, sequence[j]] += 1

# Hyperparameters
embedding_dim = 50
x_max = 100
alpha = 0.75

# Weighting function
def weighting_func(x):
    return np.minimum((x / x_max) ** alpha, 1.0)

weights = weighting_func(co_occurrence_matrix)

# Building the GloVe model
input_target = Input((1,))
input_context = Input((1,))

embedding_target = Embedding(vocab_size, embedding_dim, input_length=1, name="embedding_target")
embedding_context = Embedding(vocab_size, embedding_dim, input_length=1, name="embedding_context")
target = embedding_target(input_target)
context = embedding_context(input_context)

target = Reshape((embedding_dim,))(target)
context = Reshape((embedding_dim,))(context)

dot_product = Dot(axes=1)([target, context])

# Add bias terms
bias_input_target = Input((1,))
bias_input_context = Input((1,))
bias_target = Embedding(vocab_size, 1, input_length=1)(bias_input_target)
bias_context = Embedding(vocab_size, 1, input_length=1)(bias_input_context)
bias_target = Reshape((1,))(bias_target)
bias_context = Reshape((1,))(bias_context)

add_bias = Add()([dot_product, bias_target, bias_context])

model = Model(inputs=[input_target, input_context, bias_input_target, bias_input_context], outputs=add_bias)
model.compile(loss='mean_squared_error', optimizer=Adagrad())

# Prepare training data
pairs, labels, biases_target, biases_context = [], [], [], []

for i in range(vocab_size):
    for j in range(vocab_size):
        if co_occurrence_matrix[i, j] > 0:
            pairs.append((i, j))
            labels.append(np.log(co_occurrence_matrix[i, j]))
            biases_target.append(i)
            biases_context.append(j)

pairs = np.array(pairs)
labels = np.array(labels)
biases_target = np.array(biases_target)
biases_context = np.array(biases_context)

# Train the model
model.fit([pairs[:, 0], pairs[:, 1], biases_target, biases_context], labels, epochs=5)

# Extract embeddings
embedding_target_weights = model.get_layer('embedding_target').get_weights()[0]
embedding_context_weights = model.get_layer('embedding_context').get_weights()[0]

# Create a dictionary to map words to their embeddings
word_embedding_dict = {word: embedding_target_weights[word_index[word]] for word in word_index}

# Get the embedding vector for a word
word = 'glove'
embedding_vector = word_embedding_dict.get(word)
print(f"Vector for '{word}': {embedding_vector}")
```
```
Vector for 'glove': [-0.03988282  0.01510394 -0.04516843  0.00921018  0.01995736  0.01504889
 -0.04845165  0.01036947  0.00841802 -0.03494125 -0.00486815  0.03025544
  0.00366436 -0.04756535 -0.01701002  0.02813601 -0.03724954  0.04696105
  0.03691722  0.0395353   0.04234853 -0.01318511 -0.0009635  -0.01482506
  0.04066756  0.0400206  -0.03915076 -0.01858991  0.01982226 -0.00893309
  0.03469832 -0.03480824  0.01705096 -0.04444888 -0.03532882  0.02922454
 -0.04793992  0.01991567 -0.03989727  0.00138859 -0.0238043  -0.02344322
  0.04485585  0.03791862  0.04784629  0.01865678 -0.02116342 -0.02645371
  0.01796384 -0.01561937]
```

--- 
#### FastText

FastText, developed by Facebook AI Research (FAIR), is a popular technique for word representation in Natural Language Processing (NLP). It extends the concept of word embeddings introduced by Word2Vec by considering subword information, which is particularly useful for handling out-of-vocabulary words and morphologically rich languages.

- #### **How FastText Works:**
  - #### **Character n-grams:**
    - FastText represents each word as a bag of character n-grams, including the word itself and special boundary symbols.
    - For example, for the word `apple` and assuming `n=3`, the character n-grams would be: `<ap`, `app`, `ppl`, `ple`, `le>`, and `apple` itself.
  
  - #### **Vector Representation:**
    - Each word is represented as the sum of the vectors of its character n-grams.
    - The vectors for character n-grams are learned alongside the word embeddings during training.
  
  - #### **Word Embeddings:**
    - FastText trains word embeddings by optimizing a classification objective, typically a softmax classifier, over the entire vocabulary.
    - The context for each word is defined by the bag of its character n-grams.
  
  - #### **Training Process:**
    - FastText employs techniques like hierarchical softmax or negative sampling to efficiently train embeddings on large datasets.

- #### **Implementation Steps:**
  - #### **Data Preparation:**
    - Tokenize the text data into words.
    - Preprocess the text by lowercasing and removing punctuation if necessary.
  
  - #### **Model Building:**
    - Use an embedding layer to represent each character n-gram.
    - Sum the embeddings of all character n-grams to obtain the word representation.
    - Concatenate word and character n-gram embeddings.
    - Apply Global Average Pooling to aggregate embeddings.
  
  - #### **Training:**
    - Train the model using a softmax classifier with a cross-entropy loss function.
    - Use techniques like hierarchical softmax or negative sampling for efficiency.
  
  - #### **Evaluation:**
    - Evaluate the trained model on downstream tasks such as text classification, sentiment analysis, etc.

- #### **Advantages:**
  1. Uses character n-grams to manage out-of-vocabulary words.
  2. Captures word morphology, useful for languages with rich morphology.
  3. Effective for rare words.
  4. Adapts to different word structures.

- #### **Limitations:**
  1. Requires more memory due to character n-grams.
  2. Embeddings are less interpretable and explainable.
  3. Slower inference due to additional computation.
  4. Treats words as bags of n-grams, losing some context.

[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606v2)

--- 


#### ELMo
ELMo, short for `Embeddings from Language Models`, is a deep contextualized word representation technique developed by the Allen Institute for AI. Unlike traditional word embeddings like Word2Vec and FastText, which generate static embeddings, ELMo creates word representations that dynamically change based on the context in which the words appear. This approach significantly enhances the performance of various Natural Language Processing (NLP) tasks by providing a more nuanced understanding of words and their meanings.

#### How ELMo Works

##### Contextualized Embeddings/Dynamic Representations
Unlike static embeddings that assign a single vector to each word regardless of context, ELMo generates different vectors for a word depending on its usage in different sentences. This means that the word `bank` will have different embeddings when used in `river bank` and `savings bank`.

##### Deep, Bi-directional Language Model
- **Bi-directional LSTMs:** ELMo uses a deep bi-directional Long Short-Term Memory (bi-LSTM) network to model the word sequences. It reads the text both forward (left-to-right) and backward (right-to-left), capturing context from both directions.
- **Layered Approach:** ELMo's architecture consists of multiple layers of LSTMs. Each layer learns increasingly complex representations, from surface-level characteristics to deeper syntactic and semantic features.

##### Pre-trained on Large Corpora
- **Massive Pre-training:** ELMo models are pre-trained on large datasets, such as the 1 Billion Word Benchmark, to learn rich linguistic patterns and structures.
- **Fine-tuning for Specific Tasks:** After pre-training, these embeddings can be fine-tuned on specific NLP tasks, allowing ELMo to adapt to the nuances of the target task.

##### Architecture
- **Token Embeddings:** The input tokens are first embedded using a character-level convolutional neural network (CNN) to capture morphological information.
- **Contextualization:** The token embeddings are passed through a bi-directional LSTM to generate contextualized representations.
- **Layer Aggregation:** The final ELMo representation is a linear combination of the representations from each LSTM layer, with task-specific weights.

##### Training Procedure
- **Character-Level Embeddings:** Character convolutions are used to convert the input text into initial word embeddings.
- **Bi-directional LSTM Layers:** Two layers of bi-LSTM are used to capture contextual information in both forward and backward directions.
- **Objective Function:** ELMo is trained using a language modeling objective where the model predicts the next word in a sequence given the previous context (forward) and the previous word given the next context (backward).

##### Advantages
1. Produces context-specific embeddings for more accurate word meaning.
2. Captures complex syntactic and semantic information.
3. Outperforms static embeddings on various NLP tasks.

##### Limitations
1. Requires significant resources for training and inference.
2. Pre-training and fine-tuning are time-consuming.
3. Large memory requirements pose deployment challenges.

[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

---

#### BERT
BERT, short for `Bidirectional Encoder Representations from Transformers`, is a revolutionary language representation model developed by Google AI. Unlike previous models that process text in a unidirectional manner, BERT captures context from both directions simultaneously, providing a deeper understanding of language. This approach has set new benchmarks in various Natural Language Processing (NLP) tasks by offering more precise and comprehensive word representations.

#### How BERT Works

##### Bidirectional Contextualization
Unlike traditional models that read text sequentially, BERT uses Transformers to process text from both the left and the right simultaneously, capturing the full context of each word.

##### Transformer Architecture
- **Self-Attention Mechanism:** BERT's architecture relies on the self-attention mechanism within Transformers, which allows the model to weigh the importance of different words in a sentence, regardless of their position.
- **Layers of Transformers:** BERT consists of multiple layers of Transformer encoders, each providing a progressively richer representation of the text.

##### Pre-training and Fine-tuning
- **Pre-training Tasks:** BERT is pre-trained on large corpora using two unsupervised tasks: `Masked Language Modeling (MLM)` and `Next Sentence Prediction (NSP)`. MLM involves predicting masked words in a sentence, while NSP involves predicting the relationship between two sentences.
- **Fine-tuning:** After pre-training, BERT can be fine-tuned on specific NLP tasks (e.g., question answering, sentiment analysis) by adding a task-specific output layer.


##### Architecture
- **Input Representation:** The input to BERT includes token embeddings, segment embeddings, and positional embeddings. Token embeddings represent the words in the sentence, segment embeddings distinguish between different sentences in a pair, and positional embeddings represent the position of each token in the sequence.
- **Transformer Encoders:** BERT uses a stack of Transformer encoders. Each encoder layer consists of a multi-head self-attention mechanism and a feed-forward neural network.
- **Self-Attention:** The self-attention mechanism allows BERT to focus on different parts of the sentence when computing the representation for each word. Multi-head attention means multiple attention mechanisms run in parallel, providing different representations at different positions.

##### Training Procedure
- **Masked Language Modeling (MLM):** BERT randomly masks some tokens in the input sequence and trains the model to predict the masked tokens. This forces the model to rely on the context provided by surrounding tokens.
- **Next Sentence Prediction (NSP):** BERT is trained on pairs of sentences and learns to predict if the second sentence in the pair is the actual next sentence in the original document. This task helps the model understand sentence relationships.

##### Advantages
1. Produces embeddings that consider the context from both directions.
2. Captures intricate syntactic and semantic details.
3. Excels in a wide range of NLP tasks, setting new performance benchmarks.

##### Limitations
1. Requires substantial computational resources for both training and inference.
2. Pre-training on large datasets is time-consuming and computationally expensive.
3. The large model size demands significant memory, complicating deployment in resource-constrained environments.

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)



### Encooding vs Embedding 

| Feature             | Encoding                                                | Embedding                                               |
| :------------------ | :------------------------------------------------------ | :------------------------------------------------------ |
| **Definition**      | Convert text to numerical forms (e.g., one-hot, TF-IDF). | Represent words in continuous vector space.            |
| **Purpose**         | Prepare text for ML models.                              | Capture semantic meanings in text.                     |
| **Dimensionality**  | Fixed (vocabulary size).                                 | Variable (semantic dimensionality).                    |
| **Data Type**       | Discrete (words/characters).                            | Continuous (semantic vectors).                         |
| **Transformation**  | Tokenization, vectorization.                             | Trained on large text corpora with NNs.                |
| **Representation**  | Numeric/sparse vectors.                                  | Dense vectors capture semantic meanings.               |
| **Application**     | Classification, Name Entity Recognition.                     | Semantic search, Sentiment Analysis, Summarization.                    |
| **Complexity**      | Simple algorithms.                                      | Complex models, resource-intensive.                    |
| **Information Preservation** | Preserves basic structure.                          | Retains semantic and contextual info.                  |
| **Examples**        | Bag-of-Words, TF-IDF, Integer Encoding.                                   | Word2Vec, GloVe,FasText. ELMo.                                 |
| **Usage in NLP**    | Preprocessing text data.                                | Essential for understanding text semantics.            |
| **Representation Type** | Results in sparse vectors.                           | Dense vectors capture semantic meanings and context.    |

--- 

### Hybrid Representation Learning (Sparse + Dense)

#### SPLADE
SPLADE (SPArse Lexical AnD Expansion model) is a state-of-the-art information retrieval model that enhances the matching of queries and documents by combining the strengths of sparse lexical representations and deep learning techniques. It addresses the limitations of traditional retrieval methods and dense models to provide efficient and accurate search results.

#### Background

##### Traditional Retrieval Methods
Techniques like BM25 rely on term frequency (TF) and inverse document frequency (IDF) but often lack contextual understanding and semantic relevance.

##### Dense Models
Models such as BERT provide contextual embeddings but are computationally intensive and not scalable for large datasets.

#### Key Concepts

##### Sparse Representations
- Use of sparse term vectors, where only important terms are represented with non-zero weights.
- Efficient in terms of storage and computation, making it suitable for large-scale retrieval tasks.

##### Term Weighting
SPLADE assigns importance weights to terms during training, allowing the model to highlight significant terms for better query-document matching.

##### Query and Document Expansion
The model expands both queries and documents with semantically related terms, enhancing the retrieval process.
- For instance, a query like "best smartphone" might include terms such as "mobile," "phone," and "top."

#### How SPLADE Works

##### Training Phase

###### Based on BERT
Uses BERT to generate contextual embeddings, capturing the meaning of words within their context.

###### Learning Term Weights
During training, SPLADE learns to assign appropriate weights to terms, emphasizing those that contribute more to relevance.

###### Regularization
Techniques such as L1 regularization ensure the representations remain sparse, preventing overfitting and maintaining efficiency.

##### Inference Phase

###### Query Processing
The input query is expanded and weighted using the trained model, improving its representational richness.

###### Document Processing
Documents are similarly expanded and weighted, ensuring consistency in how terms are represented across queries and documents.

###### Matching and Ranking
SPLADE matches the expanded, weighted query representation against document representations, ranking documents based on their relevance.

#### Unique Features

##### Handling Stopwords and Punctuation
The model can assign weights to traditionally ignored terms like stopwords and punctuation if they carry contextual significance.

##### Controlled Vocabulary
Experiments show SPLADE's ability to handle a limited set of terms while still capturing relevant signals, demonstrating its robustness.

#### Advantages

##### Enhanced Matching
The combination of lexical expansion and learned term weights significantly improves the accuracy of query-document matching.

##### Efficiency
Sparse representations allow SPLADE to be efficient and scalable, making it practical for large-scale applications.

##### Contextual Understanding
Leveraging BERT embeddings enables SPLADE to understand and incorporate the context in which terms are used.

#### Example Scenario

For a query like "define androgen receptor," SPLADE processes the query by expanding it with related terms and assigning weights. This expanded and weighted query is then matched against documents that have undergone similar processing, resulting in the retrieval of the most relevant documents.

--- 

## Deep NLP 
- #### **Deep Learning - Basic Neural Network Components:**
   A basic neural network architecture includes:
   - **Input Layer:** Receives input data.
   - **Hidden Layers:** Process input through weighted connections.
   - **Activation Function:** Adds non-linearity to neuron outputs.
   - **Weights and Biases:** Parameters adjusted during training.
   - **Output Layer:** Produces the final output.
   - **Loss Function:** Measures the difference between predicted and actual output.
   - **Optimization Algorithm:** Updates weights to minimize loss.
   - **Learning Rate:** Controls the step size during optimization.

- ### **Purpose of Activation Functions in Neural Networks:**
  Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Without them, the network would only perform linear transformations, limiting its ability to model complex data.
- #### **Common Activation Functions:**
    - #### **Sigmoid Function:**
$$f(x) = \frac{1}{1 + e^{-x}}$$
  - For every input $x$ the output in between $0$ and $1$ 
  - **Use Cases:** Often used in the output layer for binary classification and multilable classification.
```python
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
   - #### **Tanh Function::**
     $$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
       - **Use Cases:** Useful for hidden layers, especially in RNNs, as it maps input to a range between $-1$ and $1$.
```python
import numpy as np
def tanh(x):
    return np.tanh(x)
```

   - #### **ReLU (Rectified Linear Unit):**
     ReLU zeros out negative values, hence its name.

$$
f(x) =
\begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

   - **Use Cases:**
       - Commonly used in hidden layers for its simplicity and efficiency.
       - Default activation function in many deep learning models like convolutional neural networks (CNNs) and deep fully connected networks.
       - Used to overcome vanishing gradient problem
```python
import numpy as np
def relu(x):
    return np.maximum(0, x)
```

   - #### **Leaky ReLU:**
$$
f(x) =
\begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$
   - **Use Cases:** Leaky ReLU is widely favoured in neural networks for its ability to prevent neuron death during training by maintaining a small gradient $𝛼$ for negative inputs. This helps accelerate convergence in deep learning models, making it more efficient than traditional activation functions like sigmoid and tanh, especially in deeper networks prone to the vanishing gradient problem.

```python
import numpy as np
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

   - #### **ELU (Exponential Linear Unit):**
$$
f(x) =
\begin{cases}
x & \text{if } x > 0 \\
\alpha (e^x - 1) & \text{if } x \leq 0
\end{cases}
$$
   - **Use Cases:**
     - Effective in preventing dead neurons and encouraging model robustness during training.
     - Facilitates smooth gradient flow, potentially improving convergence speed in complex models.


```python
import numpy as np
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

   - #### **Softmax Function:**
     Softmax is an activation function that converts logits (raw prediction values) into probabilities, ensuring that the output probabilities sum to $1$
        $$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
 - **Use Cases:** Used in the output layer for multi-class classification tasks.


```python
import numpy as np
def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)
```

--- 

- ### **Important Optimization Algorithms for NLP:**
  In NLP, optimization algorithms are crucial for training models effectively. The purpose of optimization algorithms in deep learning is to minimize the loss function, improving the model's ability to make accurate predictions by adjusting its parameters iteratively during training. Below are some of the most important optimization algorithms commonly used in NLP, along with their descriptions and implementations.
 -  #### **Gradient Descent:**
   Gradient Descent is a fundamental optimization algorithm used to minimize the loss function by iteratively moving towards the steepest descent direction of the gradient.
 -  #### **Algorithm:**
   - 1. Initialize the parameters `theta` randomly or with predefined values.
   - 2. Repeat until convergence:
       - Compute the gradient of the loss function concerning the parameters: `gradient = compute_gradient(loss_function, theta)`.
       - Update the parameters using the gradient and the learning rate: `theta = theta - learning_rate * gradient`.
       - Check for convergence criteria (e.g., small change in the loss function or maximum number of iterations reached).
``` Python
import numpy as np

# Gradient Descent optimization algorithm
def gradient_descent(loss_function, initial_theta, learning_rate, max_iterations=1000, epsilon=1e-6):
    theta = initial_theta
    for _ in range(max_iterations):
        gradient = compute_gradient(loss_function, theta)
        theta -= learning_rate * gradient
        loss = loss_function(theta)
        if abs(loss - loss_function(theta + learning_rate * gradient)) < epsilon:
            break
    return theta

# Compute gradient of the loss function
def compute_gradient(loss_function, theta, epsilon=1e-6):
    gradient = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += epsilon
        gradient[i] = (loss_function(theta_plus) - loss_function(theta)) / epsilon
    return gradient

# Loss function (squared loss)
def squared_loss(theta):
    return (theta - 5) ** 2

# Set initial parameters and hyperparameters
initial_theta = np.array([0.0])
learning_rate = 0.3

# Run gradient descent optimization
optimized_theta = gradient_descent(squared_loss, initial_theta, learning_rate)
print("Optimized theta:", optimized_theta)
print("Final loss:", squared_loss(optimized_theta))

```
```
Optimized theta: [4.99978978]
Final loss: [4.41904213e-08]
<ipython-input-1-9fdae2c3f097>:20: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  gradient[i] = (loss_function(theta_plus) - loss_function(theta)) / epsilon
```

-  ### **Comparison of Gradient Descent Algorithms:**
   - #### **Standard Gradient Descent:**
     -  **Description**: Iteratively updates parameters by computing the gradient of the loss function over the entire dataset.
     - **Update Rule**: `theta = theta - learning_rate * gradient`
     - **Advantages**: Stable convergence for convex functions.
     - **Disadvantages**: Slow for large datasets; each iteration is computationally expensive.

   - #### **Stochastic Gradient Descent (SGD):**
     - **Description**: Updates parameters using the gradient computed from a single training example, providing more frequent updates.
     - **Update Rule**: `theta = theta - learning_rate * gradient (single example)`
     - **Advantages**: Faster convergence on large datasets; can escape local minima.
     - **Disadvantages**: Can be noisy, leading to fluctuations in the loss function.
   - #### **Mini-Batch Gradient Descent:**
     - **Description**: Updates parameters using the gradient computed from a small subset (mini-batch) of training examples.
     - **Update Rule**: `theta = theta - learning_rate * gradient (mini-batch)`
     - **Advantages**: Reduces variance in updates, providing a balance between SGD and batch GD.
     - **Disadvantages**: Requires choosing an appropriate batch size; computational cost is still significant.

   - #### **Momentum-based Gradient Descent:**
     - **Description**: Incorporates a momentum term that helps accelerate updates in relevant directions and dampens oscillations.
     - **Update Rule**: 
      - `velocity = momentum * velocity + gradient`
      - `theta = theta - learning_rate * velocity`
     - **Advantages**: Speeds up convergence; reduces oscillations near minima.
     - **Disadvantages**: Requires tuning of the momentum parameter (momentum); can overshoot minima.

--- 

### Feedforward Neural Networks (FNN)

A Feedforward Neural Network (FNN) is the simplest form of artificial neural network. In this type of network, the information moves in only one direction—forward—from the input nodes, through the hidden nodes (if any), and to the output nodes. There are no cycles or loops in the network, making it straightforward to understand and implement.

#### Architecture

The basic architecture of a Feedforward Neural Network consists of the following components:

1. **Input Layer**: This layer receives the input data. Each node in this layer represents one feature of the input.
2. **Hidden Layers**: These layers process the input data. There can be one or more hidden layers in an FNN. Each node in a hidden layer applies a weighted sum of its inputs and an activation function.
3. **Output Layer**: This layer produces the final output of the network. The number of nodes in this layer corresponds to the number of output classes or the required output dimensions.

#### Mathematically

For a single hidden layer network:

$$
\text{Input Layer: } x
$$

$$
\text{Hidden Layer: } h = \sigma(W_1 x + b_1)
$$

$$
\text{Output Layer: } y = W_2 h + b_2
$$

Where:

- $W_1$ is the weight matrix connecting the input layer to the hidden layer
- $b_1$ is the bias vector for the hidden layer
- $\sigma$ is an activation function (e.g. ReLU, Sigmoid)
- $W_2$ is the weight matrix connecting the hidden layer to the output layer
- $b_2$ is the bias vector for the output layer

#### Advantages

1. Easy to implement and understand.
2. Can approximate any continuous function with enough layers and units.
3. Effective for classification and regression on structured data.
4. Predictable and easier to debug due to one-way data flow.

#### Limitations

1. Not suitable for sequential data like time series or text.
2. Cannot retain information from previous inputs.
3. Prone to overfitting, especially with limited data.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward_nn(x, W1, b1, W2, b2):
    h = sigmoid(np.dot(W1, x) + b1)
    y = sigmoid(np.dot(W2, h) + b2)
    return y

# input (3 features)
x = np.array([0.5, 0.1, 0.4])
# weights and biases
W1 = np.array([[0.2, 0.8, 0.5], [0.3, 0.4, 0.2]])
b1 = np.array([0.1, 0.2])
W2 = np.array([0.6, 0.9])
b2 = 0.3

# Forward pass
output = feedforward_nn(x, W1, b1, W2, b2)
print("Output:", output)
``` 

--- 

### Recurrent Neural Networks (RNN)

A Recurrent Neural Network (RNN) is a type of artificial neural network designed for sequential data. Unlike Feedforward Neural Networks, RNNs have connections that form directed cycles, allowing them to maintain information about previous inputs through internal states. This makes them particularly suitable for tasks where the context or order of data is crucial, such as time series prediction, natural language processing, and speech recognition.

#### Architecture

The basic architecture of a Recurrent Neural Network consists of the following components:

1. **Input Layer**: This layer receives the input data. In the context of sequences, each input is often processed one time step at a time.
2. **Hidden Layers**: These layers process the input data and maintain a memory of previous inputs. Each node in a hidden layer takes input not only from the current time step but also from the hidden state of the previous time step.
3. **Cell State**: This is a vector that stores the internal memory of the network. The cell state is updated at each time step based on the current input and the previous cell state.
4. **Output Layer**: This layer produces the final output of the network for each time step. The number of nodes in this layer corresponds to the desired output dimensions.

#### Mathematical Representation

For a single hidden layer RNN:

- **Input Layer**: $x$
- **Hidden State**: $h = \sigma(W_x \cdot x + W_h \cdot h + b)$
- **Cell State**: $c = f(c_{\text{prev}}, x)$
- **Output Layer**: $y = \sigma(W_y \cdot h + b)$

Where:
- $W_x$ is the weight matrix connecting the input layer to the hidden state
- $W_h$ is the weight matrix connecting the hidden state to itself
- $b$ is the bias vector for the hidden state
- $f$ is the forget gate function (e.g., sigmoid)
- $W_y$ is the weight matrix connecting the hidden state to the output layer
- $\sigma$ is an activation function (e.g., ReLU, Sigmoid)

$$
h = \sigma(W_x \cdot x + W_h \cdot h + b)
$$

$$
c = f(c_{\text{prev}}, x)
$$

$$
y = \sigma(W_y \cdot h + b)
$$

![RNN illustrated with this Image example](https://github.com/KaifAhmad1/Awesome-NLP-and-IR/blob/main/images/RNN.png)

#### Advantages

1. Suitable for sequential data like time series or text.
2. Can retain information from previous inputs.
3. Effective for modeling temporal dependencies in data.

#### Disadvantages/Limitations

1. Prone to vanishing gradients, which can make training difficult.
2. Difficult to train due to the complex recurrent connections.
3. Not suitable for very long sequences.

#### Example Code

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rnn(x, Wx, Wh, Wy, b, by):
    h = np.zeros((Wh.shape[0], 1))
    for t in range(len(x)):
        h = sigmoid(np.dot(Wx, x[t].reshape(-1, 1)) + np.dot(Wh, h) + b)
    y = np.dot(Wy, h) + by
    return y

# Input sequence (3 time steps, each of size 1)
x = np.array([[0.5], [0.1], [0.4]])

# Weights and biases
Wx = np.array([[0.2]])  
Wh = np.array([[0.1]])  
Wy = np.array([[0.6]]) 
b = np.array([[0.1]])   
by = np.array([[0.2]]) 

# Forward pass
output = rnn(x, Wx, Wh, Wy, b, by)
print("Output:", output)

``` 
### Different Types of RNNs

Over the years, several variants of RNNs have been developed to address various challenges and improve their performance. Here are some of the most prominent RNN variants:

#### 1. Vanilla RNNs/RNNs
- I have talked about them above.

#### 2. LSTMs

#### 3. GRUs

#### 4. Bidirectional LSTMs and GRUs

### Long Short-Term Memory (LSTM) Networks

A Long Short-Term Memory (LSTM) network is a type of Recurrent Neural Network (RNN) designed to handle the vanishing gradient problem that occurs in traditional RNNs. LSTMs are capable of learning long-term dependencies in data, making them particularly useful for tasks such as language modeling, speech recognition, and time series forecasting.

#### Architecture

The basic architecture of an LSTM network consists of the following components:

- **Input Gate:** This gate is responsible for controlling the flow of new information into the cell state. It consists of a sigmoid layer and a point-wise multiplication operation.
- **Forget Gate:** This gate determines what information to discard from the previous cell state. It also consists of a sigmoid layer and a point-wise multiplication operation.
- **Cell State:** This is the internal memory of the LSTM network, which stores information over long periods.
- **Output Gate:** This gate determines the output of the LSTM network based on the cell state and the hidden state.
- **Hidden State:** This is the internal state of the LSTM network, which captures short-term dependencies in the data.

#### Mathematically

The LSTM network can be represented mathematically as follows:

$$
\text{Input Gate: } i = σ(W_i \cdot x + U_i \cdot h + b_i)
$$

$$
\text{Forget Gate: } f = σ(W_f \cdot x + U_f \cdot h + b_f)
$$

$$
\text{Cell State: } c = f \cdot c_{\text{prev}} + i \cdot \tanh(W_c \cdot x + U_c \cdot h + b_c)
$$

$$
\text{Output Gate: } o = σ(W_o \cdot x + U_o \cdot h + b_o)
$$

$$
\text{Hidden State: } h = o \cdot \tanh(c)
$$

$$
\text{Output: } y = W_o \cdot h + b_o
$$

- **Where:**
  - $W_i, W_f, W_c, W_o$ are the weight matrices for the input, forget, cell, and output gates, respectively.
  - $U_i, U_f, U_c, U_o$ are the recurrent weight matrices for the input, forget, cell, and output gates, respectively.
  - $b_i, b_f, b_c, b_o$ are the bias vectors for the input, forget, cell, and output gates, respectively.
  - $σ$ is the sigmoid activation function.
  - $\tanh$ is the hyperbolic tangent activation function.

#### Advantages

1. LSTMs are capable of learning long-term dependencies in data, making them suitable for tasks such as language modeling and time series forecasting.
2. They are less prone to the vanishing gradient problem compared to traditional RNNs.
3. LSTMs can handle sequential data with varying lengths.

#### Limitations

1. LSTMs are computationally expensive to train and require large amounts of data.
2. They can be difficult to interpret and visualize due to their complex architecture.
3. LSTMs can suffer from overfitting if not regularized properly.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm(x, Wi, Ui, Wf, Uf, Wc, Uc, Wo, Uo, bi, bf, bc, bo):
    h = np.zeros((Ui.shape[0], 1))
    c = np.zeros((Ui.shape[0], 1))
    for t in range(len(x)):
        i = sigmoid(np.dot(Wi, x[t].reshape(-1, 1)) + np.dot(Ui, h) + bi)
        f = sigmoid(np.dot(Wf, x[t].reshape(-1, 1)) + np.dot(Uf, h) + bf)
        c = f * c + i * np.tanh(np.dot(Wc, x[t].reshape(-1, 1)) + np.dot(Uc, h) + bc)
        o = sigmoid(np.dot(Wo, x[t].reshape(-1, 1)) + np.dot(Uo, h) + bo)
        h = o * np.tanh(c)
    y = np.dot(Wo, h) + bo
    return y

# Input sequence (3 time steps, each of size 1)
x = np.array([[0.5], [0.1], [0.4]])

# Weights and biases
Wi = np.array([[0.2]])  
Ui = np.array([[0.1]])  
Wf = np.array([[0.3]])  
Uf = np.array([[0.2]])  
Wc = np.array([[0.4]])  
Uc = np.array([[0.3]])  
Wo = np.array([[0.5]])  
Uo = np.array([[0.4]])  
bi = np.array([[0.1]])   
bf = np.array([[0.2]])  
bc = np.array([[0.3]])  
bo = np.array([[0.4]])  

# Forward pass
output = lstm(x, Wi, Ui, Wf, Uf, Wc, Uc, Wo, Uo, bi, bf, bc, bo)
print("Output:", output)
```

--- 

### Gated Recurrent Unit (GRU) Networks

A Gated Recurrent Unit (GRU) network is a type of Recurrent Neural Network (RNN) designed to handle the vanishing gradient problem that occurs in traditional RNNs. GRUs are capable of learning long-term dependencies in data, making them particularly useful for tasks such as language modeling, speech recognition, and time series forecasting. GRUs are a simplified version of Long Short-Term Memory (LSTM) networks, with fewer gates and a more streamlined architecture.

#### Architecture

The basic architecture of a GRU consists of the following components:

- **Reset Gate:** This gate determines what information to discard from the previous hidden state. It consists of a sigmoid layer and a point-wise multiplication operation.
- **Update Gate:** This gate determines what information to update in the hidden state. It consists of a sigmoid layer and a point-wise multiplication operation.
- **Hidden State:** This is the internal state of the GRU network, which captures short-term dependencies in the data.
- **Output:** This is the output of the GRU network, which is calculated based on the hidden state.

#### Mathematically

The GRU network can be represented mathematically as follows:

$$
\text{Reset Gate: } r = σ(W_r \cdot x + U_r \cdot h + b_r)
$$

$$
\text{Update Gate: } z = σ(W_z \cdot x + U_z \cdot h + b_z)
$$

$$
\text{Hidden State: } h = (1 - z) \cdot h_{\text{prev}} + z \cdot \tanh(W \cdot x + U \cdot (r \cdot h) + b)
$$

$$
\text{Output: } y = W \cdot h + b
$$

- **Where:**
  - $W_r, W_z, W$ are the weight matrices for the reset, update, and output gates, respectively.
  - $U_r, U_z, U$ are the recurrent weight matrices for the reset, update, and output gates, respectively.
  - $b_r, b_z, b$ are the bias vectors for the reset, update, and output gates, respectively.
  - $σ$ is the sigmoid activation function.
  - $tanh$ is the hyperbolic tangent activation function.

#### Advantages

1. GRUs are capable of learning long-term dependencies in data, making them suitable for tasks such as language modeling and time series forecasting.
2. They are less prone to the vanishing gradient problem compared to traditional RNNs.
3. GRUs have a simpler architecture compared to LSTMs, with fewer parameters to train, making them computationally more efficient.

#### Limitations

1. GRUs, like LSTMs, can be computationally expensive to train and require large amounts of data.
2. They can be difficult to interpret and visualize due to their complex architecture.
3. GRUs can suffer from overfitting if not regularized properly.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gru(x, Wz, Uz, Wr, Ur, W, U, bz, br, b):
    h = np.zeros((Uz.shape[0], 1))
    for t in range(len(x)):
        z = sigmoid(np.dot(Wz, x[t].reshape(-1, 1)) + np.dot(Uz, h) + bz)
        r = sigmoid(np.dot(Wr, x[t].reshape(-1, 1)) + np.dot(Ur, h) + br)
        h_tilde = np.tanh(np.dot(W, x[t].reshape(-1, 1)) + np.dot(U, r * h) + b)
        h = (1 - z) * h + z * h_tilde
    return h

# Input sequence (3 time steps, each of size 1)
x = np.array([[0.5], [0.1], [0.4]])

# Weights and biases
Wz = np.array([[0.2]])  
Uz = np.array([[0.1]])  
Wr = np.array([[0.3]])  
Ur = np.array([[0.2]])  
W = np.array([[0.4]])  
U = np.array([[0.3]])  
bz = np.array([[0.1]])   
br = np.array([[0.2]])  
b = np.array([[0.3]])  

# Forward pass
output = gru(x, Wz, Uz, Wr, Ur, W, U, bz, br, b)
print("Output:", output)
```

### Bidirectional RNNs

Bidirectional Recurrent Neural Networks (BRNNs) improve upon traditional RNNs by considering both past and future information in their predictions. This makes them highly effective for tasks involving sequential data, such as text and time series.

#### Architecture

BRNNs consist of two RNNs: one processes the input sequence forward, and the other processes it backwards. The outputs of these RNNs are concatenated to form the final output, allowing the network to use information from both directions.

#### Types

- **LSTM (Long Short-Term Memory):** Effective for learning long-term dependencies, ideal for tasks like language modeling and speech recognition.
- **GRU (Gated Recurrent Unit):** Simpler and more computationally efficient than LSTMs, suitable for tasks like text classification and sentiment analysis.

#### Mathematical Representation

$$
\text{Forward RNN: } h_{\text{forward}} = \text{LSTM}(x, W, U, b) \text{ or } \text{GRU}(x, W, U, b)
$$

$$
\text{Backward RNN: } h_{\text{backward}} = \text{LSTM}(x, W, U, b) \text{ or } \text{GRU}(x, W, U, b)
$$

$$
\text{Output: } y = \text{Concat}(h_{\text{forward}}, h_{\text{backward}})
$$

- **Where:**
  - $x$ is the input sequence
  - $W$, $U$, and $b$ are the weights, recurrent weights, and biases
  - $h_{\text{forward}}$ and $h_{\text{backward}}$ are the hidden states
  - $y$ is the output

#### Advantages

1. Capture context from both past and future.
2. Handle variable-length sequential data.
3. Learn long-term dependencies.

#### Disadvantages/Limitations

1. Computationally intensive.
2. Require large datasets.
3. Complex architecture can lead to overfitting.

```python
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
import numpy as np

# Input sequence
x = np.array([[[0.5], [0.1], [0.4]]])

# BRNN model
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(3, 1)))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(x, x, epochs=10)

# Make predictions
y_pred = model.predict(x)
print("Output:", y_pred)
``` 
--- 

### Transformers
Transformers are a type of deep learning model introduced by Ashish Vaswani in the paper "Attention is All You Need" (2017). They are particularly powerful for handling sequential data, such as text, but unlike RNNs, they do not process data in a sequential manner. Instead, Transformers use self-attention mechanisms to model dependencies between all elements of the input sequence simultaneously, allowing for much greater parallelization during training.


![Transformers Architecture](https://github.com/KaifAhmad1/Awesome-NLP-and-IR/blob/main/images/Tranformers.png)


#### Secret Sauce: Self-Attention Mechanism
Self-attention is a key component of the Transformer architecture, which enables the model to weigh the significance of different parts of the input sequence when processing each element. This mechanism helps the model capture relationships and dependencies between all elements in the sequence, regardless of their distance from each other.

#### Steps
1. #### Input Sentence
   We start with the sentence: `She opened the door to the garden.`

2. #### Convert Words to Vectors
   Each word in the sentence is represented as a vector after being passed through an embedding layer. Let's use simplified vector representations for clarity:
   - She: $[1, 0, 0]$
   - opened: $[0, 1, 0]$
   - the: $[0, 0, 1]$
   - door: $[1, 1, 0]$
   - to: $[1, 0, 1]$
   - the: $[0, 1, 1]$
   - garden: $[1, 1, 1]$

3. #### Creating Q, K, and V Matrices
   For each word, we create three vectors: Query (Q), Key (K), and Value (V). These vectors are derived by multiplying the word vector by three different weight matrices $(W_Q, W_K, W_V)$.
   - **Query (Q):** Represents the word we are currently processing.
   - **Key (K):** Represents the words we compare the current word against.
   - **Value (V):** Represents the actual content of the words.
   - Example for "opened":
     - $Q_{\text{opened}} = [0, 1, 0] \times W_Q$
     - $K_{\text{opened}} = [0, 1, 0] \times W_K$
     - $V_{\text{opened}} = [0, 1, 0] \times W_V$

4. #### Calculating Attention Scores
   Dot product of the Query vector of the word with Key vectors of all words.
   - Example for `opened`:
     - Score for $She = Q_{\text{opened}} \cdot K_{\text{She}}$
     - Score for $opened = Q_{\text{opened}} \cdot K_{\text{opened}}$
     - Score for $the = Q_{\text{opened}} \cdot K_{\text{the}}$

5. #### Applying Softmax
   Pass attention scores through the Softmax function to get attention weights.
   - Softmax formula: $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

6. #### Weighted Sum of Values
   Multiply Value vectors by their corresponding attention weights.
   - Example for `opened`:
     - $$\text{Weighted sum} = \text{softmax}(\text{Scores}) \times [V_{\text{She}}, V_{\text{opened}}, V_{\text{the}}, V_{\text{door}}, V_{\text{to}}, V_{\text{the}}, V_{\text{garden}}]$$

By iteratively performing these steps for all words in the input sentence, the self-attention mechanism captures intricate relationships and dependencies across the entire sequence, facilitating effective sequence-to-sequence processing tasks like language translation or text generation.

### Key Components of Transformer Architecture
The Transformer architecture consists of an encoder and a decoder, both composed of multiple identical layers. Each layer in both the encoder and decoder contains two main sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.

#### Encoder
The encoder processes the input sequence and generates a set of feature representations for each element in the sequence. It consists of:
- **Input Embedding:** Converts input tokens into dense vectors.
- **Positional Encoding:** Adds information about the position of each token in the sequence, since the model does not inherently capture sequence order.
- **Multi-Head Self-Attention:** Allows the model to focus on different parts of the sequence simultaneously. Each head processes the sequence differently, and the results are concatenated and linearly transformed.
- **Feed-Forward Network:** Applies two linear transformations with a ReLU activation in between, applied to each position separately.
- **Layer Normalization:** Normalizes the output of each sub-layer (attention and feed-forward).
- **Residual Connection:** Adds the input of each sub-layer to its output, aiding in training deeper networks.

Mathematically, for each sub-layer:
- **Self-Attention:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
  - Where:
    - $Q$, $K$, and $V$ are the query, key, and value matrices, respectively.
    - $d_k$ is the dimension of the key/query vectors.
- **Feed-Forward Network:**
  $$\text{FFN}(x) = \text{max}(0, xW_1 + b_1) W_2 + b_2$$

#### Decoder
The decoder generates the output sequence, one token at a time, using the encoded representations and the previously generated tokens. It consists of:
- **Output Embedding:** Converts output tokens into dense vectors.
- **Positional Encoding:** Adds information about the position of each token in the sequence, since the model does not inherently capture sequence order. Similar to the encoder's positional encoding.
- **Masked Multi-Head Self-Attention:** Prevents attending to future tokens by masking them.
- **Multi-Head Attention:** Attends to the encoder's output representations.
- **Feed-Forward Network:** Applies two linear transformations with a ReLU activation in between, applied to each position separately.
- **Layer Normalization:** Normalizes the output of each sub-layer (attention and feed-forward).
- **Residual Connection:** Adds the input of each sub-layer to its output, aiding in training deeper networks.

Mathematically, for each sub-layer:
- **Attention Mechanism:** The attention mechanism allows the model to weigh the importance of different tokens when processing a sequence. In the Transformer, the scaled dot-product attention is used:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
- **Multi-Head Attention:** To allow the model to focus on different positions and features, the Transformer uses multi-head attention:
  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$
  - Where:
    $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### Advantages:
1. **Parallelization:** Unlike RNNs, Transformers can process all tokens in a sequence simultaneously, allowing for faster training.
2. **Long-Range Dependencies:** The self-attention mechanism can capture long-range dependencies more effectively than RNNs.
3. **Scalability:** Scales well with larger datasets and model sizes.

#### Disadvantages/Limitations:
1. **Computational Cost:** Self-attention has a quadratic complexity with respect to the sequence length, making it computationally expensive for long sequences.
2. **Memory Usage:** Requires significant memory to store the attention weights.

---

### Transformer Architectures: A Detailed Comparison
Transformers have become a dominant architecture in the field of natural language processing (NLP), with various flavours and applications. Below is a comparison of the key differences between the three main transformer architectures:

| **Aspect**                           | **Encoder-Style Transformer**                                     | **Decoder-Style Transformer**                                     | **Encoder-Decoder Style Transformer**                               |
|--------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|--------------------------------------------------------------------|
| **Structure**                        | Multiple layers of encoders, without any decoders                 | Multiple layers of decoders, without any encoders                 | Separate stacks of encoders and decoders                           |
| **Primary Usage**                    | Representation learning and classification tasks                  | Generative tasks (e.g., autoregressive text generation)           | Sequence-to-sequence tasks (e.g., translation, summarization)      |
| **Examples**                         | BERT, RoBERTa, DistilBERT                                         | GPT Series by OpenAI, LLaMA Series by Meta, Mistral, etc.         | Transformer, BART, T5                                              |
| **Attention Mechanism**              | Self-attention within each encoder layer                          | Self-attention within each decoder layer, with masked attention   | Self-attention in encoders, cross-attention in decoders            |
| **Training Objective**               | Masked language modeling and next sentence prediction             | Causal language modeling (predicting the next token)              | Supervised learning with source and target sequences               |
| **Advantages**                       | - Good at capturing bidirectional context                         | - Effective at generating coherent text                           | - Effective at learning mappings between input and output sequences|
|                                      | - Effective for understanding tasks (e.g., sentiment analysis)    | - Can handle long text generation                                 | - Handles both input and output dependencies effectively           |
|                                      | - Pre-training can be easily adapted to various downstream tasks  |                                                                   | - Versatile for various tasks                                      |
| **Limitations**                      | - Not designed for generative tasks                               | - Unidirectional context                                          | - More complex architecture                                        |
|                                      | - Requires large datasets for pre-training                        | - Can suffer from exposure bias                                   | - Computationally intensive                                        |
| **Applications**                     | - Text classification                                             | - Text generation                                                 | - Machine translation                                              |
|                                      | - Named entity recognition                                        | - Dialogue systems                                                | - Text summarization                                               |
|                                      | - Sentence embedding                                              | - Story completion                                                | - Speech recognition                                               |





## References

1. **Japanese and Korean Voice Search** by Mike Schuster et al. [:link:](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
2. **Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates** by Taku Kudo et al. [:link:](https://arxiv.org/pdf/1804.10959)
3. **SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing** by Taku Kudo et al. [:link:](https://arxiv.org/pdf/1808.06226)
4. **Scaling Hidden Markov Language Models** by Justin T. Chiu et al. [:link:](https://arxiv.org/pdf/2011.04640v1)
5. **Exploring Conditional Random Fields for NLP Applications** [:link:](https://www.hyperscience.com/blog/exploring-conditional-random-fields-for-nlp-applications/)
6. **Encoding categorical data: Is there yet anything ‘hotter’ than one-hot encoding** [:link:](https://arxiv.org/pdf/2312.16930)
7. **A Statistical Interpretation of Term Specificity and Its Application in Retrieval** by Karen et al. [:link:](https://www.emerald.com/insight/content/doi/10.1108/eb026526/full/html)
8. **Okapi at TREC-3** by Stephen E et al. [:link:](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/okapi_trec3.pdf)
9. **Efficient Estimation of Word Representations in Vector Space** by Tomas Mikolov et al. [:link:](https://arxiv.org/pdf/1301.3781)
10. **GloVe: Global Vectors for Word Representation** by Christopher D. Manning et al. [:link:](https://nlp.stanford.edu/pubs/glove.pdf)
11. **Enriching Word Vectors with Subword Information** by Facebook AI Research et al. [:link:](https://arxiv.org/pdf/1607.04606)
12. **Word Embeddings: A Survey** by Felipe Almeida et al. [:link:](https://arxiv.org/pdf/1901.09069)
13. **Deep contextualized word representations** by Matthew E. Peters et al. [:link:](https://arxiv.org/pdf/1802.05365)
14. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** by Google AI et al. [:link:](https://arxiv.org/pdf/1810.04805)
15. **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking** by Stanford et al. [:link:](https://arxiv.org/abs/2107.05720)
16. **SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval** by Stanford et al. [:link:](https://arxiv.org/pdf/2109.10086)
17. **Recent Trends in Deep Learning Based Natural Language Processing** by Tom Young et al. [:link:](https://arxiv.org/pdf/1708.02709)
18. **Neural Information Retrieval: A Literature Review** by Ye Zhang∗ et al. [:link:](https://arxiv.org/pdf/1611.06792)
19. **Recurrent Neural Networks and Long Short-Term Memory Networks: Tutorial and Survey** by Benyamin Ghojogh and Ali Ghodsi et al. [:link:](https://arxiv.org/pdf/2304.11461)
20. **Attention Is All You Need** by Ashish Vaswani∗ et al. [:link:](https://arxiv.org/pdf/1706.03762)
21. **Representation Learning: A Review and New Perspectives** by Yoshua Bengio et al. [:link:](https://arxiv.org/pdf/1206.5538)

