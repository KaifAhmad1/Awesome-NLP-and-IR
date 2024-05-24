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

- **Bag of words:**  The Bag of Words (BoW) model in Natural Language Processing (NLP) converts text into numerical vectors by creating a vocabulary of unique words from a corpus and representing each document by a vector of word frequencies.
  -  This method is simple and effective for tasks like text classification and clustering, though it ignores grammar, word order, and context, leading to potential loss of information and high-dimensional, sparse vectors. Despite its limitations, BoW is popular due to its ease of use and effectiveness.
  -  **Process Steps:**
     - **Corpus Collection:** Gathers a comprehensive set of text documents to form the corpus, laying the groundwork for analysis and modelling.
        - I love reading books. Books are great.
        - Books are a wonderful source of knowledge.
        - I have a great love for reading books.
        - Reading books can be very enlightening. Books are amazing.
     - **Preprocessing:** Executes meticulous text preprocessing tasks, including lowercasing, punctuation removal, and stop word elimination, to maintain standardized data quality.
        - i love reading books books are great
        - books are a wonderful source of knowledge
        - i have a great love for reading books
        - reading books can be very enlightening books are amazing
     - **Vocabulary Building:** Extracts unique words from the corpus, forming the foundational vocabulary that encompasses diverse linguistic elements.
        - Vocabulary: [`i`, `love`, `reading`, `books`, `are`, `great`, `a`, `wonderful`, `source`, `of`, `knowledge`, `have`, `for`, `can`, `be`, `very`, `enlightening`, `amazing`]
     - **Vectorization:** Transforms each document into a numerical vector representation based on the established vocabulary. Ensures vector length matches vocabulary size, with elements representing word frequencies, succinctly capturing the document's textual essence.
        - Sentence 1: `i love reading books books are great` Vector: `[1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
        - Sentence 2: `books are a wonderful source of knowledge` Vector: `[0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]`
        - Sentence 3: `i have a great love for reading books` Vector: `[1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]`
        - Sentence 4: `reading books can be very enlightening books are amazing` Vector: `[0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]`

   - Advantages:
      - 1. Simple to implement.
      - 2. Efficient conversion of text to numerical data
      - 3. Effective for basic text classification and clustering
   - Disadvantages:
      - 1. Loss of context
      - 2. High dimensionality and sparse vectors
      - 3. Sparse data representations may pose challenges for some algorithms
``` Python 
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
```
Sentence 1: I love reading books. Books are great.
Vector: [0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]

Sentence 2: Books are a wonderful source of knowledge.
Vector: [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]

Sentence 3: I have a great love for reading books.
Vector: [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]

Sentence 4: Reading books can be very enlightening. Books are amazing.
Vector: [1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]

Vocabulary: ['amazing' 'are' 'be' 'books' 'can' 'enlightening' 'for' 'great' 'have'
 'knowledge' 'love' 'of' 'reading' 'source' 'very' 'wonderful']
```
- **TF-IDF:** TF-IDF is a numerical statistic used in information retrieval and text mining. It reflects the importance of a word in a document relative to a collection of documents (corpus). TF-IDF is often used as a weighting factor in search engine algorithms and text analysis.
   - **Components of TF-IDF:**
      - **Term Frequency (TF):** Measures how frequently a term occurs in a document.
          - Term Frequency is calculated as:
          - `TF(t,d) = f(t,d) / sum(f(t',d) for all t' in d)`
      - where:
          - `f(t,d)` is the raw count of term `t` in document `d`.
          - The denominator is the total number of terms in document `d`.
      - Example:
          - If the term `data` appears 3 times in a document with 100 words, the term frequency TF for `data` would be: `TF(data, d) = 3 / 100 = 0.03`
     
      - **Inverse Document Frequency (IDF):** Measures how frequently a term occurs in a document.
          - Inverse Document Frequency is calculated as:
          - `IDF(t, D) = log(N / |{d in D : t in d}|)`
      - where:
          - `N` is the total number of documents.
          - `|{d in D : t in d}|` is the number of documents containing the term `t`.
      - Example:
          - If the corpus contains `10,000` documents, and the term `data` appears in 100 of these documents, the inverse document frequency IDF for `data` would be:
          - `IDF(data, D) = log(10000 / 100) = log(100) = 2`
      - **Calculating TF-IDF:**
          - The TF-IDF score for a term `t` in a document `d` is given by: `TF-IDF(t,d,D) = TF(t,d) * IDF(t,D)`
      - Example:
          - Using the previous values:
          - TF(data, d) = 0.03
          - IDF(data, D) = 2
          - The TF-IDF score for `data` in the document would be:
          - TF-IDF(data, d, D) = 0.03 * 2 = 0.06

   - Advantages:
      - 1. Simple and easy to understand
      - 2. Effective in identifying relevant terms and their weights
      - 3. Distinguishes between common and rare terms
      - 4. Language-independent

   - Disadvantages\Limitations:
      - 1. Doesn't consider semantics or term context
      - 2. May struggle with very long documents due to term frequency saturation. In lengthy documents, even insignificant terms can surface frequently, resulting in elevated and saturated term frequencies. Consequently, TF-IDF may encounter challenges in effectively distinguishing and assigning significant weights to important terms.
      - 3. Ignores term dependencies and phrase
      - 4. Needs large document collections for reliable IDF
``` Python
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
```
Document 1:
This: 0.500
is: 0.312
document: 0.250
1.: 0.312
It: 0.500
contains: 0.500
some: 0.312
terms.: 0.312
Document 2:
Document: 0.312
2: 0.500
has: 0.500
different: 0.500
terms: 0.250
than: 0.500
document: 0.250
1.: 0.312
Document 3:
Document: 0.250
3: 0.400
is: 0.250
another: 0.400
example: 0.400
document: 0.200
with: 0.400
some: 0.250
common: 0.400
terms.: 0.250
```
- **BM25 (Best Matching 25):** BM25 (Best Matching 25) is a ranking function used in information retrieval to estimate the relevance of documents to a given search query. It is an extension of the TF-IDF weighting scheme, designed to address some of its limitations while retaining its simplicity and effectiveness.
   - **Components of BM25:**
      - **Term Frequency (TF):** Measures how frequently a term occurs in a document, similar to TF in TF-IDF.
         - TF in BM25 is adjusted to handle document length normalization and term saturation. It is calculated as:
            - `TF(t,d) = (f(t,d) * (k + 1)) / (f(t,d) + k * (1 - b + b * (|d| / avgdl)))`
      - Where
         - `f(t,d)` is the raw count of term `t` in document `d`.
         - `|d|` is the length of document `d`.
         - `avgdl` is the average document length in the corpus.
         - `k` and `b` are tuning parameters, typically set to 1.5 and 0.75 respectively.
     
      - **Inverse Document Frequency (IDF):** Measures how frequently a term occurs in the entire document collection, similar to IDF in TF-IDF.
         - IDF in BM25 is calculated as:
            - `IDF(t, D) = log((N - n(t) + 0.5) / (n(t) + 0.5))`
       - Where
            - `N` is the total number of documents in the collection.
            - `n(t)` is the number of documents containing term t.
      - **Document length Normalization:** Adjusts the TF component based on the length of the document. This ensures that longer documents do not have an unfair advantage over shorter ones.
   
- **BM25 Score Calculation:**
    - The BM25 score for a term t in a document d given a query q is calculated as:
       - `BM25(t, d, q) = IDF(t, D) * ((f(t, d) * (k + 1)) / (f(t, d) + k * (1 - b + b * (|d| / avgdl))))`
    - Where:
       - `IDF(t, D)` is the inverse document frequency for term `t`.
       - `f(t, d)` is the term frequency for term `t` in document `d`.
       - `|d|` is the length of document `d`.
       - `avgdl` is the average document length in the corpus.
       - `k` and `b` are tuning parameters.

   - Advantages:
      - 1. Effective in ranking documents based on relevance to a query.
      - 2. Accounts for document length normalization, addressing a limitation of TF-IDF.
      - 3. Suitable for large document collections.
      - 4. Robust and widely used in practice.

   - Disadvantages\Limitations:
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
- **Embeddings in Representation Learning:** In Natural Language Processing (NLP), an `embedding` is a way of representing words, phrases, or even entire documents as continuous, dense vectors of numbers. These vectors capture the semantic meaning of the text in such a way that words or phrases with similar meanings are represented by similar vectors.
    - Example: Consider the words `king,` `queen,` `man,` and `woman.` In a well-trained embedding space, these words might be represented by the following vectors (these numbers are just illustrative examples):
        - king = `[0.25, 0.75, 0.10, 0.60]`
        - queen = `[0.20, 0.80, 0.15, 0.65]`
        - man = `[0.30, 0.60, 0.05, 0.50]`
        - woman = `[0.25, 0.70, 0.10, 0.55]`
    - In this vector space, the embeddings for `king` and `queen` are closer to each other than to `man` and `woman,` capturing the relationship between royalty. Similarly, the difference between `king` and `man` is similar to the difference between `queen` and `woman,` capturing the gender relationship.

- **Word2Vec:** Word2vec is a popular technique in Natural Language Processing (NLP) that transforms words into numerical vectors, capturing their meanings and relationships. Developed by Tomas Mikolov and his team at Google in 2013, Word2vec comes in two main models: Continuous Bag-of-Words (CBOW) and Skip-Gram.
   - **How Word2vec Works:**
       - **Continuous Bag-of-Words (CBOW):** The CBOW model predicts a target word based on its surrounding context words. Here’s how it works:
         - Context Window: Assume a context window of size m. For a given target word w_t, the context words are
            - `w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}`.
         - Input Representation: Each word `w` in the vocabulary `V` is represented as a one-hot vector `x ∈ R^{|V|}`, where only the index corresponding to `w` is `1` and all other indices are `0`.
         - Projection Layer: The input one-hot vectors are mapped to a continuous vector space using a weight matrix `W ∈ R^{|V| x d}`, where `d` is the dimensionality of the word vectors (embeddings). The context word vectors are averaged:
            - `v_c = (1 / 2m) * sum_{i=-m, i ≠ 0}^{m} W * x_{t+i}`

         - Output Layer: The averaged context vector v_c is then multiplied by another weight matrix W' ∈ R^{d x |V|} and passed through a softmax function to produce the probability distribution over the vocabulary:
            - `p(w_t | w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}) = exp(v_{w_t} ⋅ v_c) / sum_{w ∈ V} exp(v_w ⋅ v_c)`

       - **Skip-Gram:** The Skip-Gram model, on the other hand, predicts context words given the target word. The steps are:
          - Input Representation: For a target word `w_t`, represented as a one-hot vector `x_t`.
          - Projection Layer: The one-hot vector is projected into the embedding space using the weight matrix `W:v_t = W * x_t`
          - Output Layer: For each context word `w_{t+i}` (within the window of size m), the model predicts the probability distribution using the weight matrix `W'` and `softmax`:
            - `p(w_{t+i} | w_t) = exp(v_{w_{t+i}} ⋅ v_t) / sum_{w ∈ V} exp(v_w ⋅ v_t)`
       - **Negative Sampling:** Negative sampling simplifies the training process by approximating the softmax function. Instead of computing the gradient over the entire vocabulary, negative sampling updates the weights for a small number of `negative` words (words not in the context). For each context word `w_O` and target word `w_I`:
          - `log σ(v_{w_O} ⋅ v_{w_I}) + sum_{i=1}^{k} E_{w_n ∼ P_n(w)} [log σ(-v_{w_n} ⋅ v_{w_I})]`
        - where
           - `σ` is the sigmoid function
           -  `k` is the number of negative samples
           -   and `P_n(w)` is the noise distribution
       - **Hierarchical Softmax:** Hierarchical softmax reduces computational complexity by representing the vocabulary as a binary tree. Each word is a leaf node, and predicting a word involves traversing from the root to the leaf node. The probability of a word is the product of the probabilities of the decisions made at each node in the path.

``` Python 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.layers import Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model

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
```
Vector for 'word2vec': [-0.03828416  0.04326824 -0.01323479 -0.03898887 -0.0182811   0.05545079
 -0.00735809 -0.0188455   0.00851007  0.01436511 -0.04801781  0.01853973
  0.01926632  0.03376403  0.0487873  -0.01145255 -0.04073619  0.03759814
 -0.01414892 -0.05184942  0.01196407 -0.03316187 -0.00062611  0.01561211
  0.00904219 -0.03647533 -0.00858539  0.04711245 -0.02592229 -0.02614572
  0.02072014 -0.05968965 -0.0232677   0.01341342 -0.00252265 -0.03830103
  0.05060257  0.02582355 -0.01276324  0.05118259 -0.03334418  0.00891038
  0.03433971 -0.00979966 -0.00288447  0.02499143 -0.01213881  0.03580792
  0.04264407 -0.0246205  -0.05220392  0.01457958 -0.0077569  -0.02423846
 -0.04646793  0.05509846  0.00207892  0.01464774  0.05045829  0.03539532
  0.05784871 -0.00439313 -0.05079681 -0.03624607 -0.00137503 -0.01823079
 -0.03566554 -0.00455899  0.03955529  0.01125429  0.01601778 -0.01737827
  0.027907   -0.05140529 -0.01299851 -0.02882623 -0.02424621 -0.006307
 -0.00143718 -0.05011651 -0.05629823 -0.05643904  0.05539116 -0.01744897
  0.02687337 -0.00076933 -0.04316826  0.00651922 -0.00661349 -0.06343716
 -0.03426652  0.03874123 -0.04863291 -0.02591641  0.00344516  0.0478721
  0.06752533 -0.03133888  0.00209786 -0.01114183]
```
   - Advantages:
       - 1. Efficient training on large datasets.
       - 2. Captures semantic similarities.
       - 3. Enables easy comparison of words.
       - 4. Handles large datasets.
       - 5. Flexible for task-specific fine-tuning.

   - Disadvantages\Limitations:
      - 1. Ignores word order beyond a fixed window.
      - 2. Out-of-vocabulary words are not represented.
      - 3. Large embedding matrices can be memory-intensive.
      - 4. Static Embeddings which Doesn't adapt to different contexts within a document.

- **GloVe: Global Vectors for Word Representation:** GloVe (Global Vectors for Word Representation) is an advanced technique in Natural Language Processing (NLP) that transforms words into numerical vectors by leveraging global word-word co-occurrence statistics from a corpus. Developed by Christopher D. Manning at Stanford University, GloVe provides rich semantic representations of words by capturing their contextual relationships.
  - **How GloVe Works:**
      - **Word Co-occurrence Matrix:**
          - **Context Window**: Define a context window of size `m`. For a given target word `w_i`, consider the words within this window as context words.
          - **Co-occurrence Matrix**: Construct a co-occurrence matrix `X` where each element `X_ij` represents the number of times word `j` appears in the context of word `i` across the entire corpus.

  -  **Probability and Ratios:**  To extract meaningful relationships from the co-occurrence matrix, GloVe focuses on the probabilities and ratios of word co-occurrences.
      - **Probability of Co-occurrence**:
         - `P_ij = X_ij / ∑_k X_ik`
         - Here, `P_ij` denotes the probability that word `j` appears in the context of word `i`.

      - **Probability Ratio**:
         - `P_ik / P_jk = (X_ik / ∑_k X_ik) / (X_jk / ∑_k X_jk)`
         - This ratio captures the relationship between words `i` and `j` for a common context word `k`.

  -  **GloVe Model Formulation:**
      - **Objective Function**: GloVe aims to learn word vectors `w_i` and context word vectors `w~_j` such that their dot product approximates the logarithm of their co-occurrence probability:
         - `w_i^T * w~_j + b_i + b~_j ≈ log(X_ij)`
      - Where
        - `w_i` and `w~_j` are the word and context word vectors.
        - `b_i` and `b~_j` are bias terms.
      - The goal is to minimize the following weighted least squares loss:
        - `J = ∑_{i,j=1}^V f(X_ij) * (w_i^T * w~_j + b_i + b~_j - log(X_ij))^2`
      - **Weighting Function**: The weighting function `f(X_ij)` controls the influence of each co-occurrence pair, reducing the impact of very frequent or very rare co-occurrences:
        - `f(X_ij) = {(X_ij / x_max)^α if X_ij < x_max1 otherwise}`
      - Where
        - `x_max` and `α` are hyperparameters (typically `α = 0.75` and `x_max = 100`).

  -  **Training Process:**
      - **Initialization**:
         - Initialize word vectors `w_i` and context vectors `w~_j` randomly.
         - Initialize biases `b_i` and `b~_j`.
      - **Optimization:**
         - Use stochastic gradient descent (SGD) or an adaptive optimization algorithm like AdaGrad to minimize the loss function.
         - Iteratively update vectors and biases based on the gradient of the loss function.
   - Advantages:
       - 1. Captures both global and local context of words.
       - 2. Efficient in handling large corpora.
       - 3. Produces meaningful embeddings that capture semantic relationships.

   - Disadvantages\Limitations:
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
- **FastText:** FastText, developed by Facebook AI Research (FAIR), is another popular technique for word representation in Natural Language Processing (NLP). It extends the concept of word embeddings introduced by Word2Vec by considering subword information. This approach is particularly useful for handling out-of-vocabulary words and morphologically rich languages.
  - FastText works by representing each word as a bag of character n-grams, in addition to the word itself. This allows FastText to capture the morphological structure of words, making it more robust, especially for tasks involving rare words or languages with rich morphology.
    - **How FastText Works:**
       - **Character n-grams:**
         - FastText considers all character n-grams of a word, including the word itself and special boundary symbols.
         - For example, for the word `apple` and assuming `𝑛=3`, the character n-grams would be: `<ap`, `app`, `ppl`, `ple`, `le>`, and `apple` itself.
       - **Vector Representation:**
         - Each word is represented as the sum of the vectors of its character n-grams.
         - The vectors for character n-grams are learned alongside the word embeddings during training.
       - **Word Embeddings:**
         - FastText trains word embeddings by optimizing a classification objective, typically a softmax classifier, over the entire vocabulary.
         - The context for each word is defined by the bag of its character n-grams.
       - **Training Process:**
         - FastText employs techniques like hierarchical softmax or negative sampling to efficiently train embeddings on large datasets.
    - **Implementation Steps:**
       - **Data Preparation:**
         - Tokenize the text data into words.
         - Preprocess the text by lowercasing and removing punctuation if necessary.
       - **Model Building:**
         - Use an embedding layer to represent each character n-gram.
         - Sum the embeddings of all character n-grams to obtain the word representation.
         - Concatenate word and character n-gram embeddings.
         - Apply Global Average Pooling to aggregate embeddings.
       - **Training:**
         - Train the model using a softmax classifier with a cross-entropy loss function.
         - Use techniques like hierarchical softmax or negative sampling for efficiency.
       - **Evaluation:**
         - Evaluate the trained model on downstream tasks such as text classification, sentiment analysis, etc.
   - Advantages:
       - 1. Uses character n-grams to manage out-of-vocabulary words.
       - 2. Captures word morphology, useful for languages with rich morphology.
       - 3. Effective for rare words
       - 4. Adapts to different word structures.

   - Disadvantages\Limitations:
      - 1. More memory is needed due to character n-grams.
      - 2. Embeddings are less interpretable and explainable 
      - 3. Slow inference due to additional computation
      - 4. Treats words as bags of n-grams, losing some context.

[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606v2)

- **ELMo: Embeddings from Language Models:** ELMo, short for "Embeddings from Language Models," is a deep contextualized word representation technique developed by the Allen Institute for AI. Unlike traditional word embeddings like Word2Vec and FastText, which generate static embeddings, ELMo creates word representations that dynamically change based on the context in which the words appear. This approach significantly enhances the performance of various Natural Language Processing (NLP) tasks by providing a more nuanced understanding of words and their meanings.
 - **How ELMo Works:**
   - **Contextualized Embeddings/Dynamic Representations:** Unlike static embeddings that assign a single vector to each word regardless of context, ELMo generates different vectors for a word depending on its usage in different sentences. This means that the word `bank` will have different embeddings when used in `river bank` and `savings bank.`
   - **Deep, Bi-directional Language Model:**
     - **Bi-directional LSTMs:** ELMo uses a deep bi-directional Long Short-Term Memory (bi-LSTM) network to model the word sequences. It reads the text both forward (left-to-right) and backward (right-to-left), capturing context from both directions.
     - **Layered Approach:** ELMo's architecture consists of multiple layers of LSTMs. Each layer learns increasingly complex representations, from surface-level characteristics to deeper syntactic and semantic features.
   - **Pre-trained on Large Corpora:**
     - **Massive Pre-training:** ELMo models are pre-trained on large datasets, such as the 1 Billion Word Benchmark, to learn rich linguistic patterns and structures.
     - **Fine-tuning for Specific Tasks:** After pre-training, these embeddings can be fine-tuned on specific NLP tasks, allowing ELMo to adapt to the nuances of the target task.

   - Advantages:
       - 1. Produces context-specific embeddings for more accurate word meaning.
       - 2. Captures complex syntactic and semantic information.
       - 3. Outperforms static embeddings on various NLP tasks

   - Disadvantages\Limitations:
      - 1. Requires significant resources for training and inference.
      - 2. Pre-training and fine-tuning are time-consuming.
      - 3. Large memory requirements pose deployment challenges.

## Information Retrieval 

## Vector Search 

## LLMs 

## RAG 
