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

- **ELMo:** ELMo, short for `Embeddings from Language Models,` is a deep contextualized word representation technique developed by the Allen Institute for AI. Unlike traditional word embeddings like Word2Vec and FastText, which generate static embeddings, ELMo creates word representations that dynamically change based on the context in which the words appear. This approach significantly enhances the performance of various Natural Language Processing (NLP) tasks by providing a more nuanced understanding of words and their meanings.
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
     
- [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

- **BERT:** BERT, short for `Bidirectional Encoder Representations from Transformers,` is a revolutionary language representation model developed by Google AI. Unlike previous models that process text in a unidirectional manner, BERT captures context from both directions simultaneously, providing a deeper understanding of language. This approach has set new benchmarks in various Natural Language Processing (NLP) tasks by offering more precise and comprehensive word representations.
 - **How BERT Works:**
   - **Bidirectional Contextualization:** Unlike traditional models that read text sequentially, BERT uses Transformers to process text from both the left and the right simultaneously, capturing the full context of each word.
   - **Transformer Architecture:**
     - **Self-Attention Mechanism:** BERT's architecture relies on the self-attention mechanism within Transformers, which allows the model to weigh the importance of different words in a sentence, regardless of their position.
     - **Layers of Transformers:** BERT consists of multiple layers of Transformer encoders, each providing a progressively richer representation of the text.
   - **Pre-training and Fine-tuning:**
     - **Pre-training Tasks:** BERT is pre-trained on large corpora using two unsupervised tasks: `Masked Language Modeling (MLM)` and `Next Sentence Prediction (NSP)`. MLM involves predicting masked words in a sentence, while NSP involves predicting the relationship between two sentences.
     - **Fine-tuning:** After pre-training, BERT can be fine-tuned on specific NLP tasks (e.g., question answering, sentiment analysis) by adding a task-specific output layer.
   - Advantages:
       - 1. Produces embeddings that consider the context from both directions
       - 2. Captures intricate syntactic and semantic details.
       - 3. Excels in a wide range of NLP tasks, setting new performance benchmarks.
   - Disadvantages\Limitations:
      - 1. Requires substantial computational resources for both training and inference.
      - 2. Pre-training on large datasets is time-consuming and computationally expensive.
      - 3. The large model size demands significant memory, complicating deployment in resource-constrained environments.
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

### Deep NLP 
- **Deep Learning - Basic Neural Network Components:** A basic neural network architecture includes:
   - **Input Layer:** Receives input data.
   - **Hidden Layers:** Process input through weighted connections.
   - **Activation Function:** Adds non-linearity to neuron outputs.
   - **Weights and Biases:** Parameters adjusted during training.
   - **Output Layer:** Produces the final output.
   - **Loss Function:** Measures the difference between predicted and actual output.
   - **Optimization Algorithm:** Updates weights to minimize loss.
   - **Learning Rate:** Controls the step size during optimization.

- **Purpose of Activation Functions in Neural Networks:** Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Without them, the network would only perform linear transformations, limiting its ability to model complex data.
- **Common Activation Functions:**
    - **Sigmoid Function:**
      - `Formula: f(x) = 1 / (1 + e^(-x))`
      - Use Cases: Often used in the output layer for binary classification.
```python
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
   - **Tanh Function::**
       - `Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
       - Use Cases: Useful for hidden layers, especially in RNNs, as it maps input to a range between -1 and 1.
```python
import numpy as np
def tanh(x):
    return np.tanh(x)
```

   - **ReLU (Rectified Linear Unit):**
        - `Formula: f(x) = max(0, x)`
        - Use Cases: Commonly used in hidden layers for its simplicity and efficiency.
```python
import numpy as np
def relu(x):
    return np.maximum(0, x)
```

   - **Leaky ReLU:**
        - `Formula: f(x) = max(αx, x) where α is a small positive constant.`
        - Use Cases: Prevents neurons from dying during training.

```python
import numpy as np
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

   - **ELU (Exponential Linear Unit):**
        - `Formula: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0`
        - Use Cases: Helps avoid dead neurons and provides smooth gradient updates.

```python
import numpy as np
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

   - **Softmax Function:**
        - `Formula: softmax(z_i) = e^(z_i) / ∑(e^(z_j))`
        - Use Cases: Used in the output layer for multi-class classification tasks.


```python
import numpy as np
def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)
```

- **Important Optimization Algorithms for NLP:** In Natural Language Processing (NLP), optimization algorithms are crucial for training models effectively. The purpose of optimization algorithms in deep learning is to minimize the loss function, improving the model's ability to make accurate predictions by adjusting its parameters iteratively during training. Below are some of the most important optimization algorithms commonly used in NLP, along with their descriptions and implementations.
 -  **Gradient Descent:** Gradient Descent is a fundamental optimization algorithm used to minimize the loss function by iteratively moving towards the steepest descent direction of the gradient.
    -  **Algorithm:**
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

-  **Comparison of Gradient Descent Algorithms:**
   - **Standard Gradient Descent:**
     -  **Description**: Iteratively updates parameters by computing the gradient of the loss function over the entire dataset.
     - **Update Rule**: `theta = theta - learning_rate * gradient`
     - **Advantages**: Stable convergence for convex functions.
     - **Disadvantages**: Slow for large datasets; each iteration is computationally expensive.

   - **Stochastic Gradient Descent (SGD):**
     - **Description**: Updates parameters using the gradient computed from a single training example, providing more frequent updates.
     - **Update Rule**: `theta = theta - learning_rate * gradient (single example)`
     - **Advantages**: Faster convergence on large datasets; can escape local minima.
     - **Disadvantages**: Can be noisy, leading to fluctuations in the loss function.
   - **Mini-Batch Gradient Descent:**
     - **Description**: Updates parameters using the gradient computed from a small subset (mini-batch) of training examples.
     - **Update Rule**: `theta = theta - learning_rate * gradient (mini-batch)`
     - **Advantages**: Reduces variance in updates, providing a balance between SGD and batch GD.
     - **Disadvantages**: Requires choosing an appropriate batch size; computational cost is still significant.

   - **Momentum-based Gradient Descent:**
     - **Description**: Incorporates a momentum term that helps accelerate updates in relevant directions and dampens oscillations.
     - **Update Rule**: 
      - `velocity = momentum * velocity + gradient`
      - `theta = theta - learning_rate * velocity`
     - **Advantages**: Speeds up convergence; reduces oscillations near minima.
     - **Disadvantages**: Requires tuning of the momentum parameter (momentum); can overshoot minima.

- **Feedforward Neural Networks (FNN):** A Feedforward Neural Network (FNN) is the simplest form of artificial neural network. In this type of network, the information moves in only one direction—forward—from the input nodes, through the hidden nodes (if any), and to the output nodes. There are no cycles or loops in the network, making it straightforward to understand and implement.
  - **Architecture:** The basic architecture of a Feedforward Neural Network consists of the following components:
    - 1. **Input Layer**: This layer receives the input data. Each node in this layer represents one feature of the input.
    - 2. **Hidden Layers**: These layers process the input data. There can be one or more hidden layers in an FNN. Each node in a hidden layer applies a weighted sum of its inputs and an activation function.
    - 3. **Output Layer**: This layer produces the final output of the network. The number of nodes in this layer corresponds to the number of output classes or the required output dimensions.

  - **Mathematically:**
   - For a single hidden layer network:
      - `Input Layer: x`
      - `Hidden Layer: [ h = σ(W1x + b1) ]`
      - `Output Layer: [ y = W2h + b2 ]`
    - Where:
      - `W1` is the weight matrix connecting the input layer to the hidden layer
      - `b1` is the bias vector for the hidden layer
      - `σ` is an activation function (e.g. ReLU, Sigmoid)
      - `W2` is the weight matrix connecting the hidden layer to the output layer
      - `b2` is the bias vector for the output layer
   - Advantages:
       - 1. Easy to implement and understand.
       - 2. Can approximate any continuous function with enough layers and units.
       - 3. Effective for classification and regression on structured data.
       - 4. Predictable and easier to debug due to one-way data flow.
   - Disadvantages\Limitations:
      - 1. Not suitable for sequential data like time series or text.
      - 2. Cannot retain information from previous inputs.
      - 3. Prone to overfitting, especially with limited data

 ``` Python 
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
```
Output: 0.772853461396208
```
- **Recurrent Neural Networks (RNN):** A Recurrent Neural Network (RNN) is a type of artificial neural network designed for sequential data. Unlike Feedforward Neural Networks, RNNs have connections that form directed cycles, allowing them to maintain information about previous inputs through internal states. This makes them particularly suitable for tasks where the context or order of data is crucial, such as time series prediction, natural language processing, and speech recognition.
  - **Architecture:** The basic architecture of a Recurrent Neural Network consists of the following components:
    - 1. **Input Layer**:  This layer receives the input data. In the context of sequences, each input is often processed one time step at a time.
    - 2. **Hidden Layers**: These layers process the input data and maintain a memory of previous inputs. Each node in a hidden layer takes input not only from the current time step but also from the hidden state of the previous time step.
    - 3. **Cell State:** This is a vector that stores the internal memory of the network. The cell state is updated at each time step based on the current input and the previous cell state
    - 3. **Output Layer**: This layer produces the final output of the network for each time step. The number of nodes in this layer corresponds to the desired output dimensions.

  - **Mathematically:**
   - For a single hidden layer RNN:
      - `Input Layer: x`
      - `Hidden State: [ h = σ(Wx * x + Wh * h + b) ]`
      - `Cell State: [ c = f(c_prev, x) ]`
      - `Output Layer: [ y = σ(Wy * h + b) ]`
   - Where:
      - `Wx` is the weight matrix connecting the input layer to the hidden state
      - `Wh` is the weight matrix connecting the hidden state to itself
      - `b` is the bias vector for the hidden state
      - `f` is the forget gate function (e.g. sigmoid)
      - `Wy` is the weight matrix connecting the hidden state to the output layer
      - `σ` is an activation function (e.g. ReLU, Sigmoid)
![RNN illustrated with this Image example](https://github.com/KaifAhmad1/Awesome-NLP-and-IR/blob/main/images/RNN.png)

   - Advantages:
       - 1. Suitable for sequential data like time series or text.
       - 2. Can retain information from previous inputs
       - 3. Effective for modeling temporal dependencies in data.
   - Disadvantages\Limitations:
      - 1. Prone to vanishing gradients, which can make training difficult.
      - 2. Difficult to train due to the complex recurrent connections.
      - 3. Not suitable for very long sequences.

``` Python 
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
```
Output: [[0.53499449]]
```
- **Different types of RNNs:** Over the years, several variants of RNNs have been developed to address various challenges and improve their performance. Here are some of the most prominent RNN variants:
   - 1. **Vanilla RNNs\RNNs:** I have talked about above. 
   - 2. **LSTMs:**
   - 3. **GRUs:**
   - 4. **Bidirectional LSTMs and GRUs:**
  
- **Long Short-Term Memory (LSTM) Networks:** A Long Short-Term Memory (LSTM) network is a type of Recurrent Neural Network (RNN) designed to handle the vanishing gradient problem that occurs in traditional RNNs. LSTMs are capable of learning long-term dependencies in data, making them particularly useful for tasks such as language modeling, speech recognition, and time series forecasting.
  - **Architecture:** The basic architecture of an LSTM network consists of the following components:
    - **Input Gate:** This gate is responsible for controlling the flow of new information into the cell state. It consists of a sigmoid layer and a point-wise multiplication operation.
    - **Forget Gate:** This gate determines what information to discard from the previous cell state. It also consists of a sigmoid layer and a point-wise multiplication operation.
    - **Cell State:** This is the internal memory of the LSTM network, which stores information over long periods.
    - **Output Gate:** This gate determines the output of the LSTM network based on the cell state and the hidden state.
    - **Hidden State:** This is the internal state of the LSTM network, which captures short-term dependencies in the data.
   - **Mathematically:** The LSTM network can be represented mathematically as follows:
      - `Input Gate: i = σ(Wi * x + Ui * h + bi)`
      - `Forget Gate: f = σ(Wf * x + Uf * h + bf)`
      - `Cell State: c = f * c_prev + i * tanh(Wc * x + Uc * h + bc)`
      - `Output Gate: o = σ(Wo * x + Uo * h + bo)`
      - `Hidden State: h = o * tanh(c)`
      - `Output: y = Wo * h + bo`
   - **Where:**
      - `Wi, Wf, Wc, Wo` are the weight matrices for the input, forget, cell, and output gates, respectively.
      - `Ui, Uf, Uc, Uo` are the recurrent weight matrices for the input, forget, cell, and output gates, respectively.
      - `bi, bf, bc, bo` are the bias vectors for the input, forget, cell, and output gates, respectively.
      - `σ` is the sigmoid activation function.
      - `tanh` is the hyperbolic tangent activation function.
   - Advantages:
       - 1. LSTMs are capable of learning long-term dependencies in data, making them suitable for tasks such as language modeling and time series forecasting.
       - 2. They are less prone to the vanishing gradient problem compared to traditional RNNs.
       - 3. LSTMs can handle sequential data with varying lengths.
   - Disadvantages\Limitations:
      - 1. LSTMs are computationally expensive to train and require large amounts of data.
      - 2. They can be difficult to interpret and visualize due to their complex architecture.
      - 3. LSTMs can suffer from overfitting if not regularized properly.
       
``` Python 
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
```
Output: [[0.54412203]]
```
- **Gated Recurrent Unit(GRU) Networks:** A Gated Recurrent Unit (GRU) network is a type of Recurrent Neural Network (RNN) designed to handle the vanishing gradient problem that occurs in traditional RNNs. GRUs are capable of learning long-term dependencies in data, making them particularly useful for tasks such as language modeling, speech recognition, and time series forecasting. GRUs are a simplified version of Long Short-Term Memory (LSTM) networks, with fewer gates and a more streamlined architecture.
   - **Architecture:** The basic architecture of a GRU consists of the following components:
      - **Reset Gate:** This gate determines what information to discard from the previous hidden state. It consists of a sigmoid layer and a point-wise multiplication operation.
      - **Update Gate:** This gate determines what information to update in the hidden state. It consists of a sigmoid layer and a point-wise multiplication operation.
      - **Hidden State:** This is the internal state of the GRU network, which captures short-term dependencies in the data.
      - **Output:** This is the output of the GRU network, which is calculated based on the hidden state.
   - **Mathematically:** The GRU network can be represented mathematically as follows
     - `Reset Gate: r = σ(Wr * x + Ur * h + br)`
     - `Update Gate: z = σ(Wz * x + Uz * h + bz)`
     - `Hidden State: h = (1 - z) * h_prev + z * tanh(W * x + U * h + b)`
     - `Output: y = W * h + b`
   - **Where:**
     - `Wr, Wz, W` are the weight matrices for the reset, update, and output gates, respectively.
     - `Ur, Uz, U` are the recurrent weight matrices for the reset, update, and output gates, respectively.
     - `br, bz, b` are the bias vectors for the reset, update, and output gates, respectively.
     - `σ` is the sigmoid activation function.
     - `tanh` is the hyperbolic tangent activation function.
   - Advantages:
       - 1. GRUs are capable of learning long-term dependencies in data, making them suitable for tasks such as language modeling and time series forecasting.
       - 2. They are less prone to the vanishing gradient problem compared to traditional RNNs.
       - 3. GRUs have a simpler architecture compared to LSTMs, with fewer parameters to train, making them computationally more efficient.
   - Disadvantages\Limitations:
      - 1. GRUs, like LSTMs, can be computationally expensive to train and require large amounts of data.
      - 2. They can be difficult to interpret and visualize due to their complex architecture.
      - 3. GRUs can suffer from overfitting if not regularized properly.
``` Python 
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
```
Output: [[0.40272816]]
```
- **Bidirectional RNNs:** Bidirectional Recurrent Neural Networks (BRNNs) improve upon traditional RNNs by considering both past and future information in their predictions. This makes them highly effective for tasks involving sequential data, such as text and time series.
   - **Architecture:** BRNNs consist of two RNNs: one processes the input sequence forward, and the other processes it backwards. The outputs of these RNNs are concatenated to form the final output, allowing the network to use information from both directions.
    - **Types:**
      - **LSTM (Long Short-Term Memory):** Effective for learning long-term dependencies, ideal for tasks like language modeling and speech recognition.
      - **GRU (Gated Recurrent Unit):** Simpler and more computationally efficient than LSTMs, suitable for tasks like text classification and sentiment analysis.
    - Mathematical Representation:
      - `Forward RNN: h_forward = LSTM(x, W, U, b) or GRU(x, W, U, b)`
      - `Backward RNN: h_backward = LSTM(x, W, U, b) or GRU(x, W, U, b)`
      - `Output: y = Concat(h_forward, h_backward)`
    - Where:
      - `x` is the input sequence
      - `W`, `U`, and `b` are the weights, recurrent weights, and biases
      - `h_forward` and `h_backward` are the hidden states
      - `y` is the output
    - Advantages:
       - 1. Capture context from both past and future
       - 2. Handle variable-length sequential data
       - 3. Learn long-term dependencies
    - Disadvantages\Limitations:
      - 1. Computationally intensive
      - 2. Require large datasets
      - 3. Complex architecture can lead to overfitting

```  Python
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
import numpy as np

# input sequence
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
```
Epoch 1/10
1/1 [==============================] - 14s 14s/step - loss: 0.1366
Epoch 2/10
1/1 [==============================] - 0s 21ms/step - loss: 0.1362
Epoch 3/10
1/1 [==============================] - 0s 21ms/step - loss: 0.1358
Epoch 4/10
1/1 [==============================] - 0s 23ms/step - loss: 0.1353
Epoch 5/10
1/1 [==============================] - 0s 28ms/step - loss: 0.1349
Epoch 6/10
1/1 [==============================] - 0s 19ms/step - loss: 0.1345
Epoch 7/10
1/1 [==============================] - 0s 20ms/step - loss: 0.1341
Epoch 8/10
1/1 [==============================] - 0s 23ms/step - loss: 0.1337
Epoch 9/10
1/1 [==============================] - 0s 22ms/step - loss: 0.1333
Epoch 10/10
1/1 [==============================] - 0s 21ms/step - loss: 0.1328
1/1 [==============================] - 2s 2s/step
Output: [[[ 0.00630986  0.01720074  0.01832638  0.00959984  0.02938063
   -0.00989505  0.01275288  0.04869077  0.00156058 -0.00256061
   -0.01736601 -0.05287949  0.05367433  0.06530365 -0.03861162
   -0.04156534  0.09283099  0.03927685  0.05287885 -0.03797476]
  [ 0.0087546   0.01840007  0.02062777  0.01294859  0.02938948
   -0.00448973  0.01563004  0.04745301  0.00184349  0.00214797
   -0.00804165 -0.0242324   0.03008071  0.03599263 -0.01945202
   -0.01689252  0.04573838  0.02196954  0.02946377 -0.01660443]
  [ 0.01434947  0.02994058  0.03340437  0.0187251   0.04273633
   -0.0080414   0.02500243  0.07809883  0.00132269  0.00235718
   -0.0058728  -0.02651676  0.02246626  0.02886551 -0.01656632
   -0.02470564  0.04360295  0.01756983  0.02010829 -0.01958983]]]
```
- **Transformers:** Transformers are a type of deep learning model introduced by Ashish Vaswani in the paper "Attention is All You Need" (2017). They are particularly powerful for handling sequential data, such as text, but unlike RNNs, they do not process data in a sequential manner. Instead, Transformers use self-attention mechanisms to model dependencies between all elements of the input sequence simultaneously, allowing for much greater parallelization during training.
- **Secret Sauce: Self-Attention Mechanism** Self-attention is a key component of the Transformer architecture, which enables the model to weigh the significance of different parts of the input sequence when processing each element. This mechanism helps the model capture relationships and dependencies between all elements in the sequence, regardless of their distance from each other.
  - **Steps:**
    - 1. **Input Sentence:** We start with the sentence: `She opened the door to the garden.`
    - 2. **Convert Words to Vectors:** We start with the sentence: `She opened the door to the garden.` Each word in this sentence is represented as a vector after being passed through an embedding layer. Let's use simplified vector representations for clarity:
          - `She: [1, 0, 0]`
          - `opened: [0, 1, 0]`
          - `the: [0, 0, 1]`
          - `door: [1, 1, 0]`
          - `to: [1, 0, 1]`
          - `the: [0, 1, 1]`
          - `garden: [1, 1, 1]`
    - 3. **Creating Q, K, and V Matrices:** For each word, we create three vectors: `Query (Q), Key (K)`, and `Value (V)`. These vectors are derived by multiplying the word vector by three different weight matrices `(W_Q, W_K, W_V)`.
       - **Query (Q):** Represents the word we are currently processing.
       - **Key (K):** Represents the words we compare the current word against.
       - **Value (V):** Represents the actual content of the words
         -  These vectors are derived by multiplying the word vector by weight matrices (W_Q, W_K, W_V).
         - Example for "opened":
          - `Q_opened = [0, 1, 0] * W_Q`
          - `K_opened = [0, 1, 0] * W_K`
          - `V_opened = [0, 1, 0] * W_V`
     - 4. **Calculating Attention Scores:** Dot product of the Query vector of the word with Key vectors of all words.
      - Example for `opened`:
        - Score for `She` = `Q_opened · K_She`
        - Score for `opened` = `Q_opened · K_opened`
        - Score for `the` = `Q_opened · K_the`
     - 5. **Applying Softmax:**
        - Pass attention scores through the Softmax function to get attention weights.
          - Softmax formula: `softmax(x_i) = exp(x_i) / sum(exp(x))`
     - 6. **Weighted Sum of Values:**
        - Multiply Value vectors by their corresponding attention weights.
          - Example for `opened`:
          - `Weighted sum = softmax(Scores) * [V_She, V_opened, V_the, V_door, V_to, V_the, V_garden]`
  - By iteratively performing these steps for all words in the input sentence, the self-attention mechanism captures intricate relationships and dependencies across the entire sequence, facilitating effective sequence-to-sequence processing tasks like language translation or text generation.

- **Key Components of Transformer Architecture:** 
The Transformer architecture consists of an encoder and a decoder, both composed of multiple identical layers. Each layer in both the encoder and decoder contains two main sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.
  - **Encoder:** The encoder processes the input sequence and generates a set of feature representations for each element in the sequence. It consists of:
    - **Input Embedding:** Converts input tokens into dense vectors.
    - **Positional Encoding:** Adds information about the position of each token in the sequence, since the model does not inherently capture sequence order.
    - **Multi-Head Self-Attention:** Allows the model to focus on different parts of the sequence simultaneously. Each head processes the sequence differently, and the results are concatenated and linearly transformed.
    - **Feed-Forward Network:** Applies two linear transformations with a ReLU activation in between, applied to each position separately.
    - **Layer Normalization:** Normalizes the output of each sub-layer (attention and feed-forward).
    - **Residual Connection:** Adds the input of each sub-layer to its output, aiding in training deeper networks.
   - **Mathematically:** Mathematically, for each sub-layer:
   - **Self-Attention:**
      - `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`
   - Where: 
      - `Q`, `K`, and `V` are the query, key, and value matrices, respectively.
      - `d_k` is the dimension of the key/query vectors.
   - **Feed-Forward Network:**
      - `FFN(x) = max(0, xW1 + b1) W2 + b2`

  - **Decoder:** The decoder generates the output sequence, one token at a time, using the encoded representations and the previously generated tokens. It consists of:
    - **Output Embedding:** Converts output tokens into dense vectors.
    - **Positional Encoding:** Adds information about the position of each token in the sequence, since the model does not inherently capture sequence order. Similar to the encoder's positional encoding.
    - **Masked Multi-Head Self-Attention:** Prevents attending to future tokens by masking them.
    - **Multi-Head Attention:** Attends to the encoder's output representations.
    - **Feed-Forward Network:** Applies two linear transformations with a ReLU activation in between, applied to each position separately.
    - **Layer Normalization:** Normalizes the output of each sub-layer (attention and feed-forward).
    - **Residual Connection:** Adds the input of each sub-layer to its output, aiding in training deeper networks.
   - **Mathematically:** Mathematically, for each sub-layer:
   - **Attention Mechanism:** The attention mechanism allows the model to weigh the importance of different tokens when processing a sequence. In the Transformer, the scaled dot-product attention is used:
      - `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`
   - **Multi-Head Attention:** To allow the model to focus on different positions and features, the Transformer uses multi-head attention:
     - `MultiHead(Q, K, V) = Concat(head1, ..., headh) WO` 
   - **Where:**
     - `headi = Attention(QWiQ, KWiK, VWiV)`

    - Advantages:
       - 1. **Parallelization:** Unlike RNNs, Transformers can process all tokens in a sequence simultaneously, allowing for faster training.
       - 2. **Long-Range Dependencies:** The self-attention mechanism can capture long-range dependencies more effectively than RNNs.
       - 3. **Scalability:** Scales well with larger datasets and model sizes.
    - Disadvantages\Limitations:
      - 1. **Computational Cost:** Self-attention has a quadratic complexity with respect to the sequence length, making it computationally expensive for long sequences.
      - 2. **Memory Usage:** Requires significant memory to store the attention weights.

- **Transformer Architectures: A Detailed Comparison**

Transformers have become a dominant architecture in the field of natural language processing (NLP), with various flavours and applications. Below is a comparison of the key differences between the three main transformer architectures:

| **Aspect**                           | **Encoder-Style Transformer**                                     | **Decoder-Style Transformer**                                     | **Encoder-Decoder Style Transformer**                               |
|--------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|--------------------------------------------------------------------|
| **Structure**                        | Multiple layers of encoders, without any decoders                 | Multiple layers of decoders, without any encoders                 | Separate stacks of encoders and decoders                           |
| **Primary Usage**                    | Representation learning and classification tasks                  | Generative tasks (e.g., autoregressive text generation)           | Sequence-to-sequence tasks (e.g., translation, summarization)      |
| **Examples**               | BERT , RoBERTa, DistilBERT   | GPT Series by OpenAI, Llama Series by Meta, Mistral Etc.                          | Transformer, BART, T5          |
| **Attention Mechanism**              | Self-attention within each encoder layer                          | Self-attention within each decoder layer, with masked attention   | Self-attention in encoders, cross-attention in decoders            |
| **Training Objective**               | Masked language modeling and next sentence prediction             | Causal Language modeling (predicting the next token)                     | Supervised learning with source and target sequences               |
| **Advantages**                       | - Good at capturing bidirectional context                         | - Effective at generating coherent text                           | - Effective at learning mappings between input and output sequences|
|                                      | - Effective for understanding tasks (e.g., sentiment analysis)    | - Can handle long text generation                                 | - Handles both input and output dependencies effectively           |
|                                      | - Pre-training can be easily adapted to various downstream tasks  |                                                                   | - Versatile for various tasks                                      |
| **Limitations**                      | - Not designed for generative tasks                               | - Unidirectional context                                          | - More complex architecture                                        |
|                                      | - Requires large datasets for pre-training                        | - Can suffer from exposure bias                                   | - Computationally intensive                                        |
| **Applications**                     | - Text classification                                             | - Text generation                                                 | - Machine translation                                              |
|                                      | - Named entity recognition                                        | - Dialogue systems                                                | - Text summarization                                               |
|                                      | - Sentence embedding                                              | - Story completion                                                | - Speech recognition                                               |

This table provides a clear comparison of the different transformer architectures, their use cases, techniques, advantages, and limitations.


## Vector Search: A Comprehensive Overview
- **What is a Vector?:** In mathematics, a vector is a quantity defined by both magnitude and direction. Vectors are represented as arrays of numbers, which correspond to coordinates in a multidimensional space. They are foundational in various fields, including physics, engineering, and computer science.
  - Typically represented as `V = [v1, v2, v3, ...., vn]` where `n` is the magnitude of the vector in high dimensional space.
   - **Basic Properties of Vectors:**
     - **Magnitude:** The length of the vector.
     - **Direction:** The orientation of the vector in space.
  ![Vector Representation in High Dimensional Space](https://github.com/KaifAhmad1/Awesome-NLP-and-IR/blob/main/images/Vector%20Representation.png)
  - **Vector Representation in Machine Learning** In machine learning and information retrieval, vectors are used to represent data points in a high-dimensional space. This representation is crucial for tasks like similarity search, where the goal is to find data points that are similar to a given query.
    - 1. **Text Data:**
       - **Word Embeddings:** Words are mapped to vectors using models like Word2Vec, GloVe, or FastText. These vectors capture semantic meanings, where similar words have similar vectors.
       - **Sentence Embeddings:**  Models like BERT and GPT transform entire sentences or documents into vectors, preserving contextual meaning.
     - 2. **Image Data:**
       - **Feature Vectors:** Convolutional Neural Networks (CNNs) are used to extract features from images, which are then represented as vectors.
     - 3. **Audio and Video Data:**
       - **Audio Vectors:** Deep learning models like VGGish convert audio signals into vectors that capture the essential characteristics of the sound.
       - **Video Vectors:** Similar to images, videos are processed frame by frame or using 3D CNNs to generate vectors representing the video content.
  - **Distance Metrics:** Distance metrics are used to quantify the similarity or dissimilarity between vectors. Different metrics are suited for different types of data and applications.
    - 1. **Euclidean Distance:** Measures the straight-line distance between two points in Euclidean space.
          - `Formula: √(Σ (v_i - u_i)^2)`
        -  For vectors V = [1, 2] and U = [4, 6], the Euclidean distance is √((4-1)^2 + (6-2)^2) = √(9 + 16) = √25 = 5.
           - **Advantages:**
             - Intuitive and easy to compute.
             - Well-suited for small, low-dimensional datasets.
           - **Limitations:**
             - Sensitive to differences in magnitude and scaling.
             - Not suitable for high-dimensional spaces due to the curse of dimensionality, where distances become less meaningful.
    - 2. **Manhattan Distance:**  Measures the distance between two points along axes at right angles, also known as L1 distance or taxicab distance.
          - `Formula:  Σ |v_i - u_i|`
        -  For vectors V = [1, 2] and U = [4, 6], the Manhattan distance is |4-1| + |6-2| = 3 + 4 = 7.
           - **Advantages:**
             - Robust to outliers and useful in grid-based pathfinding problems, such as robotics and game design.
           - **Limitations:**
             - Can be less intuitive for non-grid-based data.
             - Sensitive to scale, like Euclidean distance.

    - 3. **Cosine Similarity:**  Measures the cosine of the angle between two vectors, indicating their similarity in terms of direction rather than magnitude.
          - `Formula: cos(θ) = (v ⋅ u) / (||v|| ||u||)`
        -  For vectors V = [1, 2] and U = [2, 3], the cosine similarity is (12 + 23) / (√(1^2 + 2^2) * √(2^2 + 3^2)) = 8 / (√5 * √13) ≈ 0.98.
           - **Advantages:**
             - Useful for high-dimensional data, such as text data represented as word vectors.
             - Ignores magnitude, focusing on the direction of the vectors.
           - **Limitations:**
             - Ignores magnitude, which can be a drawback if magnitude differences are important.
             - Requires non-zero vectors to compute.
    - 4. **Jaccard Similarity:**  Measures the similarity between finite sets by considering the size of the intersection divided by the size of the union of the sets.
          - `Formula: J(A, B) = |A ∩ B| / |A ∪ B|`
        -  For sets A = {1, 2, 3} and B = {2, 3, 4}, the Jaccard similarity is |{2, 3}| / |{1, 2, 3, 4}| = 2 / 4 = 0.5.
           - **Advantages:**
             - Handles binary or categorical data well.
             - Simple interpretation and calculation.
           - **Limitations:**
             - Not suitable for continuous data.
             - Can be less informative for datasets with many common elements.
    - 5. **Hamming Distance:**  Measures the number of positions at which the corresponding elements of two binary vectors are different.
          - `Formula: H(v, u) = Σ (v_i ≠ u_i)`
        -  For binary vectors V = [1, 0, 1] and U = [0, 1, 1], the Hamming distance is (1 ≠ 0) + (0 ≠ 1) + (1 = 1) = 2.
           - **Advantages:**
             - Effective for error detection and correction in binary data.
             - Simple and fast to compute.
           - **Limitations:**
             - Only applicable to binary vectors.
             - Not useful for continuous or non-binary categorical data.
    - 6. **Earth Mover's Distance (EMD):**  Measures the minimum amount of `work` needed to transform one distribution into another, often used in image retrieval. Also known as the Wasserstein distance.
          - `Formula: EMD(P, Q) = inf_γ ∫_X×Y d(x,y) dγ(x,y)`
        -  Given two distributions of points, EMD calculates the cost of moving distributions to match each other. For instance, if distribution 𝑃 has points [1,2] and 𝑄 has points [2,3], EMD would calculate the minimal transportation cost.
           - **Advantages:**
             - Provides a meaningful metric for comparing distributions, taking into account the underlying geometry.
             - Applicable to various types of data, including images and histograms.
           - **Limitations:**
             - Computationally intensive, especially for large datasets.
             - Requires solving an optimization problem, which can be complex.


  - **Vector Search Techniques:** Vector search involves finding vectors in a database that are similar to a given query vector. Techniques include:
   - 1. **Brute-Force Search:**
      - Computes similarity between the query vector and all vectors in the dataset.
      - Inefficient for large datasets due to high computational cost.
   - 2. **k-Nearest Neighbors (k-NN):**
      - Finds the k vectors closest to the query vector.
      - Can be implemented using efficient data structures like KD-Trees or Ball Trees for lower-dimensional data.
   - 3. **Approximate Nearest Neighbor (ANN):**
      - Speeds up search by approximating the nearest neighbours.
      - Methods include Locality-Sensitive Hashing (LSH) and Hierarchical Navigable Small World (HNSW) graphs.
  - **Applications of Vector Search:** Vector search is transforming various industries by enabling more accurate and context-aware search functionalities:
    - **Search Engines:**
      - Enhance traditional keyword-based searches by incorporating semantic understanding.
      - Google’s BERT and MUM models are examples of using vector search to improve search relevance.
    - **E-commerce:**
      - Improve product recommendations by understanding user preferences and product features through vector embeddings.
      - Amazon and other retailers use vector search to provide contextually relevant search results.
    - **Content Platforms:**
      - Platforms like Spotify and YouTube use vector search to recommend music and videos based on user behavior and preferences.
    - **Healthcare:**
      - Retrieve relevant medical documents, research papers, and clinical notes to support diagnostics and treatment planning.

- **Nearest Neighbor Search**: Nearest neighbor search is a fundamental technique used to identify the closest data points to a given query point within a dataset. It is essential in various applications such as recommendation systems, image and video retrieval, and machine learning classification tasks.
   - **Example:** In a recommendation system, nearest neighbor search helps find users with similar preferences, enabling the system to suggest products or services that align with a user's tastes. For instance, Netflix recommends movies by identifying viewers with similar viewing habits and suggesting what others with similar preferences have enjoyed.

- **High-Dimensional Data:** High-dimensional data refers to datasets with a large number of features or dimensions, such as text data represented by word embeddings or image data characterized by pixel values. Analyzing and managing high-dimensional data presents several challenges:
  - **Increased Computational Complexity:** The number of calculations required increases exponentially with the number of dimensions, leading to significant computational costs.
  - **Data Sparsity:** As dimensions increase, data points become sparse, making it difficult to draw meaningful comparisons.
  - **Overfitting:** With a large number of features, models may capture noise rather than underlying patterns, resulting in overfitting.
In image search, each image can be represented as a high-dimensional vector. Comparing these vectors directly is computationally intensive due to the vast number of dimensions involved.

- **Curse of Dimensionality:** The curse of dimensionality, a term coined by `Richard Bellman`, describes the various phenomena that arise when analyzing data in high-dimensional spaces. As the number of dimensions increases:
  - **Distance Measures Become Less Meaningful:** In high-dimensional spaces, the distance between data points becomes more uniform, making it difficult to differentiate between the nearest and farthest neighbours.
  - **Volume of Space Increases Exponentially:** The volume of the space grows exponentially with the number of dimensions, causing data points to become sparse and reducing statistical significance.
  - **Increased Noise and Redundancy:** Higher dimensions can introduce more noise and redundant information, complicating the learning process and degrading the performance of algorithms.
- **Example:** Consider a facial recognition system operating in high-dimensional space. The Euclidean distance between facial vectors becomes less effective, necessitating more advanced techniques to accurately measure similarity. This phenomenon illustrates the need for innovative solutions to manage high-dimensional data efficiently.

- **Linear Search:** Linear search is a straightforward method for finding a specific element in a vector (or array) by checking each element sequentially until the desired element is found or the end of the vector is reached. It operates in a vector space, which is essentially a one-dimensional array of elements.
   - **Mathematical Explanation:** Given a vector `V = [v1, v2, ..., vn]` and a target element `t`, the linear search algorithm checks each element vi in V sequentially:
      - 1.  Start from the first element: i = 1
      - 2.  Compare t with vi.
      - 3.  If `t = vi`, the search is successful, and the position i is returned.
      - 4. If `t ≠ vi`, increment i and repeat steps 2-3 until i = n or t is found.
The Time Complexity of Linear Search is Linear O(n) and the Space Complexity is Constant O(1)
    - Advantages:
       - 1. Linear search is straightforward to implement and understand.
       - 2. Linear search does not require the dataset to be sorted or preprocessed in any way.
       - 3. Linear search can be used on any type of dataset, regardless of structure or order.
    - Limitations:
      - 1. Linear search is inefficient for large datasets because it requires checking each element sequentially.
      - 2. For large datasets, linear search can be very slow compared to other search algorithms KNN search or hash-based searches.
```  Python 
def linear_search(vector, target):
    for i in range(len(vector)):
        if vector[i] == target:
            return i
    return -1

# Input 
vector = [4, 2, 9, 1, 5]
target = 9
result = linear_search(vector, target)
if result != -1:
    print(f"Element found at index {result}")
else:
    print("Element not found")
```
- **Dimensionality Reduction:** 
- Dimensionality reduction is a fundamental technique in data analysis and machine learning, aimed at transforming high-dimensional data into a lower-dimensional representation while preserving its essential characteristics. This process offers several advantages, including enhanced computational efficiency, improved model performance, and better visualization of complex datasets.
- Reducing dimensions helps address the Curse of Dimensionality by making data more interpretable and patterns more discernible. It also boosts computational efficiency by reducing complexity, leading to faster algorithms. Furthermore, it improves model performance by focusing on relevant features and mitigating overfitting.
- Dimensionality reduction techniques like PCA and t-SNE facilitate data visualization by projecting high-dimensional data into lower-dimensional spaces, making complex relationships easier to understand.
  
- **Principal Component Analysis:** PCA is a widely used technique for linear dimensionality reduction. It aims to find the directions, or principal components, in which the data varies the most and projects the data onto these components to obtain a lower-dimensional representation.
  - At its core, PCA seeks to transform high-dimensional data into a lower-dimensional form while preserving the most important information. It achieves this by identifying the directions in which the data varies the most, known as the principal components, and projecting the data onto these components.
- **Mathematical Foundation:**
   - **Centering the Data:** PCA begins by centering the data, which involves subtracting the mean vector `( Xmean )` from each sample.
   - **Covariance Matrix:** Next, it computes the covariance matrix `( C )` of the centered data. This matrix quantifies the relationships between different features and how they vary together.
   - **Eigen Decomposition:** PCA then proceeds to compute the eigenvectors and eigenvalues of the covariance matrix. These eigenvectors represent the principal components, and the corresponding eigenvalues indicate the amount of variance explained by each component.

- **Steps in PCA:** 
  - 1. **Standardization:** Center the data by subtracting the mean vector from each sample.
  - 2. **Covariance Matrix Computation:** Compute the covariance matrix of the centered data.
  - 3. **Eigen Decomposition:** Compute the eigenvectors and eigenvalues of the covariance matrix.
  - 4. **Selection of Principal Components:** Select the top 𝑘 eigenvectors based on their corresponding eigenvalues to form the new feature space.
  - 5. **Transformation:** Project the original data onto the selected principal components to obtain the lower-dimensional representation.
 ``` Python
import numpy as np

# Create a random dataset
np.random.seed(0)
X = np.random.rand(100, 3)

# Center the data
X_centered = X - np.mean(X, axis=0)
# Compute the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)
# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# Sort eigenvectors based on eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
# Select top 2 eigenvectors
k = 2
top_k_eigenvectors = sorted_eigenvectors[:, :k]
# Transform original data
X_transformed = np.dot(X_centered, top_k_eigenvectors)
 ```

   - Advantages:
       - 1. Simplifies models and reduces computational costs.
       - 2. Filters out noise, improving data quality.
       - 3. Eases visualization of high-dimensional data.
       - 4. Identifies significant features for better model performance.
   - Limitations:
      - 1. Assumes linear relationships, missing non-linear patterns.
      - 2. Principal components may be hard to interpret
      - 3. Requires standardized data.
      - 4. Captures variance, not necessarily the most important features for all tasks.

# t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a machine learning algorithm primarily used for dimensionality reduction and visualizing high-dimensional data. It is a non-linear technique particularly well-suited for embedding high-dimensional data into a low-dimensional space (typically 2D or 3D) while aiming to preserve the local structure and similarities within the data. Developed by Geoffrey Hinton and Laurens van der Maaten in 2008, t-SNE has gained immense popularity due to its ability to produce high-quality visualizations and uncover hidden patterns and clusters in complex datasets.

## Key Concepts

- **Dimensionality Reduction:** This means reducing the number of variables in the data. t-SNE reduces data from high-dimensional space to a 2D or 3D space, making it easier to plot and visually inspect.
- **Stochastic Neighbor Embedding:** This idea models the probability distribution of pairs of high-dimensional objects. Nearby points in high-dimensional space remain close in the low-dimensional space, and distant points stay far apart.
- **t-Distribution:** Unlike linear techniques like PCA (Principal Component Analysis), t-SNE is non-linear. It uses a heavy-tailed t-distribution in the low-dimensional space to prevent points from clumping together.

## How t-SNE Works

- **Pairwise Similarities:** t-SNE starts by calculating how similar each pair of points is in the high-dimensional space. It measures the Euclidean distance between points and converts these distances into probabilities that represent similarities.

  The similarity $p_{ij}$ between two points $x_i$ and $x_j$ is calculated as:
  $$
  p_{ij} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
  $$
  Here, $\sigma_i$ is the variance of the Gaussian distribution centered at $x_i$.
  
- **Joint Probabilities:** These probabilities are made symmetrical to ensure that the similarity between point A and point B is the same as between point B and point A.

  The joint probability $P_{ij}$ is:
  $$
  P_{ij} = \frac{p_{ij} + p_{ji}}{2N}
  $$
  Here, $N$ is the number of data points.

- **Low-Dimensional Mapping:** Points are initially placed randomly in a low-dimensional space. t-SNE then adjusts their positions to minimize the difference between the high-dimensional and low-dimensional similarities.

- **Gradient Descent:** Positions are adjusted using an optimization method called gradient descent. This minimizes the Kullback-Leibler divergence between the two probability distributions (high-dimensional and low-dimensional).

  The Kullback-Leibler divergence $KL(P \parallel Q)$ is:
  $$
  KL(P \parallel Q) = \sum_{i \neq j} P_{ij} \log\left(\frac{P_{ij}}{Q_{ij}}\right)
  $$

  Here, $Q_{ij}$ is the similarity between points $y_i$ and $y_j$ in the low-dimensional space, calculated as:
  $$
  Q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
  $$

  The gradient descent algorithm updates the positions $y_i$ to minimize $KL(P \parallel Q)$, ensuring that the low-dimensional representation maintains the structure of the high-dimensional data as closely as possible.




``` python 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load example data
digits = load_digits()
X = digits.data
y = digits.target

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=10)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE visualization of Digits dataset')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
```

   - Advantages:
       - 1. Provides clear insights into complex data.
       - 2. Maintains local similarities effectively.
       - 3. Robust to Noise
       - 4. Non-Linear Representation and Captures complex relationships accurately.
   - Limitations:
      - 1. Resource-demanding, especially for large datasets.
      - 2. May distort overall data relationships.
      - 3. Complex datasets may pose challenges in exact interpretation.

- **Approximate Nearest Neighbor (ANN) Search:** Approximate Nearest Neighbor (ANN) search is a technique used to find points in a high-dimensional space that are approximately closest to a given query point. This method is particularly crucial when dealing with large datasets where exact nearest neighbor search becomes computationally infeasible. ANN search balances between accuracy and computational efficiency, making it an invaluable tool in various fields such as machine learning, data mining, and information retrieval.
   - **ANN Search in Machine Learning** ANN search is crucial for high-dimensional data tasks, such as:
     - **Feature Matching in Computer Vision:** Identifies similar features across images for tasks like image stitching, object recognition, and 3D reconstruction.
     - **Recommendation Systems:** Recommends items by identifying similar users or items based on behavior or attributes represented as vectors.
     - **Clustering:** Accelerates clustering large datasets by quickly finding approximate clusters, which can then be refined.
   - **ANN Search in Data Mining** In data mining, ANN search enhances:
     - **Efficient Data Retrieval:** Quickly finds relevant data points similar to a query, essential for applications like anomaly detection.
     - **Pattern Recognition:** Identifies patterns or associations within large datasets, aiding in market basket analysis and customer segmentation.
   - **ANN Search in Information Retrieval** Information retrieval systems use ANN search for:
     - **Semantic Search:** Retrieves documents or information semantically similar to a user's query by representing text data as vectors.
     - **Multimedia Retrieval:** Finds similar images, videos, or audio files based on content rather than metadata, using high-dimensional vectors.

- **Trade-Off Between Accuracy and Efficiency in ANN Search** In Approximate Nearest Neighbor (ANN) search, balancing accuracy and efficiency is crucial, especially for large-scale and high-dimensional datasets. While the aim is to quickly find the nearest neighbors with high precision, achieving both accuracy and speed is challenging due to computational constraints.
  - **Accuracy vs. Efficiency**
    - **Accuracy:** Ensures the search results closely match the exact nearest neighbors. High accuracy is vital for tasks requiring precise similarity measures but demands extensive computations, making it resource-intensive and slow.
    - **Efficiency:** Focuses on the speed and resource usage of the search. Efficient algorithms deliver quick results and use minimal memory, but they may sacrifice some accuracy by employing approximations and heuristics.
  - **Importance of Faster Search Methods**
  - 1 **Large-Scale Datasets**
     - **Real-Time Processing:** In applications like online search engines, recommendation systems, and real-time analytics, delivering results almost instantaneously is crucial. Efficient ANN search methods enable these systems to provide timely and relevant results without delays.
     - **Scalability:** As datasets grow, the computational burden increases exponentially. Efficient ANN search algorithms ensure the system can handle this growth without a proportional rise in resource requirements, maintaining performance and responsiveness.
  - 2 **High-Dimensional Data:**
     - **Reduced Computational Complexity:** Techniques that reduce the number of dimensions or approximate distances help manage the computational load, making it feasible to process high-dimensional data effectively. This is crucial in fields like image and video processing, natural language processing, and genomics.
     - **Handling Sparsity:** High-dimensional spaces often lead to sparse data distributions. Efficient ANN search methods are designed to navigate this sparsity, finding relevant neighbors without exhaustive searches.
- **Techniques to Balance Accuracy and Efficiency**
   - Flat Indexing 
   - Inverted Index
   - Locality-Sensitive Hashing (LSH)
   - Product Quanitzation
   - Vector Quantization
   - Tree Based Indexing like K-D Tree, Ball Tree and R Tree
   - Graph based indexing algorithms like HNSW and Vamana
   - Inverted File Indexing (IVF) 
   - LSH Forest
   - Composite Indexing (e.g., IVF + PQ, LSH + KDTree, HNSW + IVF)
- **Flat Indexing:** Flat indexing, also referred to as brute-force or exhaustive indexing, entails storing all dataset vectors within a single index structure, typically an array or list. Each vector is assigned a unique identifier or index within this structure. Upon receiving a query vector, the index is sequentially traversed, and the similarity between the query and each dataset vector is computed. This iterative process continues until all vectors are assessed, ultimately identifying the closest matches to the query.
  - **How it works:**
     - **Index Construction:** Initially, all dataset vectors are stored in memory or on disk to construct the index.
       -  All dataset vectors $X = \{x_1, x_2, \ldots, x_n\}$ are stored in memory or on disk.
     - **Query Processing:** Upon receiving a query vector, the system systematically compares it with every vector in the index, computing the similarity or distance metric (e.g., Euclidean distance, cosine similarity) between the query and each vector.
     - **Ranking:** As comparisons progress, vectors are ranked based on their similarity to the query, thereby pinpointing the closest matches.
     - **Retrieval:** After evaluating all vectors, the system retrieves either the top-k closest matches or all vectors meeting a specified similarity threshold.

   - Advantages:
       - 1. Easy implementation and comprehension.
       - 2. Avoids complex index structures, suitable for smaller datasets.
       - 3. Adaptable to various data types and metrics.
   - Limitations:
      - 1. Inefficient for large datasets, leading to slower query processing.
      - 2. Can be computationally intensive for high-dimensional data.
      - 3. Requires substantial memory resources for storage.

``` python 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
dataset_vectors = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

query_vector = np.array([1, 0, 0])
similarities = cosine_similarity([query_vector], dataset_vectors)[0]

ranked_indices = np.argsort(-similarities)

top_k = 3
top_k_indices = ranked_indices[:top_k]
top_k_matches = dataset_vectors[top_k_indices]

print("Top-k indices:", top_k_indices)
print("Top-k matches:\n", top_k_matches)
print("Similarity scores:", similarities[top_k_indices])
```
```
Top-k indices: [2 1 0]
Top-k matches:
 [[7 8 9]
 [4 5 6]
 [1 2 3]]
Similarity scores: [0.50257071 0.45584231 0.26726124]
```
- **Inverted Index:** An Inverted Index is a data structure used primarily in information retrieval systems, such as search engines, to efficiently map content to its location in a database, document, or set of documents. It enables quick full-text searches by maintaining a mapping from content terms to their occurrences in the dataset.
   - **How It Works**
     - **Tokenization:** The process starts with tokenizing the text data. Tokenization involves breaking down text into individual tokens, typically words or terms.
     - **Normalization:** Tokens are often normalized, which may include converting to lowercase, removing punctuation, and applying stemming or lemmatization to reduce words to their base forms.
     - **Index Construction:** Each unique token is stored in the index, along with a list of documents or positions where it appears. This mapping allows for efficient look-up during search queries.
     - **Posting List:** Each token in the index has an associated posting list, which is a list of all documents and positions where the token appears.

   - Example Consider three documents
     - $Document 1:$ `apple banana fruit`
     - $Document 2:$ `banana apple juice`
     - $Document 3:$ `fruit apple orange`
   - The inverted index for these documents would look like this:
     - $apple: [1, 2, 3]$
     - $banana: [1, 2]$
     - $fruit: [1, 3]$
     - $juice: [2]$
     - $orange: [3]$
   - Here, the numbers represent the document IDs where each term appears.

   - Advantages:
       - 1. Allows quick retrieval of documents containing the queried terms.
       - 2. Optimizes storage space by indexing only terms and their occurrences rather than entire documents.
   - Limitations:
      - 1. Adding or deleting documents requires updating the index, which can be complex and resource-intensive.
      - 2. Large datasets with many unique terms can result in significant storage overhead for the index.

``` python
import re
from collections import defaultdict
# Function to tokenize and normalize text
def tokenize(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    return tokens

# Function to build the inverted index
def build_inverted_index(docs):
    inverted_index = defaultdict(list)
    for doc_id, doc in enumerate(docs):
        tokens = tokenize(doc)
        for token in tokens:
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)
    return inverted_index

documents = [
    "apple banana fruit",
    "banana apple juice",
    "fruit apple orange"
]

inverted_index = build_inverted_index(documents)
for term, postings in sorted(inverted_index.items()):
    print(f"{term}: {postings}")
```
```
apple: [0, 1, 2]
banana: [0, 1]
fruit: [0, 2]
juice: [1]
orange: [2]
```

- **Locality-Sensitive Hashing (LSH)** Locality-sensitive hashing (LSH) is a technique used to efficiently find approximate nearest neighbors in high-dimensional data. This method is particularly useful when dealing with large datasets where the exact nearest neighbor search would be too slow. LSH aims to hash similar items into the same buckets with high probability, which makes searching faster.
  - **Key Concepts**
    - **Locality Preservation:** LSH ensures that items that are close to each other in high-dimensional space are likely to be in the same bucket after hashing.
    - **Hash Function Family:** LSH uses a set of hash functions $\mathcal{H}$ that have a high probability of assigning similar items to the same bucket and a low probability of assigning dissimilar items to the same bucket.
    - **Approximation:** LSH provides approximate results, which means it finds neighbors that are close enough rather than the exact nearest neighbours.

  - **How LSH Works**
    - 1. **Hash Function Selection:** Choose or design hash functions that are locality-sensitive to the chosen similarity metric.
    - 2. **Index Construction:** Apply the hash functions to all items in the dataset, distributing them into buckets.
    - 3. **Query Processing:**
       - Hash the query item using the same hash functions.
       - Retrieve and compare items from the corresponding bucket(s).
       - Use a secondary, more precise similarity measure to rank the retrieved items and find the approximate nearest neighbours.

  - **Mathematics of LSH**
     - **Distance Measure:** $d(\mathbf{x}, \mathbf{y})$ denotes the distance between two points $\mathbf{x}$ and $\mathbf{y}$ in a high-dimensional space.
     - **Hash Function Family:** $\mathcal{H}$ is a set of hash functions. A hash function $h \in \mathcal{H}$ maps a point $\mathbf{x}$ to a bucket.
     - **Probabilities:**
      - $P_1 = \Pr[h(\mathbf{x}) = h(\mathbf{y}) \mid d(\mathbf{x}, \mathbf{y}) \leq r]$ is the probability that $h$ hashes two points $\mathbf{x}$ and $\mathbf{y}$ to the same bucket if $\mathbf{x}$ and $\mathbf{y}$ are within distance $r$.
      - $P_2 = \Pr[h(\mathbf{x}) = h(\mathbf{y}) \mid d(\mathbf{x}, \mathbf{y}) > cr]$ is the probability that $h$ hashes two points $\mathbf{x}$ and $\mathbf{y}$ to the same bucket if $\mathbf{x}$ and $\mathbf{y}$ are further than $cr$ apart, where $c > 1$.
     - **Locality-Sensitive Hash Family:** A hash family $\mathcal{H}$ is called $(r, cr, P_1, P_2)$-sensitive if $P_1 > P_2$, which ensures that similar items have a higher probability of colliding than dissimilar ones.

- **Example: Euclidean Distance**
A common hash function for Euclidean distance is:

$$
h_{\mathbf{a}, b}(\mathbf{x}) = \left\lfloor \frac{\mathbf{a} \cdot \mathbf{x} + b}{w} \right\rfloor
$$

where:
- $\mathbf{a}$ is a random vector with each component drawn from a Gaussian distribution.
- $b$ is a random shift drawn uniformly from the range $[0, w]$.
- $w$ is the width of the hash bin.

This hash function ensures that points close in Euclidean space are more likely to fall into the same hash bin.

   - Advantages:
       - 1. Speeds up approximate nearest neighbor search, especially in high-dimensional spaces.
       - 2. Handles large datasets effectively.
       - 3. Similar items are likely to be hashed into the same bucket.
   - Limitations:
      - 1. Does not guarantee exact nearest neighbors.
      - 2. Requires careful tuning of parameters like hash functions and bin width.
      - 3. Possible false positives where dissimilar items collide.
      - 4. Reduced accuracy in very high-dimensional spaces.

``` python
import numpy as np

# Generate random hyperplanes
def generate_random_hyperplanes(num_planes, dim):
    return np.random.randn(num_planes, dim)

# Compute LSH signatures
def compute_lsh_signatures(data, hyperplanes):
    return np.dot(data, hyperplanes.T) > 0

# Query with LSH
def lsh_query(query, hyperplanes, dataset_signatures):
    query_signature = compute_lsh_signatures(query.reshape(1, -1), hyperplanes)
    matches = np.all(dataset_signatures == query_signature, axis=1)
    return matches

# data
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

query = np.array([1, 0, 0])

# LSH parameters
num_planes = 5
dim = data.shape[1]

# Generate hyperplanes and compute signatures
hyperplanes = generate_random_hyperplanes(num_planes, dim)
dataset_signatures = compute_lsh_signatures(data, hyperplanes)

# Query and find matches
matches = lsh_query(query, hyperplanes, dataset_signatures)
print("Matched indices:", np.where(matches)[0])
```
```
Matched indices: []
```
- **Quantization:** Quantization is a crucial technique in Approximate Nearest Neighbor (ANN) search, particularly when dealing with large and high-dimensional datasets. By approximating data points with a limited set of representative points (centroids), quantization reduces storage requirements and computational complexity, facilitating faster and more efficient similarity searches.
  - **Key Concepts in Quantization** 
    - **Quantization:** The process of mapping high-dimensional vectors to a finite set of representative points, thereby reducing data complexity.
    - **Centroids/Codewords:** Representative points used in the quantization process. Each data point is approximated by the nearest centroid.
       - Consider approximating the value of $π(pi)$, which is approximately $3.14159$. Let's use a simple codebook with two centroids: $C1=3.0$ and $C2=3.2$
       - The nearest centroid to $π(3.14159)$ is $C2=3.2$ since, $∣3.14159−3.2∣=0.05841$ is less than $∣3.14159−3.0∣=0.14159$
         - Therefore, $π$ is approximated by $C2$ 
    - **Codebook:** A collection of centroids that are used to approximate the original data points.
    - **Quantization Error:** The difference between the original data point and its quantized approximation. Lower quantization error implies higher accuracy in search results.
        - The quantization error is the squared difference between $π$ and the centroid $C2$
           - Quatization Error = $(3.14159−3.2)^2$ $=(−0.05841)^2$ $≈0.00341$
- Quantization helps manage large datasets by simplifying data representation, which in turn speeds up the process of finding similar data points through approximate nearest neighbor search techniques.
- **Types of Quantization:**
   - 1. **Scalar Quantization:** Scalar quantization is a technique where each component of a vector is quantized independently, simplifying the data representation process by breaking down the high-dimensional problem into individual dimensions.
      - **Example:** Suppose we have a dataset of 2D points $(x, y)$, and we want to quantize each dimension independently. Let's consider quantizing $x$ and $y$ into three levels: ${1.0, 2.0, 3.0}$ for $x$ and ${4.0, 5.0, 6.0}$ for y.
         - **Quantization Process** Given a point $(2.3, 4.7)$, we quantize $x$ to the nearest level, which is $2.0$, and $y$ to $5.0$. So, the quantized point becomes $(2.0, 5.0)$.
         - **Quantization Error** To compute the error, we take the sum of squared differences between the original and quantized values:
           - Quantization Error = $(2.3 - 2.0)^2 + (4.7 - 5.0)^2 = 0.09 + 0.09 = 0.18$
   - 2. **Vector Quantization:** This technique quantizes the entire vector as a whole rather than its individual components, capturing the correlations between different dimensions of the vector. The data points are mapped to the nearest centroid in a set of predefined centroids (codebook) based on the overall similarity.
      - **Example:**  Consider the same 2D dataset, but this time, we want to quantize the entire vector as a single entity. Let's have centroids ${(1.0, 2.0), (3.0, 4.0)}$.
         - **Quantization Process** For the point $(2.3, 4.7)$, we find the nearest centroid, which is $(3.0, 4.0)$. Thus, the quantized point becomes $(3.0, 4.0)$.
         - **Quantization Error** The error is computed as the squared Euclidean distance between the original and quantized vectors:
           - Quantization Error = $(2.3 - 3.0)^2 + (4.7 - 4.0)^2 = 0.49 + 0.49 = 0.98$

   - 3. **Product Quantization:** Product quantization is an advanced technique designed to handle very large and high-dimensional datasets efficiently by decomposing the original space into lower-dimensional subspaces.
         - **Process**
            - **Decomposition:** Divide the high-dimensional vector into smaller, non-overlapping sub-vectors.
            - **Independent Quantization:** Quantize each sub-vector independently using its own set of centroids.
            - **Complexity Reduction:** Break down the high-dimensional quantization problem into several lower-dimensional problems.
            - **Centroid Assignment:**
               - Assign each sub-vector a centroid from a sub-codebook.
               - Combine these centroids to represent the original vector.
      - **Example:** Suppose we have a $4D$ vector $(𝑥1, 𝑥2, 𝑥3, 𝑥4)$ and want to perform product quantization by splitting it into two 2D sub-vectors: $(𝑥1, 𝑥2)$ and $(𝑥3, 𝑥4)$. Let's use centroids ${(1.0, 2.0), (3.0, 4.0)}$ for each sub-vector.
         - **Quantization Process** For the vector $(1.1, 2.2, 3.1, 3.9)$, the sub-vector $(𝑥1, 𝑥2)$ is closest to $(1.0, 2.0)$, and $(𝑥3, 𝑥4)$ is closest to $(3.0, 4.0)$. So, the quantized vector becomes $(1.0, 2.0, 3.0, 4.0)$.
         - **Quantization Error** The total error is the sum of errors from quantizing each sub-vector:
           - Quantization Error = $(0.1)^2 + (0.2)^2 + (0.1)^2 + (0.1)^2 = 0.01 + 0.04 + 0.01 + 0.01 = 0.07$
``` python
import numpy as np
from sklearn.cluster import KMeans

# Split vectors into sub-vectors
def split_vectors(data, num_subvectors):
    subvector_length = data.shape[1] // num_subvectors
    return np.split(data, num_subvectors, axis=1)

# Perform product quantization
def product_quantization(data, num_subvectors, num_centroids):
    subvectors = split_vectors(data, num_subvectors)
    codebooks = []
    quantized_indices = []

    for subvector in subvectors:
        kmeans = KMeans(n_clusters=num_centroids, n_init=10, random_state=0).fit(subvector)
        codebooks.append(kmeans.cluster_centers_)
        quantized_indices.append(kmeans.predict(subvector))

    return codebooks, np.array(quantized_indices).T

# Reconstruct vectors from quantized indices
def reconstruct_vectors(codebooks, quantized_indices):
    reconstructed_vectors = np.hstack([codebooks[i][quantized_indices[:, i]] for i in range(len(codebooks))])
    return reconstructed_vectors

# Data
data = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18]
])

# Parameters
num_subvectors = 2  
num_centroids = 3 

codebooks, quantized_indices = product_quantization(data, num_subvectors, num_centroids)
reconstructed_vectors = reconstruct_vectors(codebooks, quantized_indices)

print("Original Data:\n", data)
print("\nCodebooks:\n", codebooks)
print("\nQuantized Indices:\n", quantized_indices)
print("\nReconstructed Vectors:\n", reconstructed_vectors)
```
- **Tree-Based Indexing in Approximate Nearest Neighbor Search:** Tree-based indexing techniques are critical for efficiently managing and querying high-dimensional data. These structures organize data points hierarchically, allowing quick search and retrieval operations. The primary types of tree-based indexing methods used in Approximate Nearest Neighbor (ANN) search include K-D Tree, Ball Tree, and R-Tree. Each of these trees has unique characteristics and applications, as detailed below.
  - 1. **K-D Tree (K-Dimensional Tree)** A K-D Tree is a binary tree that organizes points in a k-dimensional space. It is particularly effective for low-dimensional data but can suffer from inefficiencies as the dimensionality increases.
    - **Construction**
      - **Splitting Planes:** At each level of the tree, the dataset is split along one of the k dimensions. The splitting dimension is usually chosen in a round-robin fashion or based on the dimension with the highest variance.
      - **Median Selection:** The splitting point is typically the median of the selected dimension, ensuring that each subtree has roughly half of the points.
    - **Query Processing**
      - **Traversal:** For a given query point, the tree is traversed starting from the root, moving left or right depending on the value of the splitting dimension.
      - **Backtracking:** Once a leaf node is reached, the algorithm may backtrack to explore other branches that could potentially contain closer points.
    - **Balancing** The tree is balanced to minimize the depth, which helps maintain efficient query times. Balancing involves ensuring that the median point is chosen for splitting at each node, which divides the data into roughly equal parts.
   - Advantages:
       - 1. Efficient for low-dimensional spaces (usually less than 20 dimensions).
       - 2. Simple to implement and understand.
       - 3. Provides exact nearest neighbor search.
   - Limitations:
      - 1. Performance degrades significantly with increasing dimensionality due to the `curse of dimensionality`.
      - 2. Requires periodic rebalancing to maintain efficiency.
      - 3. Not suitable for dynamic datasets where frequent insertions and deletions are needed.
 ``` python 
from sklearn.neighbors import KDTree
import numpy as np

#  data
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Build KDTree
tree = KDTree(data, leaf_size=2)

# Query
query = np.array([[1, 2]])
dist, ind = tree.query(query, k=2)
print("Distances:", dist)
print("Indices:", ind)
 ```



## LLMs 
- **Large Language Models:**
- Large language models (LLMs) are super-smart computer programs that understand and generate human-like text. They're like really big brains made of Transformer-based architecture, with hundreds of billions or even more parameters. Examples include GPT Series by OpenAI, Llama Series by Meta, Mistral Series, and Claude by Anthropic. These models are incredibly good at understanding language and doing all sorts of tasks with it.
- Think of LLMs as the giants of AI language processing. They're built on this thing called the Transformer architecture, which has layers of attention mechanisms and neural networks. LLMs differ from smaller models mainly because they're, well, huge! They have way more parameters, bigger datasets to learn from, and need a ton of computational power.
- So, LLMs are like the superheroes of text understanding and generation, making them super valuable for all sorts of applications, from chatbots to content creation tools. They're like the big brains behind the scenes, helping computers understand and interact with human language in really cool ways.

## RAG 
