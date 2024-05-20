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
  # Dictionary of contractions and their expanded forms
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
  print(expanded_text)  # Output: "I am learning NLP. It is fascinating! Do not you think so?"  
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

"""
Output:
Original sentence: I havv a pbroblem wit my speling.
Corrected sentence: I have a problem with my spelling.
"""
```


     
## Information Retrieval 

## Vector Search 

## LLMs 

## RAG 
