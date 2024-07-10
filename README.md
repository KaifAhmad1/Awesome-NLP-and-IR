# Awesome NLP and IR

## A Unified Repository

Welcome to your ultimate resource hub for concepts and papers related to:

- **Natural Language Processing (NLP)**
- **Large Language Models (LLMs)**
- **LLM Efficiency**
- **Vector Search and Information Retrieval** 
- **Retrieval Augmented Generation, RAG Optimization and Best Practices**

Find everything you need in one concise, well-organized place.

## Table of Contents

**Natural Language Processing (NLP)**

- [Preprocessing](#preprocessing)
  - [Case Folding](#case-folding)
  - [Contraction Mapping](#contraction-mapping)
  - [Correcting Spelling Errors](#correcting-spelling-errors)
    - [Dictionary-based approaches](#dictionary-based-approaches)
    - [Edit distance algorithms](#edit-distance-algorithms)
    - [Rule-based methods](#rule-based-methods)
  - [Deduplication / Duplicate Removal](#deduplication--duplicate-removal)
    - [Using Pandas for Deduplication](#using-pandas-for-deduplication)
    - [Using Fuzzy Matching for Deduplication](#using-fuzzy-matching-for-deduplication)
  - [Expanding Abbreviations and Acronyms](#expanding-abbreviations-and-acronyms)
    - [Dictionary-Based Methods](#dictionary-based-methods)
    - [Rule-Based Methods](#rule-based-methods)
    - [Statistical Methods](#statistical-methods)
  - [Stemming](#stemming)
  - [Lemmatization](#lemmatization)
    - [Rule-Based Lemmatization](#rule-based-lemmatization)
    - [Lexical Resource-based Lemmatization](#lexical-resource-based-lemmatization)
  - [Noise Removing](#noise-removing)
    - [Stripping HTML Tags](#stripping-html-tags)
    - [Removing Special Characters](#removing-special-characters)
    - [Removing Stop Words](#removing-stop-words)
    - [Removing Numerical Values](#removing-numerical-values)
    - [Handling Emojis and Emoticons](#handling-emojis-and-emoticons)
    - [Removing Non-Linguistic Symbols](#removing-non-linguistic-symbols)
  - [Tokenization](#tokenization)
    - [Word Tokenization](#word-tokenization)
    - [Subword Tokenization](#subword-tokenization)
      - [Byte Pair Encoding (BPE)](#byte-pair-encoding-bpe)
      - [WordPiece Tokenization](#wordpiece-tokenization)
      - [Unigram Tokenization](#unigram-tokenization)
      - [SentencePiece Tokenization](#sentencepiece-tokenization)
- [Statical NLP](#statical-nlp)
  - [Naive Bayes](#naive-bayes)
  - [N-gram Language Model](#n-gram-language-model)
  - [Markov Chain](#markov-chain)
  - [Hidden Markov Model (HMM)](#hidden-markov-model-hmm)
  - [Conditional Random Fields (CRFs)](#conditional-random-fields-crfs)
- [Representation Learning in NLP](#statical-nlp)
  - [Encoding \ Sparse Vectors](#correcting-spelling-errors)
      - [One Hot Encoding](#dictionary-based-approaches)
      - [Integer Encoding](#dictionary-based-approaches)
      - [Bag of Words](#dictionary-based-approaches)
      - [TF-IDF](#dictionary-based-approaches)
      - [BM25](#dictionary-based-approaches)

  - [Embedding \ Dense Vectors](#correcting-spelling-errors)
      - [Word2Vec](#dictionary-based-approaches)
      - [GloVe](#dictionary-based-approaches)
      - [FastText](#dictionary-based-approaches)
      - [ELMo](#dictionary-based-approaches)
      - [BERT](#dictionary-based-approaches)
    
- [Deep NLP](#statical-nlp)
  - [Different Activation Functions](#correcting-spelling-errors)
  - [Optimization Algorithms](#correcting-spelling-errors)
      - [Comparision of Gradient Descent](#dictionary-based-approaches)
  - [Feedforward Neural Networks (FNN)](#correcting-spelling-errors)
  - [Recurrent Neural Networks (RNN)](#correcting-spelling-errors)
      - [Long Short-Term Memory (LSTM)](#dictionary-based-approaches)
      - [Gated Recurrent Unit(GRU)](#dictionary-based-approaches)
      - [Bidirectional RNNs](#dictionary-based-approaches)
  - [Transformers](#dictionary-based-approaches)
      - [Key Components of Transformer Architecture](#dictionary-based-approaches)
      - [Transformer Architectures: A Detailed Comparison](#dictionary-based-approaches)
         
       
**Large Language Models(LLMs)**
- [Architecture](#preprocessing)
- [LLM Pretraining](#preprocessing)
  - [Self Supervised Learning](#case-folding)
      - [Masked Language Modeling](#dictionary-based-approaches)
      - [Masked Multimodal Language Modeling](#dictionary-based-approaches)
      - [Next Sentence Prediction](#dictionary-based-approaches)
      - [Causal Language Modeling](#dictionary-based-approaches)
      - [Denoising Autoencoders](#dictionary-based-approaches)
      - [Contrastive Learning](#dictionary-based-approaches)
- [LLM FineTuning](#preprocessing)
  - [Supervised Fine Tuning](#preprocessing)
  - [Full Fine-Tuning](#dictionary-based-approaches)
  - [PEFT (Parameter-Efficient Fine-Tuning)](#dictionary-based-approaches)
      - [Additive PEFT](#dictionary-based-approaches)
      - [Selective PEFT](#dictionary-based-approaches)
      - [Background of Matrix Decomposition](#dictionary-based-approaches)
      - [Reparameterized PEFT](#dictionary-based-approaches)
      - [Hybrid PEFT](#dictionary-based-approaches)







  **LLM Efficiency**
  
  **Vector Search and Information Retrieval**

  **Retrieval Augmented Generation, RAG Optimization and Best Practices**
