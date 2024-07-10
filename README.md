# Awesome NLP and IR

## A Unified Repository

Welcome to your ultimate resource hub for concepts and papers related to:

- **Natural Language Processing (NLP)**
- **Large Language Models (LLMs)**
- **LLM Efficiency**
- **Vector Search and Information Retrieval** 
- **Retrieval Augmented Generation, RAG Optimization and Best Practices**

Find everything you need in one concise, well-organized place.

# Table of Contents

## Natural Language Processing (NLP)

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

## Statistical NLP

- [Naive Bayes](#naive-bayes)
- [N-gram Language Model](#n-gram-language-model)
- [Markov Chain](#markov-chain)
- [Hidden Markov Model (HMM)](#hidden-markov-model-hmm)
- [Conditional Random Fields (CRFs)](#conditional-random-fields-crfs)

## Representation Learning in NLP

- [Encoding / Sparse Vectors](#encoding--sparse-vectors)
  - [One Hot Encoding](#one-hot-encoding)
  - [Integer Encoding](#integer-encoding)
  - [Bag of Words](#bag-of-words)
  - [TF-IDF](#tf-idf)
  - [BM25](#bm25)
- [Embedding / Dense Vectors](#embedding--dense-vectors)
  - [Word2Vec](#word2vec)
  - [GloVe](#glove)
  - [FastText](#fasttext)
  - [ELMo](#elmo)
  - [BERT](#bert)

## Deep NLP

- [Different Activation Functions](#different-activation-functions)
- [Optimization Algorithms](#optimization-algorithms)
  - [Comparison of Gradient Descent](#comparison-of-gradient-descent)
- [Feedforward Neural Networks (FNN)](#feedforward-neural-networks-fnn)
- [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
  - [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
  - [Bidirectional RNNs](#bidirectional-rnns)
- [Transformers](#transformers)
  - [Key Components of Transformer Architecture](#key-components-of-transformer-architecture)
  - [Transformer Architectures: A Detailed Comparison](#transformer-architectures-a-detailed-comparison)

## Large Language Models (LLMs)

- [Architecture](#architecture)
- [LLM Pretraining](#llm-pretraining)
  - [Self-Supervised Learning](#self-supervised-learning)
    - [Masked Language Modeling](#masked-language-modeling)
    - [Masked Multimodal Language Modeling](#masked-multimodal-language-modeling)
    - [Next Sentence Prediction](#next-sentence-prediction)
    - [Causal Language Modeling](#causal-language-modeling)
    - [Denoising Autoencoders](#denoising-autoencoders)
    - [Contrastive Learning](#contrastive-learning)
- [LLM Fine-Tuning](#llm-fine-tuning)
  - [Supervised Fine Tuning](#supervised-fine-tuning)
  - [Full Fine-Tuning](#full-fine-tuning)
  - [PEFT (Parameter-Efficient Fine-Tuning)](#peft-parameter-efficient-fine-tuning)
    - [Additive PEFT](#additive-peft)
      - [Sparse Regularization](#sparse-regularization)
      - [Structured Regularization](#structured-regularization)
    - [Selective PEFT](#selective-peft)
      - [Task-Specific Parameter Identification](#task-specific-parameter-identification)
      - [Parameter Masking](#parameter-masking)
    - [Background of Matrix Decomposition](#background-of-matrix-decomposition)
      - [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)
      - [Low-Rank Approximations](#low-rank-approximations)
    - [Reparameterized PEFT](#reparameterized-peft)
      - [Parameter Transformation](#parameter-transformation)
      - [Learned Parameter Adaptation](#learned-parameter-adaptation)
    - [Hybrid PEFT](#hybrid-peft)







  **LLM Efficiency**
  
  **Vector Search and Information Retrieval**

  **Retrieval Augmented Generation, RAG Optimization and Best Practices**
