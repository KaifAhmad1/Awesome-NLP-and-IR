# 🚀 Awesome NLP and IR

## A Unified Repository

Welcome to your ultimate resource hub for concepts and papers related to:

- **Natural Language Processing (NLP)**
- **Large Language Models (LLMs)**
- **LLM Efficiency**
- **Vector Search and Information Retrieval**
- **Retrieval Augmented Generation, RAG Optimization and Best Practices**

This repository is a comprehensive, well-structured, and user-friendly guide designed to help you navigate the latest advancements, research papers, techniques, and best practices in these fields. Whether you're a researcher, developer, or enthusiast, you'll find a wealth of information and resources that cover everything from fundamental NLP concepts to advanced RAG techniques.

Explore and leverage this repository to deepen your understanding, enhance your projects, and stay updated with the cutting-edge developments in NLP and IR.


## Table of Contents

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

### Statistical NLP

- [Naive Bayes](#naive-bayes)
- [N-gram Language Model](#n-gram-language-model)
- [Markov Chain](#markov-chain)
- [Hidden Markov Model (HMM)](#hidden-markov-model-hmm)
- [Conditional Random Fields (CRFs)](#conditional-random-fields-crfs)

### Representation Learning in NLP

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
- [Hybrid](#embedding--dense-vectors)
  - [SPLADE](#word2vec)
    
### Deep NLP

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

### Large Language Models (LLMs)

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
      - [Adapters](#adapters)
      - [Soft Prompt-based Fine-tuning](#soft-prompt-based-fine-tuning)
      - [IA^3](#ia3)
    - [Selective PEFT](#selective-peft)
      - [Unstructured Masking](#unstructured-masking)
      - [Structured Masking](#structured-masking)
    - [Background of Matrix Decomposition](#background-of-matrix-decomposition)
    - [Reparameterized PEFT](#reparameterized-peft)
      - [LoRA: Low-Rank Adaptation](#lora-low-rank-adaptation)
      - [DyLoRA: Dynamic LoRA](#dylora-dynamic-lora)
      - [AdaLoRA: Adaptive LoRA](#adalora-adaptive-lora)
      - [DoRA: Weight-Decomposed Low-Rank Adaptation](#dora-weight-decomposed-low-rank-adaptation)
    - [Hybrid PEFT](#hybrid-peft)
      - [UniPELT](#unipelt)
      - [S4](#s4)
      - [MAM Adapter](#mam-adapter)
      - [LLM-Adapters](#llm-adapters)
      - [Neural Architecture Search (NAS)](#neural-architecture-search-nas)
  - [MEFT (Memory-Efficient Fine Tuning)](#meft-memory-efficient-fine-tuning)
      - [LoRA-FA (LoRA with Frozen Activations)](#lora-fa-lora-with-frozen-activations)
      - [HyperTuning](#hypertuning)
      - [Memory-Efficient Zeroth-Order Optimizer (MeZO)](#memory-efficient-zeroth-order-optimizer-mezo)
      - [QLoRA: Quantized Low-Rank Adaptation](#qlora-quantized-low-rank-adaptation)
      - [Expert-Specialized Fine-Tuning](#expert-specialized-fine-tuning)
      - [Sparse Matrix Tuning](#sparse-matrix-tuning)
      - [Representation Finetuning (ReFT)](#representation-finetuning-reft)
   - [Alignment-Based Fine-Tuning](#alignment-based-fine-tuning)
      - [RLHF](#rlhf)
      - [RLAIF](#rlaif)
      - [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
      - [Identity Preference Optimization (IPO)](#identity-preference-optimization-ipo)
      - [Kahneman-Tversky Optimization (KTO)](#kahneman-tversky-optimization-kto)
      - [Odds Ratio Preference Optimization (ORPO)](#odds-ratio-preference-optimization-orpo)
      - [Alignment Techniques Comparison](#alignment-techniques-comparison)

### LLM Efficiency

- [LLM Efficiency, Need, and Benefits](#llm-efficiency-need-and-benefits)
- [Data Level Optimization](#data-level-optimization)
  - [Input Compression](#input-compression)
      - [Prompt Pruning](#prompt-pruning)
      - [Prompt Summarization](#prompt-summarization)
  - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Output Organization](#output-organization)
      - [Skeleton-of-Thought (SoT)](#skeleton-of-thought-sot)
      - [SGD (Sub-Problem Directed Graph)](#sgd-sub-problem-directed-graph)
- [Model Level Optimization](#model-level-optimization)
  - [Efficient Structure Design](#efficient-structure-design)
      - [Sparse Mixture of Experts](#sparse-mixture-of-experts)
      - [Multiscale Transformer](#multiscale-transformer)
      - [Mixture of Denoisers](#mixture-of-denoisers)
      - [Sparseformer](#sparseformer)
      - [Sparse Switch Transformer](#sparse-switch-transformer)
  - [Efficient Attention Mechanisms](#efficient-attention-mechanisms)
      - [Perceiver](#perceiver)
      - [Perceiver AR](#perceiver-ar)
      - [Perceiver IO](#perceiver-io)
      - [Reformer](#reformer)
      - [Linformer](#linformer)
      - [Longformer](#longformer)
      - [Performer](#performer)
      - [Big Bird](#big-bird)
      - [Multi Query Attention (MQA)](#multi-query-attention-mqa)
      - [Group Query Attention (GQA)](#group-query-attention-gqa)
      - [Sliding Window Attention](#sliding-window-attention)
      - [Low-Complexity Attention Models](#low-complexity-attention-models)
      - [Low-Rank Attention](#low-rank-attention)
      - [Flash Attention](#flash-attention)
  - [Transformer Alternatives](#transformer-alternatives)
      - [State Space Models (SSM) and Mamba](#state-space-models-ssm-and-mamba)
      - [RWKV: Reinventing RNNs for the Transformer Era](#rwkv-reinventing-rnns-for-the-transformer-era)
      - [Extended Long Short-Term Memory (xLSTM)](#extended-long-short-term-memory-xlstm)
      - [Parameterization Improvements](#parameterization-improvements)
  - [Model Compression Techniques](#model-compression-techniques)
      - [Quantization](#quantization)
      - [Sparsification](#sparsification)
- [System Level Optimization](#system-level-optimization)
  - [Inference Engine Optimization](#inference-engine-optimization)
      - [Graph and Operator Optimization](#graph-and-operator-optimization)
      - [Different Decoding Strategies like Greedy, Speculative, and Lookahead](#different-decoding-strategies-like-greedy-speculative-and-lookahead)
      - [Graph-Level Optimization](#graph-level-optimization)
  - [Challenges and Solutions in System-Level Optimization](#challenges-and-solutions-in-system-level-optimization)
      - [KV Cache Optimization](#kv-cache-optimization)
      - [Continuous Batching](#continuous-batching)
      - [Scheduling Strategies](#scheduling-strategies)
      - [Distributed Systems Optimization](#distributed-systems-optimization)
      - [Hardware Accelerator Design](#hardware-accelerator-design)

### Vector Search and Information Retrieval

- [Vector Search](#vector-search)
  - [Vector Representation in ML](#vector-representation-in-ml)
  - [Distance Metrics](#distance-metrics)
      - [Euclidean Distance](#euclidean-distance)
      - [Manhattan Distance](#manhattan-distance)
      - [Cosine Similarity](#cosine-similarity)
      - [Jaccard Similarity](#jaccard-similarity)
      - [Hamming Distance](#hamming-distance)
      - [Earth Mover's Distance (EMD)](#earth-movers-distance-emd)
  - [Vector Search Techniques and Their Applications](#vector-search-techniques-and-their-applications)
  - [Nearest Neighbor Search](#nearest-neighbor-search)
  - [Problems with High Dimensional Data](#problems-with-high-dimensional-data)
  - [Linear Search](#linear-search)
  - [Dimensionality Reduction](#dimensionality-reduction)
      - [Principal Component Analysis](#principal-component-analysis)
      - [t-Distributed Stochastic Neighbor Embedding (t-SNE)](#t-distributed-stochastic-neighbor-embedding-t-sne)
  - [Approximate Nearest Neighbor (ANN) Search](#approximate-nearest-neighbor-ann-search)
      - [Trade-Off Between Accuracy and Efficiency](#trade-off-between-accuracy-and-efficiency)
      - [Flat Indexing](#flat-indexing)
      - [Inverted Index](#inverted-index)
      - [Locality-Sensitive Hashing (LSH)](#locality-sensitive-hashing-lsh)
      - [Quantization and their types](#quantization-and-their-types)
      - [Tree-Based Indexing in ANN](#tree-based-indexing-in-ann)
      - [Random Projection in ANN](#random-projection-in-ann)
      - [Graph-based Indexing for ANN Search](#graph-based-indexing-for-ann-search)
      - [LSH Forest](#lsh-forest)
      - [Composite Indexing in ANN](#composite-indexing-in-ann)
  
### Retrieval Augmented Generation, RAG Optimization, and Best Practices

- [RAG](#rag)
  - [Benefits of RAG](#benefits-of-rag)
  - [Limitations and Challenges Addressed by RAG](#limitations-and-challenges-addressed-by-rag)
  - [Types of RAG](#types-of-rag)
    - [Simple RAG](#simple-rag)
    - [Simple RAG with Memory](#simple-rag-with-memory)
    - [Branched RAG](#branched-rag)
    - [Adaptive RAG](#adaptive-rag)
    - [Corrective RAG](#corrective-rag)
    - [Self RAG](#self-rag)
    - [Agentic RAG](#agentic-rag)

- [RAG Optimization and Best Practices](#rag-optimization-and-best-practices)
  - [Challenges](#challenges)
  - [RAG Workflow Components and Optimization](#rag-workflow-components-and-optimization)
    - [Query Classification](#query-classification)
    - [Document Processing and Indexing](#document-processing-and-indexing)
      - [Chunking](#chunking)
      - [Metadata Addition](#metadata-addition)
      - [Embedding Models](#embedding-models)
      - [Embedding Quantization](#embedding-quantization)
      - [Vector Databases](#vector-databases)
    - [Retrieval Optimization](#retrieval-optimization)
      - [Source Selection and Granularity](#source-selection-and-granularity)
      - [Retrieval Methods](#retrieval-methods)
    - [Reranking and Contextual Curation](#reranking-and-contextual-curation)
      - [Reranking Methods](#reranking-methods)
      - [Repacking and Summarization](#repacking-and-summarization)
        - [Repacking](#repacking)
        - [Summarization](#summarization)
    - [Generation Optimization](#generation-optimization)
      - [Language Model Fine-Tuning](#language-model-fine-tuning)
      - [Co-Training Strategies](#co-training-strategies)
    - [Advanced Augmentation Techniques](#advanced-augmentation-techniques)
      - [Iterative Refinement](#iterative-refinement)
      - [Recursive Retrieval](#recursive-retrieval)
      - [Hybrid Approaches](#hybrid-approaches)
    - [Evaluation and Optimization Metrics](#evaluation-and-optimization-metrics)
      - [Performance Metrics](#performance-metrics)
      - [Benchmark Datasets](#benchmark-datasets)
    - [Tools and Platforms for Optimization](#tools-and-platforms-for-optimization)
    - [Recommendations for Implementing RAG Systems](#recommendations-for-implementing-rag-systems)
