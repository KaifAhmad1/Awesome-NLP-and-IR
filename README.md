# 🚀 Awesome NLP and IR

Welcome to your ultimate resource hub for concepts and papers related to:

### 📘 Topics Covered

- **Natural Language Processing (NLP)**
- **Large Language Models (LLMs)**
- **LLM Efficiency and Inference Optimization**
- **Vector Search and Information Retrieval**
- **Retrieval Augmented Generation, RAG Optimization and Best Practices**

This repository offers a comprehensive and structured guide to the latest advancements, research, techniques, and best practices in NLP and Retrieval-Augmented Generation (RAG). It serves as a valuable resource for researchers, developers, and enthusiasts, covering everything from foundational NLP concepts to advanced RAG techniques. Leverage this repository to deepen your expertise, enhance your projects, and stay abreast of cutting-edge developments in NLP and IR.

## Repository Structure

```
Awesome-NLP-and-IR
│
├── README.md
├── LLM Efficiency and Inference Optimization
│   └── LLM Efficiency and Inference Optimization.md
├── LLMs Pretraining and FineTuning
│   └── LLMs, Pretraining and FineTuning.md
├── NLP
│   └── NLP.md
├── RAG and RAG Optimization
│   └── RAG, Optimization and Best Practices.md
├── Vector Search and IR
│   └── Vector Search and IR.md
└── images
```


## Table of Contents

- [Preprocessing](NLP/NLP.md#preprocessing)
  - [Case Folding](NLP/NLP.md#case-folding)
  - [Contraction Mapping](NLP/NLP.md#contraction-mapping)
  - [Correcting Spelling Errors](NLP/NLP.md#correcting-spelling-errors)
    - [Dictionary-based approaches](NLP/NLP.md#dictionary-based-approaches)
    - [Edit distance algorithms](NLP/NLP.md#edit-distance-algorithms)
    - [Rule-based methods](NLP/NLP.md#rule-based-methods)
  - [Deduplication / Duplicate Removal](NLP/NLP.md#deduplication--duplicate-removal)
    - [Using Pandas for Deduplication](NLP/NLP.md#using-pandas-for-deduplication)
    - [Using Fuzzy Matching for Deduplication](NLP/NLP.md#using-fuzzy-matching-for-deduplication)
  - [Expanding Abbreviations and Acronyms](NLP/NLP.md#expanding-abbreviations-and-acronyms)
    - [Dictionary-Based Methods](NLP/NLP.md#dictionary-based-methods)
    - [Rule-Based Methods](NLP/NLP.md#rule-based-methods)
    - [Statistical Methods](NLP/NLP.md#statistical-methods)
  - [Stemming](NLP/NLP.md#stemming)
  - [Lemmatization](NLP/NLP.md#lemmatization)
    - [Rule-Based Lemmatization](NLP/NLP.md#rule-based-lemmatization)
    - [Lexical Resource-based Lemmatization](NLP/NLP.md#lexical-resource-based-lemmatization)
  - [Noise Removing](NLP/NLP.md#noise-removing)
    - [Stripping HTML Tags](NLP/NLP.md#stripping-html-tags)
    - [Removing Special Characters](NLP/NLP.md#removing-special-characters)
    - [Removing Stop Words](NLP/NLP.md#removing-stop-words)
    - [Removing Numerical Values](NLP/NLP.md#removing-numerical-values)
    - [Handling Emojis and Emoticons](NLP/NLP.md#handling-emojis-and-emoticons)
    - [Removing Non-Linguistic Symbols](NLP/NLP.md#removing-non-linguistic-symbols)
  - [Tokenization](NLP/NLP.md#tokenization)
    - [Word Tokenization](NLP/NLP.md#word-tokenization)
    - [Subword Tokenization](NLP/NLP.md#subword-tokenization)
      - [Byte Pair Encoding (BPE)](NLP/NLP.md#byte-pair-encoding-bpe)
      - [WordPiece Tokenization](NLP/NLP.md#wordpiece-tokenization)
      - [Unigram Tokenization](NLP/NLP.md#unigram-tokenization)
      - [SentencePiece Tokenization](NLP/NLP.md#sentencepiece-tokenization)

### Statistical NLP

- [Naive Bayes](NLP/NLP.md#naive-bayes)
- [N-gram Language Model](NLP/NLP.md#n-gram-language-model)
- [Markov Chain](NLP/NLP.md#markov-chain)
- [Hidden Markov Model (HMM)](NLP/NLP.md#hidden-markov-model-hmm)
- [Conditional Random Fields (CRFs)](NLP/NLP.md#conditional-random-fields-crfs)

### Representation Learning in NLP

- [Encoding / Sparse Vectors](NLP/NLP.md#encoding--sparse-vectors)
  - [One Hot Encoding](NLP/NLP.md#one-hot-encoding)
  - [Integer Encoding](NLP/NLP.md#integer-encoding)
  - [Bag of Words](NLP/NLP.md#bag-of-words)
  - [TF-IDF](NLP/NLP.md#tf-idf)
  - [BM25](NLP/NLP.md#bm25)
- [Embedding / Dense Vectors](NLP/NLP.md#embedding--dense-vectors)
  - [Word2Vec](NLP/NLP.md#word2vec)
  - [GloVe](NLP/NLP.md#glove)
  - [FastText](NLP/NLP.md#fasttext)
  - [ELMo](NLP/NLP.md#elmo)
  - [BERT](NLP/NLP.md#bert)
- [Hybrid](NLP/NLP.md#embedding--dense-vectors)
  - [SPLADE](NLP/NLP.md#word2vec)
- [Encoding vs Embedding Comparision](NLP/NLP.md#embedding--dense-vectors)  
### Deep NLP

- [Different Activation Functions](NLP/NLP.md#different-activation-functions)
- [Optimization Algorithms](NLP/NLP.md#optimization-algorithms)
  - [Comparison of Gradient Descent](NLP/NLP.md#comparison-of-gradient-descent)
- [Feedforward Neural Networks (FNN)](NLP/NLP.md#feedforward-neural-networks-fnn)
- [Recurrent Neural Networks (RNN)](NLP/NLP.md#recurrent-neural-networks-rnn)
  - [Long Short-Term Memory (LSTM)](NLP/NLP.md#long-short-term-memory-lstm)
  - [Gated Recurrent Unit (GRU)](NLP/NLP.md#gated-recurrent-unit-gru)
  - [Bidirectional RNNs](NLP/NLP.md#bidirectional-rnns)
- [Transformers](NLP/NLP.md#transformers)
  - [Key Components of Transformer Architecture](NLP/NLP.md#key-components-of-transformer-architecture)
  - [Transformer Architectures: A Detailed Comparison](NLP/NLP.md#transformer-architectures-a-detailed-comparison)

### Large Language Models (LLMs)

- [Architecture](LLM-Pretraining-and-FineTuning/LLMs.md#architecture)
- [LLM Pretraining](LLM-Pretraining-and-FineTuning/LLMs.md#llm-pretraining)
  - [Self-Supervised Learning](LLM-Pretraining-and-FineTuning/LLMs.md#self-supervised-learning)
    - [Masked Language Modeling](LLM-Pretraining-and-FineTuning/LLMs.md#masked-language-modeling)
    - [Masked Multimodal Language Modeling](LLM-Pretraining-and-FineTuning/LLMs.md#masked-multimodal-language-modeling)
    - [Next Sentence Prediction](LLM-Pretraining-and-FineTuning/LLMs.md#next-sentence-prediction)
    - [Causal Language Modeling](LLM-Pretraining-and-FineTuning/LLMs.md#causal-language-modeling)
    - [Denoising Autoencoders](LLM-Pretraining-and-FineTuning/LLMs.md#denoising-autoencoders)
    - [Contrastive Learning](LLM-Pretraining-and-FineTuning/LLMs.md#contrastive-learning)
- [LLM Fine-Tuning](LLM-Pretraining-and-FineTuning/LLMs.md#llm-fine-tuning)
  - [Supervised Fine Tuning](LLM-Pretraining-and-FineTuning/LLMs.md#supervised-fine-tuning)
  - [Full Fine-Tuning](LLM-Pretraining-and-FineTuning/LLMs.md#full-fine-tuning)
  - [PEFT (Parameter-Efficient Fine-Tuning)](LLM-Pretraining-and-FineTuning/LLMs.md#peft-parameter-efficient-fine-tuning)
    - [Additive PEFT](LLM-Pretraining-and-FineTuning/LLMs.md#additive-peft)
      - [Adapters](LLM-Pretraining-and-FineTuning/LLMs.md#adapters)
      - [Soft Prompt-based Fine-tuning](LLM-Pretraining-and-FineTuning/LLMs.md#soft-prompt-based-fine-tuning)
      - [IA^3](LLM-Pretraining-and-FineTuning/LLMs.md#ia3)
    - [Selective PEFT](LLM-Pretraining-and-FineTuning/LLMs.md#selective-peft)
      - [Unstructured Masking](LLM-Pretraining-and-FineTuning/LLMs.md#unstructured-masking)
      - [Structured Masking](LLM-Pretraining-and-FineTuning/LLMs.md#structured-masking)
    - [Background of Matrix Decomposition](LLM-Pretraining-and-FineTuning/LLMs.md#background-of-matrix-decomposition)
    - [Reparameterized PEFT](LLM-Pretraining-and-FineTuning/LLMs.md#reparameterized-peft)
      - [LoRA: Low-Rank Adaptation](LLM-Pretraining-and-FineTuning/LLMs.md#lora-low-rank-adaptation)
      - [DyLoRA: Dynamic LoRA](LLM-Pretraining-and-FineTuning/LLMs.md#dylora-dynamic-lora)
      - [AdaLoRA: Adaptive LoRA](LLM-Pretraining-and-FineTuning/LLMs.md#adalora-adaptive-lora)
      - [DoRA: Weight-Decomposed Low-Rank Adaptation](LLM-Pretraining-and-FineTuning/LLMs.md#dora-weight-decomposed-low-rank-adaptation)
    - [Hybrid PEFT](LLM-Pretraining-and-FineTuning/LLMs.md#hybrid-peft)
      - [UniPELT](LLM-Pretraining-and-FineTuning/LLMs.md#unipelt)
      - [S4](LLM-Pretraining-and-FineTuning/LLMs.md#s4)
      - [MAM Adapter](LLM-Pretraining-and-FineTuning/LLMs.md#mam-adapter)
      - [LLM-Adapters](LLM-Pretraining-and-FineTuning/LLMs.md#llm-adapters)
      - [Neural Architecture Search (NAS)](LLM-Pretraining-and-FineTuning/LLMs.md#neural-architecture-search-nas)
  - [MEFT (Memory-Efficient Fine Tuning)](LLM-Pretraining-and-FineTuning/LLMs.md#meft-memory-efficient-fine-tuning)
      - [LoRA-FA (LoRA with Frozen Activations)](LLM-Pretraining-and-FineTuning/LLMs.md#lora-fa-lora-with-frozen-activations)
      - [HyperTuning](LLM-Pretraining-and-FineTuning/LLMs.md#hypertuning)
      - [Memory-Efficient Zeroth-Order Optimizer (MeZO)](LLM-Pretraining-and-FineTuning/LLMs.md#memory-efficient-zeroth-order-optimizer-mezo)
      - [QLoRA: Quantized Low-Rank Adaptation](LLM-Pretraining-and-FineTuning/LLMs.md#qlora-quantized-low-rank-adaptation)
      - [Expert-Specialized Fine-Tuning](LLM-Pretraining-and-FineTuning/LLMs.md#expert-specialized-fine-tuning)
      - [Sparse Matrix Tuning](LLM-Pretraining-and-FineTuning/LLMs.md#sparse-matrix-tuning)
      - [Representation Finetuning (ReFT)](LLM-Pretraining-and-FineTuning/LLMs.md#representation-finetuning-reft)
   - [Alignment-Based Fine-Tuning](LLM-Pretraining-and-FineTuning/LLMs.md#alignment-based-fine-tuning)
      - [RLHF](LLM-Pretraining-and-FineTuning/LLMs.md#rlhf)
      - [RLAIF](LLM-Pretraining-and-FineTuning/LLMs.md#rlaif)
      - [Direct Preference Optimization (DPO)](LLM-Pretraining-and-FineTuning/LLMs.md#direct-preference-optimization-dpo)
      - [Identity Preference Optimization (IPO)](LLM-Pretraining-and-FineTuning/LLMs.md#identity-preference-optimization-ipo)
      - [Kahneman-Tversky Optimization (KTO)](LLM-Pretraining-and-FineTuning/LLMs.md#kahneman-tversky-optimization-kto)
      - [Odds Ratio Preference Optimization (ORPO)](LLM-Pretraining-and-FineTuning/LLMs.md#odds-ratio-preference-optimization-orpo)
      - [Alignment Techniques Comparison](LLM-Pretraining-and-FineTuning/LLMs.md#alignment-techniques-comparison)

### LLM Efficiency

- [LLM Efficiency, Need, and Benefits](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#llm-efficiency-need-and-benefits)
- [Data Level Optimization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#data-level-optimization)
  - [Input Compression](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#input-compression)
      - [Prompt Pruning](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#prompt-pruning)
      - [Prompt Summarization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#prompt-summarization)
  - [Retrieval-Augmented Generation (RAG)](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#retrieval-augmented-generation-rag)
  - [Output Organization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#output-organization)
      - [Skeleton-of-Thought (SoT)](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#skeleton-of-thought-sot)
      - [SGD (Sub-Problem Directed Graph)](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#sgd-sub-problem-directed-graph)
- [Model Level Optimization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#model-level-optimization)
  - [Efficient Structure Design](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#efficient-structure-design)
      - [Mixture-of-Experts (MoE)](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#mixture-of-experts)
      - [Switch Transformers](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#switch-transformers)
  - [Efficient Attention Mechanisms](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#efficient-attention-mechanisms)
      - [Multi Query Attention (MQA)](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#multi-query-attention-mqa)
      - [Group Query Attention (GQA)](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#group-query-attention-gqa)
      - [Sliding Window Attention](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#sliding-window-attention)
      - [Low-Complexity Attention Models](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#low-complexity-attention-models)
      - [Low-Rank Attention](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#low-rank-attention)
      - [Flash Attention](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#flash-attention)
  - [Transformer Alternatives](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#transformer-alternatives)
      - [State Space Models (SSM) and Mamba](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#state-space-models-ssm-and-mamba)
      - [RWKV: Reinventing RNNs for the Transformer Era](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#rwkv-reinventing-rnns-for-the-transformer-era)
      - [Extended Long Short-Term Memory (xLSTM)](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#extended-long-short-term-memory-xlstm)
      - [Funnel-Transformer](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#funnel-transformer)
      - [Parameterization Improvements](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#parameterization-improvements)
  - [Model Compression Techniques](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#model-compression-techniques)
      - [Quantization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#quantization)
      - [Sparsification](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#sparsification)
      - [Pruning](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#pruning)
- [System Level Optimization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#system-level-optimization)
  - [Inference Engine Optimization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#inference-engine-optimization)
      - [Graph and Operator Optimization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#graph-and-operator-optimization)
      - [Different Decoding Strategies like Greedy, Speculative, and Lookahead](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#different-decoding-strategies-like-greedy-speculative-and-lookahead)
      - [Graph-Level Optimization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#graph-level-optimization)
  - [Challenges and Solutions in System-Level Optimization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#challenges-and-solutions-in-system-level-optimization)
      - [KV Cache Optimization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#kv-cache-optimization)
      - [Continuous Batching](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#continuous-batching)
      - [Scheduling Strategies](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#scheduling-strategies)
      - [Distributed Systems Optimization](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#distributed-systems-optimization)
      - [Hardware Accelerator Design](LLM-Efficiency-and-Inference-Optimization/LLM-Efficiency.md#hardware-accelerator-design)

### Vector Search and Information Retrieval

- [Vector Search](VectorSearch-and-IR/VectorSearch-and-IR.md#vector-search)
  - [Vector Representation in ML](VectorSearch-and-IR/VectorSearch-and-IR.md#vector-representation-in-ml)
  - [Distance Metrics](VectorSearch-and-IR/VectorSearch-and-IR.md#distance-metrics)
      - [Euclidean Distance](VectorSearch-and-IR/VectorSearch-and-IR.md#euclidean-distance)
      - [Manhattan Distance](VectorSearch-and-IR/VectorSearch-and-IR.md#manhattan-distance)
      - [Cosine Similarity](VectorSearch-and-IR/VectorSearch-and-IR.md#cosine-similarity)
      - [Jaccard Similarity](VectorSearch-and-IR/VectorSearch-and-IR.md#jaccard-similarity)
      - [Hamming Distance](VectorSearch-and-IR/VectorSearch-and-IR.md#hamming-distance)
      - [Earth Mover's Distance (EMD)](VectorSearch-and-IR/VectorSearch-and-IR.md#earth-movers-distance-emd)
  - [Vector Search Techniques and Their Applications](VectorSearch-and-IR/VectorSearch-and-IR.md#vector-search-techniques-and-their-applications)
  - [Nearest Neighbor Search](VectorSearch-and-IR/VectorSearch-and-IR.md#nearest-neighbor-search)
  - [Problems with High Dimensional Data](VectorSearch-and-IR/VectorSearch-and-IR.md#problems-with-high-dimensional-data)
  - [Linear Search](VectorSearch-and-IR/VectorSearch-and-IR.md#linear-search)
  - [Dimensionality Reduction](VectorSearch-and-IR/VectorSearch-and-IR.md#dimensionality-reduction)
      - [Principal Component Analysis](VectorSearch-and-IR/VectorSearch-and-IR.md#principal-component-analysis)
      - [t-Distributed Stochastic Neighbor Embedding (t-SNE)](VectorSearch-and-IR/VectorSearch-and-IR.md#t-distributed-stochastic-neighbor-embedding-t-sne)
  - [Approximate Nearest Neighbor (ANN) Search](VectorSearch-and-IR/VectorSearch-and-IR.md#approximate-nearest-neighbor-ann-search)
      - [Trade-Off Between Accuracy and Efficiency](VectorSearch-and-IR/VectorSearch-and-IR.md#trade-off-between-accuracy-and-efficiency)
      - [Flat Indexing](VectorSearch-and-IR/VectorSearch-and-IR.md#flat-indexing)
      - [Inverted Index](VectorSearch-and-IR/VectorSearch-and-IR.md#inverted-index)
      - [Locality-Sensitive Hashing (LSH)](VectorSearch-and-IR/VectorSearch-and-IR.md#locality-sensitive-hashing-lsh)
      - [Quantization and their types](VectorSearch-and-IR/VectorSearch-and-IR.md#quantization-and-their-types)
      - [Tree-Based Indexing in ANN](VectorSearch-and-IR/VectorSearch-and-IR.md#tree-based-indexing-in-ann)
      - [Random Projection in ANN](VectorSearch-and-IR/VectorSearch-and-IR.md#random-projection-in-ann)
      - [Graph-based Indexing for ANN Search](VectorSearch-and-IR/VectorSearch-and-IR.md#graph-based-indexing-for-ann-search)
      - [LSH Forest](VectorSearch-and-IR/VectorSearch-and-IR.md#lsh-forest)
      - [Composite Indexing in ANN](VectorSearch-and-IR/VectorSearch-and-IR.md#composite-indexing-in-ann)
  - [Comparision between Different Indexing Techniques](VectorSearch-and-IR/VectorSearch-and-IR.md#comparision-between-different-indexing-techniques)

### Retrieval Augmented Generation, RAG Optimization, and Best Practices

- [RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#rag)
  - [Benefits of RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#benefits-of-rag)
  - [Limitations and Challenges Addressed by RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#limitations-and-challenges-addressed-by-rag)
  - [Types of RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#types-of-rag)
    - [Simple RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#simple-rag)
    - [Simple RAG with Memory](RAG-and-RAG-Optimization/RAG-and-Optimization.md#simple-rag-with-memory)
    - [Branched RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#branched-rag)
    - [Adaptive RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#adaptive-rag)
    - [Corrective RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#corrective-rag)
    - [Self RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#self-rag)
    - [Agentic RAG](RAG-and-RAG-Optimization/RAG-and-Optimization.md#agentic-rag)

- [RAG Optimization and Best Practices](RAG-and-RAG-Optimization/RAG-and-Optimization.md#rag-optimization-and-best-practices)
  - [Challenges](RAG-and-RAG-Optimization/RAG-and-Optimization.md#challenges)
  - [RAG Workflow Components and Optimization](RAG-and-RAG-Optimization/RAG-and-Optimization.md#rag-workflow-components-and-optimization)
    - [Query Classification](RAG-and-RAG-Optimization/RAG-and-Optimization.md#query-classification)
    - [Document Processing and Indexing](RAG-and-RAG-Optimization/RAG-and-Optimization.md#document-processing-and-indexing)
      - [Chunking](RAG-and-RAG-Optimization/RAG-and-Optimization.md#chunking)
      - [Metadata Addition](RAG-and-RAG-Optimization/RAG-and-Optimization.md#metadata-addition)
      - [Embedding Models](RAG-and-RAG-Optimization/RAG-and-Optimization.md#embedding-models)
      - [Embedding Quantization](RAG-and-RAG-Optimization/RAG-and-Optimization.md#embedding-quantization)
      - [Vector Databases](RAG-and-RAG-Optimization/RAG-and-Optimization.md#vector-databases)
    - [Retrieval Optimization](RAG-and-RAG-Optimization/RAG-and-Optimization.md#retrieval-optimization)
      - [Source Selection and Granularity](RAG-and-RAG-Optimization/RAG-and-Optimization.md#source-selection-and-granularity)
      - [Retrieval Methods](RAG-and-RAG-Optimization/RAG-and-Optimization.md#retrieval-methods)
    - [Reranking and Contextual Curation](RAG-and-RAG-Optimization/RAG-and-Optimization.md#reranking-and-contextual-curation)
      - [Reranking Methods](RAG-and-RAG-Optimization/RAG-and-Optimization.md#reranking-methods)
      - [Repacking and Summarization](RAG-and-RAG-Optimization/RAG-and-Optimization.md#repacking-and-summarization)
        - [Repacking](RAG-and-RAG-Optimization/RAG-and-Optimization.md#repacking)
        - [Summarization](RAG-and-RAG-Optimization/RAG-and-Optimization.md#summarization)
    - [Generation Optimization](RAG-and-RAG-Optimization/RAG-and-Optimization.md#generation-optimization)
      - [Language Model Fine-Tuning](RAG-and-RAG-Optimization/RAG-and-Optimization.md#language-model-fine-tuning)
      - [Co-Training Strategies](RAG-and-RAG-Optimization/RAG-and-Optimization.md#co-training-strategies)
    - [Advanced Augmentation Techniques](RAG-and-RAG-Optimization/RAG-and-Optimization.md#advanced-augmentation-techniques)
      - [Iterative Refinement](RAG-and-RAG-Optimization/RAG-and-Optimization.md#iterative-refinement)
      - [Recursive Retrieval](RAG-and-RAG-Optimization/RAG-and-Optimization.md#recursive-retrieval)
      - [Hybrid Approaches](RAG-and-RAG-Optimization/RAG-and-Optimization.md#hybrid-approaches)
    - [Evaluation and Optimization Metrics](RAG-and-RAG-Optimization/RAG-and-Optimization.md#evaluation-and-optimization-metrics)
      - [Performance Metrics](RAG-and-RAG-Optimization/RAG-and-Optimization.md#performance-metrics)
      - [Benchmark Datasets](RAG-and-RAG-Optimization/RAG-and-Optimization.md#benchmark-datasets)
    - [Tools and Platforms for Optimization](RAG-and-RAG-Optimization/RAG-and-Optimization.md#tools-and-platforms-for-optimization)
    - [Recommendations for Implementing RAG Systems](RAG-and-RAG-Optimization/RAG-and-Optimization.md#recommendations-for-implementing-rag-systems)



### 🤝 Contributions

Contributions are welcome! Please feel free to submit issues, fork the repository, and make pull requests to enhance the repository.

### 📄 License

This repository is licensed under the Apache License. See the [LICENSE](LICENSE) file for more information.

```
{
  "title": "🚀 Awesome NLP and IR",
  "author": "Mohammad Kaif",
  "date": "July 2024",
  "sections": [
    {
      "type": "introduction",
      "content": "Welcome to the ultimate resource hub for:\n\n- **Natural Language Processing (NLP)**\n- **Large Language Models (LLMs)**\n- **LLM Efficiency**\n- **Vector Search and Information Retrieval**\n- **Retrieval Augmented Generation (RAG) Optimization and Best Practices**\n\nThis repository offers a comprehensive, well-structured, and user-friendly guide to the latest advancements, research papers, techniques, and best practices in these fields. Ideal for researchers and developers alike."
    }
  ]
}
```
