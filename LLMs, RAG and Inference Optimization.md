# LLMs
## **Large Language Models (LLMs):**
Large Language Models (LLMs) are a significant advancement in the field of natural language processing (NLP). These models are designed to understand and generate human language by leveraging deep learning techniques and vast amounts of data. Unlike simpler models, LLMs can perform a wide range of tasks, from translation and summarization to answering questions and engaging in conversational dialogues.
 - ### **Architecture:**
The architecture of LLMs is typically based on the Transformer model, which was introduced by Vaswani et al. in 2017 in the paper "Attention is All You Need". The key components of this architecture are:
  - #### 1. **Self-Attention Mechanism:** 
    This allows the model to weigh the importance of different parts of the input sequence, enabling it to capture long-range dependencies and relationships within the text.
  - #### 2. **Multi-Head Attention:** 
    Multiple attention heads allow the model to focus on different parts of the input simultaneously, enhancing its ability to understand context.
  - #### 3. **Feed-forward Neural Networks:** 
    These are applied to each position in the sequence independently, adding a layer of non-linearity.
  - #### 4. **Positional Encoding:** 
    Since Transformers do not inherently capture the order of the sequence, positional encodings are added to the input embeddings to provide information about the position of each word in the sequence.
 - ### **Training:**
LLMs are trained using large datasets that encompass a diverse range of text sources, such as books, articles, websites, and more. The training process involves:
  - #### **Pre-training:**
    The model is trained on a large corpus of text to learn language patterns. This is typically done in an unsupervised manner, where the model learns to predict the next word in a sentence or fill in the blanks.
  - #### **Fine-tuning:**
    The pre-trained model is then fine-tuned on a smaller, task-specific dataset to improve its performance on particular tasks, such as question answering or text summarization.
 - #### **Advantages:**
    - **Versatility:** LLMs can handle a wide variety of tasks without the need for task-specific architectures.
    - **Contextual Understanding:** Their ability to capture long-range dependencies makes them adept at understanding context.
    - **Scalability:** They can be scaled to larger datasets and more parameters, leading to improved performance.
  - #### **Limitations:**
    - **Computationally Intensive:** Training and deploying LLMs require significant computational resources, including powerful GPUs and extensive memory.
    - **Data Hungry:** They need vast amounts of training data to achieve high performance.
    - **Bias and Fairness:** LLMs can inadvertently learn and propagate biases present in the training data, leading to ethical concerns.
    - **Interpretability:** Their large and complex architectures make it difficult to understand and interpret their decision-making processes.

   ### **LLM Pretraining** 
  Pretraining is the process of training a machine learning model on a large dataset before fine-tuning it on a specific task. This initial phase involves learning general patterns, representations, and features from the data, which can then be adapted and refined for various downstream tasks, improving performance and efficiency.
- #### **Self Supervised Learning:**
  - Self-supervised learning (SSL) is a cutting-edge approach in machine learning where the model trains itself using automatically generated labels derived from the data, eliminating the need for manual annotation. This method harnesses the abundance of unlabeled data, which is both plentiful and inexpensive, to build robust and powerful models.
  - SSL is considered a subset of unsupervised learning, where the data itself provides the necessary supervision. The model is trained to predict one part of the input data from another part, allowing it to learn meaningful patterns and representations. This approach is particularly beneficial for training large language models (LLMs) because it enables them to extract rich and diverse features from vast amounts of text data without requiring human-labeled examples.
  - #### Key Concepts
    - **Pretext Task:** The core idea of SSL is to create a pretext task, a pseudo-task for which labels can be generated from the data itself. The model is first trained on this task, and the learned representations are then fine-tuned on the actual downstream task.
    - **Downstream Task:** After learning representations from the pretext task, the model is fine-tuned or directly applied to the downstream task, which is the actual task of interest (e.g., image classification, object detection, natural language understanding).
      
- #### **Key Techniques in Self-Supervised Learning**
    - Masked Language Modeling (MLM) and Masked Multimodal Language Modeling (MMLM)
    - Next Sentence Prediction (NSP)
    - Causal Language Modeling (CLM)
    - Denoising Autoencoders
    - Contrastive Learning
 - ### **Masked Language Modeling:**
     - In MLM, certain tokens in the input sequence are masked at random, and the model is trained to predict these masked tokens based on the context provided by the remaining tokens.
     - This approach helps the model understand bidirectional contexts.
- #### **Mathematical Formulation**
  - Given a sequence of tokens $\mathbf{X} = \{x_1, x_2, \ldots, x_n\}$, we create a corrupted version $\mathbf{X}_{\text{masked}}$ by replacing some tokens with a special `[MASK]` token.
  - Let $M$ be the set of masked positions.
  - The training objective is to maximize the likelihood of the masked tokens given the context:
  
    $$L_{\text{MLM}} = -\sum_{i \in M} \log P(x_i \mid \mathbf{X}_{\text{masked}})$$
    
  - Here, $P(x_i \mid \mathbf{X}_{\text{masked}})$ is the probability of token $x_i$ given the masked sequence.

 In MLM, the model enhances its understanding of bidirectional contexts by predicting masked tokens.


- ### **Masked Multimodal Language Modeling:**
    - In MMLM, certain tokens in the input sequence are masked at random, similar to MLM, but the model is trained using both textual and non-textual data such as images, audio, or video.
    - This approach enhances the model's ability to understand and integrate multimodal information, learning representations that capture the interplay between different modalities.

- #### **Mathematical Formulation**
  - Given a sequence of tokens $\mathbf{X} = \{x_1, x_2, \ldots, x_n\}$ and a sequence of non-textual features $\mathbf{Y} = \{y_1, y_2, \ldots, y_m\}$, we create a corrupted version $\mathbf{X}_{\text{masked}}$ by replacing some tokens with a special `[MASK]` token.
  - Let $M$ be the set of masked positions in $\mathbf{X}$.
  - The training objective is to maximize the likelihood of the masked tokens given the context provided by both the masked textual sequence and the associated non-textual data:

    $$L_{\text{MMLM}} = -\sum_{i \in M} \log P(x_i \mid \mathbf{X}_{\text{masked}}, \mathbf{Y})$$
    
  - Here, $P(x_i \mid \mathbf{X}_{\text{masked}}, \mathbf{Y})$ is the probability of token $x_i` given the masked sequence and the additional multimodal context.

By integrating information from multiple modalities, MMLM allows models to learn richer, more comprehensive representations, leading to improvements in tasks such as visual question answering, image captioning, and more complex language understanding scenarios where context is derived from both text and other types of data.
     
- ### **Next Sentence Prediction:**
    - In NSP, the model is trained to understand the relationship between pairs of sentences, determining whether a given sentence B naturally follows a given sentence A.
    - This technique is particularly useful for tasks that require a coherent understanding of longer texts, such as document-level question answering and summarization.

- #### **Mathematical Formulation**
  - Given a pair of sentences $(A, B)$, the model's task is to predict whether $B$ is the actual next sentence that follows $A$ in the corpus.
  - We introduce a binary classification variable $y \in \{0, 1\}$, where $y = 1$ indicates that $B$ is the next sentence following $A$, and $y = 0$ otherwise.
  - The training objective is to minimize the binary cross-entropy loss:

    $$L_{\text{NSP}} = -\left[ y \log P(y=1 \mid A, B) + (1 - y) \log P(y=0 \mid A, B) \right]$$
    
  - Here, $P(y=1 \mid A, B)$ is the probability that $B$ is the next sentence following $A$.

Training on NSP tasks, the model learns to capture sentence-level coherence and dependencies, enhancing its ability to perform tasks that require understanding the flow of information across multiple sentences crucial for tasks like document-level QA and summarization.

- ### **Causal Language Modeling:**
    - In CLM, the model is trained to predict the next token in a sequence, given all the previous tokens. This unidirectional approach models the probability of a token based on its preceding context.
    - This technique is particularly effective for tasks that require generative capabilities, such as text generation and language modeling.

- #### **Mathematical Formulation**
  - Given a sequence of tokens $\mathbf{X} = \{x_1, x_2, \ldots, x_n\}$, the model is trained to predict each token $x_t$ based on the preceding tokens $\{x_1, x_2, \ldots, x_{t-1}\}$.
  - The training objective is to maximize the likelihood of each token in the sequence:

    $$L_{\text{CLM}} = -\sum_{t=1}^{n} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$
    
  - Here, $P(x_t \mid x_1, x_2, \ldots, x_{t-1})$ is the probability of token $x_t$ given the preceding tokens.

CLM focuses on generating coherent text by predicting tokens based on their sequential context, making it ideal for generative tasks.


 ### **Denoising Autoencoders:**
   - Denoising Autoencoders (DAE) are trained to reconstruct the original input from a corrupted version of it. This process helps the model learn robust representations by focusing on essential features and ignoring noise.
   - This technique is widely used for tasks such as feature learning, dimensionality reduction, and anomaly detection.

- #### **Mathematical Formulation**
  - Given an input sequence $\mathbf{X} = \{x_1, x_2, \ldots, x_n\}$, we create a corrupted version $\mathbf{X}_{\text{corrupted}}$ by applying noise or perturbations.
  - The model is trained to reconstruct the original input sequence $\mathbf{X}$ from the corrupted version $\mathbf{X}_{\text{corrupted}}$.
  - The training objective is to minimize the reconstruction loss, often measured by mean squared error (MSE) or cross-entropy loss, depending on the nature of the data:

    $$L_{\text{DAE}} = \sum_{t=1}^{n} \| x_t - \hat{x}_t \|^2$$
    
  - Here, $\hat{x}_t$ is the reconstructed token corresponding to the original token $x_t$.

DAE improves model robustness by reconstructing clean inputs from corrupted versions, essential for tasks requiring accurate data representation.

- ### **Contrastive Learning:**
    - Contrastive Learning trains the model to distinguish between similar and dissimilar pairs of data. By learning to bring similar pairs closer and push dissimilar pairs apart in the representation space, the model captures meaningful patterns and structures in the data.
    - This technique is particularly effective for tasks such as image and text clustering, and representation learning.

- #### **Mathematical Formulation**
  - Given a set of input pairs $\{(\mathbf{x}_i, \mathbf{x}_i^+), (\mathbf{x}_i, \mathbf{x}_i^-)\}$ where $(\mathbf{x}_i, \mathbf{x}_i^+)$ are similar pairs and $(\mathbf{x}_i, \mathbf{x}_i^-)$ are dissimilar pairs, the model learns to minimize the distance between similar pairs and maximize the distance between dissimilar pairs.
  - The training objective is to minimize the contrastive loss, which can be defined as:

    $$L_{\text{contrastive}} = \sum_{i} \left[ \max(0, d(\mathbf{x}_i, \mathbf{x}_i^+) - d(\mathbf{x}_i, \mathbf{x}_i^-) + \alpha) \right]$$
    
  - Here, $d(\mathbf{x}_i, \mathbf{x}_j)$ is the distance measure (e.g., Euclidean distance) between the representations of $\mathbf{x}_i$ and $\mathbf{x}_j$, and $\alpha$ is a margin that enforces a minimum separation between dissimilar pairs.

This technique enhances model performance by learning discriminative features from data pairs, beneficial for tasks like image and text clustering, retrieval.

   ### **LLM Fine Tuning** 
   Fine-tuning is the process of adapting a pre-trained language model to specific downstream tasks by further training it on task-specific data. While pre-training provides a strong foundation by learning general language patterns, fine-tuning tailors the model to excel in particular applications, improving its performance on specific tasks.
   - #### **Why We Need Fine-Tuning:**
     - **Task Specialization:** Pre-trained models are generalists. Fine-tuning allows these models to specialize in particular tasks such as sentiment analysis, translation, or question answering.
     - **Improved Performance:** Fine-tuning on task-specific data enhances the model's ability to perform well on that task by leveraging the relevant information it has seen during pretraining.
     - **Efficiency:** Fine-tuning requires significantly less data and computational resources compared to training a model from scratch, making it a practical approach for many applications.
### Supervised Fine-Tuning

Supervised fine-tuning adapts a pre-trained model to a specific task using labeled data, refining the general knowledge the model gained during pre-training for particular applications.

#### Steps in Supervised Fine-Tuning

1. **Pre-trained Model Selection:**
   - Choose a model pre-trained on a large dataset (e.g., BERT, GPT, ResNet).

2. **Task-Specific Dataset:**
   - Prepare a labeled dataset relevant to the task (e.g., sentiment analysis, image classification).

3. **Model Adjustment (Optional):**
   - Modify the model architecture if necessary (e.g., add task-specific layers).

4. **Training Process:**
   - **Initialization:** Load pre-trained weights.
   - **Loss Function:** Choose a loss function (e.g., cross-entropy for classification):
     $$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$
   - **Optimizer:** Select an optimizer (e.g., Adam, SGD).

5. **Training Loop:**
   - Train the model, including forward propagation, loss calculation, backpropagation, and parameter updates:
     $$\theta = \theta - \eta \nabla_{\theta} \mathcal{L}$$
     where $\theta$ represents model parameters and $\eta$ is the learning rate.

6. **Evaluation and Validation:**
   - Monitor performance on a validation set using metrics like accuracy or F1 score to avoid overfitting.

7. **Hyperparameter Tuning:**
   - Adjust learning rate, batch size, and epochs to optimize performance.

8. **Testing:**
   - Test the fine-tuned model on an unseen test set to assess generalization.

#### Advantages

- **Improved Performance:** Enhances accuracy by specializing the model for specific tasks.
- **Efficiency:** Requires less data and computational resources than training from scratch.
- **Flexibility:** Allows the same model to be fine-tuned for various tasks.

    
- ### **Types of Fine-Tuning Methods**
    - Full Fine-Tuning
    - Parameter-Efficient Fine-Tuning (PEFT)
    - Memory-Efficient Fine-Tuning (MEFT)
    - Alignment-Based Fine-Tuning
      
 ### Full Fine-Tuning: Simplified Explanation

Full fine-tuning involves adjusting all parts of a pre-trained model $\theta$ to fit a new task-specific dataset. Here’s how it works:

#### How It Works:

1. **Starting Point**: You begin with a pre-trained model $\theta$, which has already learned useful patterns from a large dataset during its initial training.

2. **New Task**: You have a new dataset $\mathcal{D}$ for a specific task, such as recognizing objects in images or understanding sentiment in text.

3. **Adjusting the Model**: The goal is to make $\theta$ better at the new task by updating all its parameters based on $\mathcal{D}$. This is done by minimizing a measure of how well the model predicts on $\mathcal{D}$, known as the loss function $\mathcal{L}(\theta)$.

   - **Loss Function**: This function $\mathcal{L}(\theta)$ tells us how wrong the model's predictions are compared to the correct answers in `$\mathcal{D}$`.
   
     $$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(\theta; x_i, y_i)$$
   
     Here, $\ell(\theta; x_i, y_i)$ quantifies the error between the model's prediction and the true label $y_i$ for input $x_i$.

   - **Optimization**: Techniques like gradient descent are used to adjust $\theta$ to minimize $\mathcal{L}(\theta)$. This means tweaking $\theta$ in small steps to improve its performance on the new task.

#### Advantages:

- **Efficiency**: It builds on what the model already knows, saving time compared to training from scratch.
- **Performance**: Can lead to better results on the new task because the model starts with a strong foundation.

#### Challenges, including Catastrophic Forgetting:

- **Catastrophic Forgetting**: While adjusting to the new dataset, the model might forget some things it learned from its original training. This can lead to poorer performance on tasks it used to handle well.
  
- **Balancing Act**: It's important to find a balance between adapting to the new data and retaining valuable knowledge from previous training.

