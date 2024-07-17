## Large Language Models (LLMs)
Large Language Models (LLMs) are a significant advancement in the field of natural language processing (NLP). These models are designed to understand and generate human language by leveraging deep learning techniques and vast amounts of data. Unlike simpler models, LLMs can perform a wide range of tasks, from translation and summarization to answering questions and engaging in conversational dialogues.

---

### **Architecture:**
The architecture of LLMs is typically based on the Transformer model, introduced by Vaswani et al. in 2017 in the paper "Attention is All You Need". The key components of this architecture are:

- #### **1. Embedding Layer:**
  Converts the input text into dense vectors of fixed size. Each word or token is mapped to a continuous vector space where similar words have similar representations.

- #### **2. Self-Attention Mechanism:** 
  Allows the model to weigh the importance of different parts of the input sequence, enabling it to capture long-range dependencies and relationships within the text.

- #### **3. Multi-Head Attention:** 
  Multiple attention heads allow the model to focus on different parts of the input simultaneously, enhancing its ability to understand context.

- #### **4. Feed-forward Neural Networks:** 
  Applied to each position in the sequence independently, adding a layer of non-linearity to the model's transformations.

- #### **5. Positional Encoding:** 
  Since Transformers do not inherently capture the order of the sequence, positional encodings are added to the input embeddings to provide information about the position of each word in the sequence.

- #### **6. Layer Normalization (Layer Norm):**
  Normalizes the activations of the input to each sub-layer (e.g., self-attention and feed-forward layers), stabilizing and accelerating the training process.

- #### **7. Add & Norm:**
  Each sub-layer (e.g., self-attention and feed-forward layers) is followed by a residual connection around it and a layer normalization. This helps in stabilizing the training and allows the model to train deeper architectures.

---

### **Training:**
LLMs are trained using large datasets that encompass a diverse range of text sources, such as books, articles, websites, and more. The training process involves:

- #### **Pre-training:**
  The model is trained on a large corpus of text to learn language patterns. This is typically done in an unsupervised manner, where the model learns to predict the next word in a sentence or fill in the blanks.

- #### **Fine-tuning:**
  The pre-trained model is then fine-tuned on a smaller, task-specific dataset to improve its performance on particular tasks, such as question answering or text summarization.

---

### **Advantages:**
- **Versatility:** LLMs can handle a wide variety of tasks without the need for task-specific architectures.
- **Contextual Understanding:** Their ability to capture long-range dependencies makes them adept at understanding context.
- **Scalability:** They can be scaled to larger datasets and more parameters, leading to improved performance.

### **Limitations:**
- **Computationally Intensive:** Training and deploying LLMs require significant computational resources, including powerful GPUs and extensive memory.
- **Data Hungry:** They need vast amounts of training data to achieve high performance.
- **Bias and Fairness:** LLMs can inadvertently learn and propagate biases present in the training data, leading to ethical concerns.
- **Interpretability:** Their large and complex architectures make it difficult to understand and interpret their decision-making processes.

---

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
 ---
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

---
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

---     

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

--- 

- ### **Causal Language Modeling:**
    - In CLM, the model is trained to predict the next token in a sequence, given all the previous tokens. This unidirectional approach models the probability of a token based on its preceding context.
    - This technique is particularly effective for tasks that require generative capabilities, such as text generation and language modeling.

- #### **Mathematical Formulation**
  - Given a sequence of tokens $\mathbf{X} = \{x_1, x_2, \ldots, x_n\}$, the model is trained to predict each token $x_t$ based on the preceding tokens $\{x_1, x_2, \ldots, x_{t-1}\}$.
  - The training objective is to maximize the likelihood of each token in the sequence:

    $$L_{\text{CLM}} = -\sum_{t=1}^{n} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$
    
  - Here, $P(x_t \mid x_1, x_2, \ldots, x_{t-1})$ is the probability of token $x_t$ given the preceding tokens.

CLM focuses on generating coherent text by predicting tokens based on their sequential context, making it ideal for generative tasks.

--- 

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

--- 

- ### **Contrastive Learning:**
    - Contrastive Learning trains the model to distinguish between similar and dissimilar pairs of data. By learning to bring similar pairs closer and push dissimilar pairs apart in the representation space, the model captures meaningful patterns and structures in the data.
    - This technique is particularly effective for tasks such as image and text clustering, and representation learning.

- #### **Mathematical Formulation**
  - Given a set of input pairs $\{(\mathbf{x}_i, \mathbf{x}_i^+), (\mathbf{x}_i, \mathbf{x}_i^-)\}$ where $(\mathbf{x}_i, \mathbf{x}_i^+)$ are similar pairs and $(\mathbf{x}_i, \mathbf{x}_i^-)$ are dissimilar pairs, the model learns to minimize the distance between similar pairs and maximize the distance between dissimilar pairs.
  - The training objective is to minimize the contrastive loss, which can be defined as:

    $$L_{\text{contrastive}} = \sum_{i} \left[ \max(0, d(\mathbf{x}_i, \mathbf{x}_i^+) - d(\mathbf{x}_i, \mathbf{x}_i^-) + \alpha) \right]$$
    
  - Here, $d(\mathbf{x}_i, \mathbf{x}_j)$ is the distance measure (e.g., Euclidean distance) between the representations of $\mathbf{x}_i$ and $\mathbf{x}_j$, and $\alpha$ is a margin that enforces a minimum separation between dissimilar pairs.

This technique enhances model performance by learning discriminative features from data pairs, beneficial for tasks like image and text clustering, retrieval.

---
### LLM Fine Tuning

Fine-tuning is the process of adapting a pre-trained language model to specific downstream tasks by further training it on task-specific data. While pre-training provides a strong foundation by learning general language patterns, fine-tuning tailors the model to excel in particular applications, improving its performance on specific tasks.

#### Why We Need Fine-Tuning:
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

### Types of Fine-Tuning Methods

- Full Fine-Tuning
- Parameter-Efficient Fine-Tuning (PEFT)
- Memory-Efficient Fine-Tuning (MEFT)
- Alignment-Based Fine-Tuning

---

### Full Fine-Tuning

Full fine-tuning involves adjusting all parts of a pre-trained model $\theta$ to fit a new task-specific dataset. Here’s how it works:

#### How It Works:

1. **Starting Point**: You begin with a pre-trained model $\theta$, which has already learned useful patterns from a large dataset during its initial training.

2. **New Task**: You have a new dataset $\mathcal{D}$ for a specific task, such as recognizing objects in images or understanding sentiment in text.

3. **Adjusting the Model**: The goal is to make $\theta$ better at the new task by updating all its parameters based on $\mathcal{D}$. This is done by minimizing a measure of how well the model predicts on $\mathcal{D}$, known as the loss function $\mathcal{L}(\theta)$.

   - **Loss Function**: This function $\mathcal{L}(\theta)$ tells us how wrong the model's predictions are compared to the correct answers in $\mathcal{D}$.
     
     $$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(\theta; x_i, y_i)$$
     
     Here, $\ell(\theta; x_i, y_i)$ quantifies the error between the model's prediction and the true label $y_i$ for input $x_i$.

   - **Optimization**: Techniques like gradient descent are used to adjust $\theta$ to minimize $\mathcal{L}(\theta)$. This means tweaking $\theta$ in small steps to improve its performance on the new task.

#### Advantages:

- **Efficiency**: It builds on what the model already knows, saving time compared to training from scratch.
- **Performance**: Can lead to better results on the new task because the model starts with a strong foundation.

#### Challenges, including Catastrophic Forgetting:

- **Catastrophic Forgetting**: While adjusting to the new dataset, the model might forget some things it learned from its original training. This can lead to poorer performance on tasks it used to handle well.
  
- **Balancing Act**: It's important to balance adapting to the new data and retaining valuable knowledge from previous training.

---

### PEFT (Parameter-Efficient Fine-Tuning)

PEFT is a technique used in machine learning, particularly in deep learning and LLMs, where instead of updating all parameters of a pre-trained model during adaptation to a new task or dataset, only a subset of parameters are adjusted. This approach aims to optimize model performance with fewer trainable parameters than full fine-tuning methods.

#### Why PEFT is Needed:
- **Efficiency:** It reduces computational resources and time required for training by updating only the most relevant parameters, making it feasible to deploy models on hardware with limited capabilities.
- **Preservation of Knowledge:** PEFT retains valuable knowledge from pre-training, minimizing changes to the original model architecture while adapting it to new tasks.
- **Generalization:** PEFT can improve model generalization on new datasets by avoiding overfitting by focusing updates on task-relevant parameters.

#### Advantages of PEFT over Full Fine-Tuning:
- **Speed:** Faster convergence during training due to fewer parameters being updated.
- **Resource Efficiency:** Reduced memory and computational demands, suitable for deployment on hardware with constraints.
- **Flexibility:** Adaptable to various deep learning architectures and scales, including large models with millions or billions of parameters.
- **Improved Performance:** Enhanced model efficiency and effectiveness on new tasks, leveraging pre-trained knowledge effectively.

#### Types of PEFT:
- Additive PEFT
- Selective PEFT
- Reparameterized PEFT
- Hybrid PEFT


---

### Additive PEFT

Fine-tuning large pre-trained models (PLMs) involves adapting them to specific downstream tasks, but this process can be computationally expensive and may degrade generalization. Additive PEFT methods address these challenges by introducing minimal trainable parameters strategically within the model architecture, preserving the bulk of the pre-trained weights.

#### 1. Adapters

Adapters are small, task-specific layers inserted within Transformer blocks. Each adapter consists of:
- **Down-projection matrix** $W_{\text{down}}$
- **Non-linear activation function** $\sigma$
- **Up-projection matrix** $W_{\text{up}}$

The adapter module applies a residual connection to compute:
$$\text{Adapter}(x) = W_{\text{up}} \sigma (W_{\text{down}} x) + x$$

#### Adapter Variants

- **Serial Adapter**: Placed after the self-attention and feed-forward network (FFN) layers in each Transformer block.
- **Parallel Adapter**: Runs alongside each Transformer sublayer, maintaining model parallelism.
- **CoDA**: Combines parallel adapters with sparse activation mechanisms for computational efficiency.

#### 2. Soft Prompt-based Fine-tuning

This approach enhances model performance by appending continuous vectors (soft prompts) to input sequences, leveraging the rich information in embedding spaces rather than discrete tokens.

#### Prominent Approaches

- **Prefix-tuning**: Extends the model's capacity by adding learnable vectors to keys and values across all Transformer layers. Variants like p-tuning v2 and Adaptive Prefix Tuning (APT) refine this method further.

#### 3. $IA^3$: Infused Adapter by Inhibiting and Amplifying Inner Activations

$IA^3$ introduces minimal additional parameters while effectively adapting the model to new tasks.

#### Core Principles

1. **Parameter Efficiency**: Minimizes additional parameter overhead to reduce computational costs and memory usage.
   
2. **Mixed-Task Batches**: Supports handling mixed-task batches within the same computational graph, optimizing efficiency and flexibility.
   
3. **Activation Modification**: Utilizes element-wise rescaling with task-specific vectors ($lk$, $lv$, $lff$) to adjust attention and feed-forward mechanisms dynamically.

#### Implementation Details

- **Controlled Parameter Addition**: Introduces a limited number of new parameters ($dk$, $dv$, $dff$) per Transformer layer block to maintain computational efficiency and model stability.

#### Advantages and Performance

1. **Superior Accuracy**: Outperforms traditional fine-tuning methods in few-shot learning scenarios by preserving the model's pre-trained weights.
   
2. **Practical Application**: Simplifies deployment in production environments by maintaining robust performance across diverse tasks without extensive retraining.

---

### Selective PEFT

Selective PEFT methods focus on fine-tuning a subset of existing parameters rather than introducing additional parameters. This approach aims to enhance model performance on specific downstream tasks while minimizing computational overhead. Selective PEFT can be broadly categorized into unstructured and structured masking techniques.

- #### Unstructured Masking

  Unstructured masking involves applying binary masks to the model's parameters to determine which ones are updated during fine-tuning. The binary mask $M = { m_i }$ indicates whether a parameter $θ_i$ is frozen (0) or trainable (1). The updated parameters after fine-tuning are calculated as:

  $$θ_i' = θ_i - η \cdot BL_{θ_i} \cdot m_i$$

  where η is the learning rate, $BL_{θ_i}$ is the gradient of the loss function with respect to $θ_i$. This selective updating process optimizes resource allocation by focusing computational efforts on task-critical parameters.

  Representative methods in unstructured masking include:
  - **Diff Pruning**: Uses a differentiable L0-norm penalty to regularize a learnable binary mask applied to model weights.
  - **FishMask**: Uses Fisher information to determine parameter importance, selecting top parameters for updating based on task relevance.
  - **Child-tuning**: Introduces dynamic selection of a `child` network during training iterations, where only parameters within the chosen network are updated.

- #### Structured Masking

  Structured masking organizes parameter selection into regular patterns rather than applying it randomly, enhancing computational and hardware efficiency during training.

  Techniques in structured selective PEFT include:
  - **Structured Pruning**: Techniques like Diff Pruning partition weight parameters into local groups and systematically prune them based on predefined criteria, improving computational efficiency.
  - **Bitfit**: Focuses on fine-tuning bias parameters of DNN layers, demonstrating competitive results for smaller models.
  - **SPT (Sensitivity-aware Parameter-Efficient Fine-Tuning)**: Identifies sensitive parameters through first-order Taylor expansions, selecting and fine-tuning only those critical for task performance.

---

### Background of Matrix Decomposition

Matrix decomposition, also known as factorization, breaks down matrices into simpler components. It is essential across mathematics, engineering, and data science for simplifying operations and revealing underlying structures.

- #### Why Matrix Decomposition is Essential:
  1. **Dimensionality Reduction:** Simplifies data/models by retaining key information in lower-dimensional forms, reducing complexity and storage needs.
  2. **Feature Extraction:** Identifies data patterns/features crucial in analysis and machine learning, enhancing efficiency in processing and interpretation.
  3. **Numerical Stability:** Enhances stability in computations, particularly for large or ill-conditioned matrices, by reducing errors through decomposition.
  4. **Algorithm Efficiency:** Speeds up operations and reduces memory usage compared to original matrices, crucial in fields like image/signal processing.

- #### Advantages:
  - **Simplicity and Interpretability:** Simplifies relationships within data/models (e.g., PCA's eigen-decomposition).
  - **Computational Efficiency:** Faster operations and lower memory use post-decomposition, vital for large datasets/models.
  - **Optimization Applications:** Crucial in solving linear equations and iterative algorithms (e.g., SVD in recommendations).
  - **Flexibility:** Various methods (e.g., SVD, QR, LU) cater to different challenges, adaptable to diverse applications.
  - **Feature Transformation:** Enhances feature representation, e.g., in deep learning, via matrix transformations.

  
---

### Reparameterized PEFT
Reparameterized PEFT optimizes the fine-tuning process by restructuring a model's parameters. This approach reduces the number of parameters requiring updates, cutting computational and memory costs while maintaining or improving performance. By introducing efficient parameter modifications, PEFT enhances the scalability and effectiveness of fine-tuning operations.

#### LoRA: Low-Rank Adaptation

LoRA (Low-Rank Adaptation) is a widely recognized reparameterization technique.

- **Concept:**
  - For a given pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA introduces two trainable weight matrices:
    - $W_{up} \in \mathbb{R}^{d \times r}$
    - $W_{down} \in \mathbb{R}^{r \times k}$
  - Here, the rank $r \leq \min(d, k)$.
  - These matrices operate in parallel to $W_0$.

- **Operation:**
  - The input $h_{\text{in}}$ typically produces the output $h_{\text{out}} = W_0 h_{\text{in}}$.
  - LoRA modifies this output by introducing an incremental update $\Delta W$:

    $$h_{\text{out}} = W_0 h_{\text{in}} + \frac{\alpha}{r} W_{up} W_{down} h_{\text{in}}$$

    where $\alpha$ is a scaling factor.

- **Implementation:**
  - Initially, $W_{down}$ is randomized, and $W_{up}$ is zeroed out, ensuring $\Delta W$ starts at zero.
  - LoRA is straightforward to implement and effective on models with up to 175 billion parameters.
  - Once fine-tuning is complete, LoRA’s adaptive weights integrate seamlessly with the pre-trained backbone, maintaining efficiency without adding inference burden.

#### DyLoRA: Dynamic LoRA

DyLoRA addresses the challenge of selecting an appropriate rank in LoRA training.

- **Concept:**
  - DyLoRA trains the LoRA module on a range of ranks within a predefined training budget.
  - For a given rank range, DyLoRA dynamically chooses a rank at each iteration.

- **Operation:**
  - Matrices $W_{down}$ and $W_{up}$ are tailored for the selected rank, reducing the training time required to find an optimal rank.

#### AdaLoRA: Adaptive LoRA

AdaLoRA reformulates $\Delta W$ with a singular value decomposition (SVD).

- **Concept:**
  - $\Delta W = P \Lambda Q$, where:
    - $P$ and $Q$ are orthogonal matrices.
    - $\Lambda$ is a diagonal matrix containing singular values.

- **Operation:**
  - Singular values are pruned iteratively during training based on their importance scores.
  - An additional regularizer term is included in the loss to ensure orthogonality:

    $$R(P, Q) = \| P^T P - I \|^2_F + \| QQ^T - I \|^2_F$$

- **Advantages:**
  - This approach allows the model to dynamically adjust the rank within each LoRA module, effectively managing its parameter counts.
  - AdaLoRA delivers superior performance by leveraging a predefined training budget for pruning, orthogonality maintenance, and learning module-specific ranks dynamically.

#### DoRA: Weight-Decomposed Low-Rank Adaptation

DoRA (Weight-Decomposed Low-Rank Adaptation) is an advanced parameter-efficient fine-tuning (PEFT) method designed to enhance the performance of models like LoRA (Low-Rank Adaptation). It improves learning capacity and stability while maintaining efficiency and avoiding extra inference costs.

- **Key Concepts:**
  1. **Weight Decomposition**:
     - **Magnitude**: Represents the scale of the weights.
     - **Direction**: Represents the orientation of the weights in the parameter space.

  2. **Directional Updates**:
     - Utilizes LoRA to update the directional component of the weights, enabling efficient fine-tuning with fewer trainable parameters.

  3. **Learning Capacity**:
     - By focusing on both magnitude and direction, DoRA closely resembles the learning capacity of full fine-tuning.

- **Implementation Steps:**
  1. **Weight Decomposition**:
     - Decompose the pre-trained weight matrix $W$ into magnitude ($m$) and directional ($V$) components:
       $$W = m \cdot \frac{V}{\|V\|_c}$$
       where $\| \cdot \|_c$ denotes the vector-wise norm across each column.

  2. **Fine-Tuning with LoRA**:
     - Apply LoRA to update the directional component:
       $$\Delta W = B \cdot A$$
       where $B$ and $A$ are low-rank matrices trained to adapt the directional component.

  3. **Magnitude Updates**:
     - Make the magnitude component ($m$) trainable, allowing it to adjust during fine-tuning.

  4. **Combine Components**:
     - After fine-tuning, combine the updated magnitude and directional components:
       $$W' = m \cdot \frac{W_0 + \Delta V}{\|W_0 + \Delta V\|_c}$$
       where $W_0$ is the initial pre-trained weight matrix and $\Delta V$ represents the directional updates.

- **Benefits of DoRA:**
  1. **Enhanced Learning Capacity**:
     - Closely mimics full fine-tuning by adapting both magnitude and direction.

  2. **Stability**:
     - Simplifies optimization, making training more stable and less sensitive to initialization.

  3. **Efficiency**:
     - No additional inference overhead, ensuring efficient model deployment.

- **Applications:**
  - DoRA consistently outperforms LoRA across various tasks and model architectures, including:
    1. **Commonsense Reasoning**:
       - Improvements on models like LLaMA, LLaVA, and VL-BART.

    2. **Visual Instruction Tuning**:
       - Better performance on vision-language tasks.

    3. **Image/Video-Text Understanding**:
       - Enhanced results on multimodal data benchmarks.


---

### Hybrid PEFT

The effectiveness of Parameter-Efficient Fine-Tuning (PEFT) methods varies across tasks. Thus, many studies focus on combining the advantages of different PEFT approaches or unifying them through commonalities. Here are some important approaches:

#### 1. UniPELT

UniPELT integrates LoRA, prefix-tuning, and adapters within each Transformer block, using a gating mechanism to control the activation of PEFT submodules. This mechanism consists of three small feed-forward networks (FFNs), each producing a scalar value $G \in [0,1]$, applied to LoRA, prefix, and adapter matrices respectively. UniPELT consistently improves accuracy by 1% to 4% across various setups.

#### 2. S4

S4 explores design spaces for Adapter (A), Prefix (P), BitFit (B), and LoRA (L), identifying key design patterns:

- **Spindle Grouping**: Divides Transformer layers into four groups $G_i$ for $i \in \{1,2,3,4\}$, with each group applying similar PEFT strategies.
- **Uniform Parameter Allocation**: Distributes trainable parameters uniformly across layers.
- **Diverse Strategies per Group**: Assigns different PEFT strategies to different groups. Optimal configuration:
  - $G_1$: (A, L)
  - $G_2$: (A, P)
  - $G_3$: (A, P, B)
  - $G_4$: (P, B, L)

#### 3. MAM Adapter

MAM Adapter examines the similarities between adapters, prefix-tuning, and LoRA, creating three variants:

- **Parallel Adapter**: Places adapter layers alongside specific layers (SA or FFN).
- **Multi-head Parallel Adapter**: Divides the parallel adapter into multiple heads affecting head attention output in SA.
- **Scaled Parallel Adapter**: Adds a scaling term post-adapter layer, akin to LoRA.

The best setup, called the MAM Adapter, uses prefix-tuning in the SA layer and a scaled parallel adapter in the FFN layer.

#### 4. LLM-Adapters

LLM-Adapters offer a framework incorporating various PEFT techniques into large language models (LLMs). Key insights include:

- Effective placements for series adapters, parallel adapters, and LoRA are after MLP layers, alongside MLP layers, and following both Attention and MLP layers, respectively.
- Smaller LLMs with PEFT can match or surpass larger models on certain tasks.

#### 5. Neural Architecture Search (NAS)

NAS is used to discover optimal PEFT combinations:
- **NOAH**: Uses NAS to find the best PEFT configurations for each dataset, employing AutoFormer, a one-shot NAS algorithm. The search space includes Adapter, LoRA, and Visual Prompt Tuning (VPT).
- **AUTOPEFT**: Defines a search space with serial adapters, parallel adapters, and prefix tuning, using high-dimensional Bayesian optimization for effective NAS. Both NOAH and AUTOPEFT show NAS's potential in optimizing PEFT configurations across various tasks.

--- 

### Memory-Efficient Fine Tuning

Memory-Efficient Fine-Tuning (MEFT) is an umbrella term encompassing various strategies and techniques aimed at fine-tuning large language models (LLMs) and other deep learning models in a way that minimizes memory consumption. This is crucial for making the deployment and adaptation of large models feasible on hardware with limited memory resources, such as GPUs with lower VRAM or even edge devices.

#### Types of MEFT:

1. LoRA-FA (LoRA with Frozen Activations)
2. HyperTuning
3. Memory-Efficient Zeroth-Order Optimizer (MeZO)
4. QLoRA: Quantized Low-Rank Adaptation
5. Expert-Specialized Fine-Tuning
6. Sparse Matrix Tuning
7. Representation Finetuning (ReFT)

---

#### 1. LoRA-FA (LoRA with Frozen Activations)

##### Introduction

LoRA-FA, or Low-Rank Adaptation with Frozen Activations, is an advanced technique for fine-tuning large language models (LLMs). By leveraging low-rank matrix approximations and selectively freezing activations during training, this approach reduces computational and memory demands, enhancing efficiency without compromising performance.

##### Core Concepts

###### Low-Rank Adaptation (LoRA)
- **Objective**: Reduce the number of trainable parameters in large models.
- **Method**: Decompose weight matrices $W$ into two low-rank matrices $A$ and $B$ such that $W \approx A \times B$.
- **Benefit**: Lowers parameter count and computational complexity.

###### Frozen Activations
- **Objective**: Decrease computational load by freezing specific layer activations.
- **Method**: Selectively freeze activations of chosen layers, skipping gradient computation for these layers during backpropagation.
- **Benefit**: Saves computational resources and memory by reducing forward and backward passes.

##### Implementation Details

###### Architecture Modifications
- **Base Model**: Applicable to any large pre-trained model (e.g., BERT, GPT).
- **Layer Modification**: Introduce low-rank matrices $A$ and $B$ in target layers.
- **Freezing Strategy**: Determine which layers' activations to freeze based on computational cost and task-specific performance impact.

###### Training Procedure
1. **Initialization**: Start with a pre-trained model and initialize low-rank matrices $A$ and $B$.
2. **Activation Freezing**: Select layers for activation freezing, either statically or dynamically.
3. **Fine-Tuning**: Train with adjusted learning rates and batch sizes to optimize the low-rank components while keeping frozen layers unchanged.

###### Hyperparameters
- **Rank (r)**: Balances performance and efficiency; critical for low-rank matrices.
- **Learning Rate (lr)**: Typically lower than standard rates for stable convergence.
- **Freezing Strategy**: Requires empirical evaluation or computational profiling to select optimal layers for freezing.

##### Advantages and Challenges

###### Advantages
- **Computational Efficiency**: Reduces training resource requirements.
- **Scalability**: Enables fine-tuning of larger models on resource-constrained hardware.
- **Performance**: Maintains or enhances model performance compared to full fine-tuning.

###### Challenges
- **Layer Selection**: Identifying which activations to freeze is crucial and can impact performance.
- **Hyperparameter Tuning**: Optimizing rank and learning rate is complex and essential.
- **Implementation Complexity**: Managing low-rank matrices and frozen activations adds complexity to the training pipeline.

##### Applications
- **Fine-Tuning Large Models**: Adapt extensive pre-trained models to specific tasks with minimized computational costs.
- **Resource-Constrained Environments**: Suitable for deploying models in settings with limited computational resources.
- **Rapid Prototyping**: Facilitates quicker development cycles by reducing training times.

---

#### 2. HyperTuning

HyperTuning is an advanced method for fine-tuning large language models efficiently. It utilizes hypermodels to generate task-specific tuning parameters, optimizing performance while reducing computational requirements. This method integrates Parameter-Efficient Fine-Tuning (PEFT) techniques such as Prefix Tuning and LoRA (Low-Rank Adaptation).

##### Key Concepts

###### Parameter-Efficient Fine-Tuning (PEFT)

1. **Prefix Tuning**:
   - Adds learnable prefix tokens to each transformer layer's input.
   - Modifies the attention mechanism without changing the model parameters.

2. **LoRA (Low-Rank Adaptation)**:
   - Adjusts the linear maps within the model in a low-rank manner.
   - Significantly reduces the number of parameters to be updated.

###### HyperModels

- Hypermodels generate parameters for other models.
- In HyperTuning, a hypermodel processes few-shot examples to generate PEFT parameters for the downstream model.

##### HyperTuning Framework

###### Architecture

- **HyperT5**:
  - **Encoder**: Processes few-shot input examples.
  - **Decoder**: Generates token representations.
  - **MLPs**: Convert token representations into PEFT parameters.
  - Integrates Prefix Tuning and LoRA for parameter generation.

###### HyperPretraining

- **Objective**: Prepare the hypermodel to generate effective PEFT parameters using Context-Augmented Conditional Language Modeling (CACLM).
- **Process**:
  1. **Data Preparation**: Sample and segment sequences from a pretraining corpus.
  2. **Segmentation**: Split sequences into parts (A, B, C, and D).
  3. **Training**:
     - **Hypermodel**: Uses segments A and D.
     - **Downstream Model**: Uses segments B and C.

###### Multi-Task Fine-Tuning (MTF)

- **Training Setup**:
  - Few-shot inputs for the hypermodel.
  - Target example for the downstream model with generated PEFT parameters.
  - Only hypermodel parameters are updated during training.

- **Datasets**:
  - **P3**: 62 task datasets for multi-task learning.
  - **MetaICL**: Few-shot learning tasks dataset.
  - **Super-NaturalInstructions (S-NI)**: Over 1,600 task datasets for evaluation.

##### Evaluation and Results

- **Metrics**:
  - **P3**: Multiple-choice accuracy.
  - **MetaICL**: ROUGE or Macro-F1 scores.
  - **S-NI**: ROUGE-L scores.

- **Performance**:
  - **P3**: HyperT5 with Prefix and LoRA performs nearly as well as full fine-tuning.
  - **MetaICL**: Significant improvement with HyperTuning.
  - **S-NI**: Outperforms standalone PEFT methods and approaches full fine-tuning performance.

##### Intuition

HyperTuning uses hypermodels to generate task-specific parameters dynamically, minimizing the need for extensive fine-tuning. This approach makes the downstream model adaptable to various tasks with minimal computational overhead.

##### Benefits

- **Efficiency**: Reduces computational costs by minimizing parameter updates.
- **Performance**: Achieves high task performance, comparable to full fine-tuning.
- **Versatility**: Applicable to a wide range of NLP tasks.

##### Applications

HyperTuning is ideal for scenarios requiring high-performance NLP models with limited computational resources. It is suitable for real-time applications, mobile devices, and environments needing quick adaptation to new tasks.

##### Conclusion

HyperTuning is a novel method combining hypermodels and PEFT techniques to fine-tune large language models efficiently. It strikes a balance between computational efficiency and task performance, making it a promising solution for diverse NLP applications.

---

#### 3. Zero-Redundancy Optimizer (ZeRO) and Memory-Efficient Zeroth-Order Optimizer (MeZO)

##### Zero-Redundancy Optimizer (ZeRO)

###### Introduction
ZeRO (Zero Redundancy Optimizer) is a memory optimization technique designed to enable the training of large-scale language models by distributing memory loads across multiple devices.

###### Key Concepts

1. **Optimizer State Partitioning**
   - **Optimizer states**: Momentum, variance (e.g., in Adam).
   - **Partitioning**: States are divided among $d$ devices.
   - **Memory reduction**: $\mathcal{O}\left(\frac{|O|}{d}\right)$, where $|O|$ is the size of optimizer states.

2. **Gradient Partitioning**
   - **Gradients**: Computed during backpropagation.
   - **Partitioning**: Gradients are split across devices.
   - **Memory reduction**: $\mathcal{O}\left(\frac{|G|}{d}\right)$, where $|G|$ is the size of gradients.

3. **Parameter Partitioning**
   - **Parameters**: Model weights.
   - **Partitioning**: Parameters are distributed across devices.
   - **Memory reduction**: $\mathcal{O}\left(\frac{|P|}{d}\right)$, where $|P|$ is the size of parameters.

###### Three Stages of ZeRO

1. **Stage 1: Optimizer State Partitioning**
   - Distributes optimizer states to reduce memory per device.

2. **Stage 2: Gradient Partitioning**
   - Further reduces memory by partitioning gradients.

3. **Stage 3: Parameter Partitioning**
   - Combines all partitioning techniques for maximal memory efficiency.
   - **Memory per Device**: $\mathcal{O}\left(\frac{|P| + |O| + |G|}{d}\right)$.

###### Benefits
- **Scalability**: Supports training larger models.
- **Efficiency**: Reduces memory redundancy.
- **Flexibility**: Works with various distributed training setups.


##### Memory-Efficient Zeroth-Order Optimizer (MeZO)

###### Introduction
MeZO (Memory-Efficient Zeroth-Order Optimizer) uses zeroth-order optimization, relying only on forward passes, eliminating the need for backpropagation and reducing memory usage.

###### Key Concepts

1. **Gradient Estimation**
   - **Zeroth-order optimization**: Estimates gradients using only loss function evaluations.
   - **Formula**:
     $$\hat{\nabla} \mathcal{L}(\theta) = \frac{\mathcal{L}(\theta + \delta \mathbf{u}) - \mathcal{L}(\theta - \delta \mathbf{u})}{2\delta} \mathbf{u}$$
     where $\delta$ is a small scalar perturbation and $\mathbf{u}$ is a random direction vector.

###### MeZO Algorithm

1. **Initialization**: Start with parameters $\theta_0$.
2. **Forward Passes**: Compute $\mathcal{L}(\theta + \delta \mathbf{u})$ and $\mathcal{L}(\theta - \delta \mathbf{u})$.
3. **Gradient Estimate**: Use the zeroth-order formula.
4. **Parameter Update**:
   $$\theta_{t+1} = \theta_t - \eta \hat{\nabla} \mathcal{L}(\theta_t)$$
   where $\eta$ is the learning rate.

###### Memory Efficiency
- **In-Place Updates**: Parameters are updated without storing intermediate states.
- **No Backpropagation**: Reduces memory overhead.

###### Performance and Compatibility
- **Efficiency**: Comparable to traditional fine-tuning methods.
- **PEFT Compatibility**: Works with techniques like LoRA and prefix tuning.
- **Non-Differentiable Objectives**: Can optimize non-differentiable objectives.

##### Practical Use and Benefits

1. **Training Larger Models**
   - **ZeRO and MeZO** enable training on limited hardware by optimizing memory use.

2. **Resource Efficiency**
   - **ZeRO**'s distributed memory and **MeZO**'s forward-pass-only technique ensure efficient resource use.

3. **Flexibility**
   - Compatible with various optimization and fine-tuning strategies.

4. **Cost Reduction**
   - Lower memory requirements reduce overall training costs.

---

#### 4. QLoRA: Quantized Low-Rank Adaptation

QLoRA (Quantized Low-Rank Adaptation) is an advanced technique designed to efficiently fine-tune large language models. It integrates quantization, double quantization, low-rank adaptation (LoRA), and paged attention to address computational and storage challenges.

##### Key Concepts

1. **Quantization**

   Quantization reduces the precision of model parameters from 32-bit floating point to lower precision formats like 8-bit or 4-bit integers, decreasing memory and computational requirements.

2. **Double Quantization**

   Double quantization refines the quantization process by adding an intermediate step:

   $$\mathbf{w}' = \mathbf{Q}_2(\mathbf{Q}_1(\mathbf{w}))$$

   where $\mathbf{Q}_1$ and $\mathbf{Q}_2$ are the first and second quantization functions, respectively.

3. **Low-Rank Adaptation (LoRA)**

   LoRA introduces trainable low-rank matrices to adjust pre-trained model weights during fine-tuning:

   $$\mathbf{W}_{\text{adapted}} = \mathbf{W} + \alpha \mathbf{AB}$$

   where:

   - $\mathbf{W}$ is the original weight matrix.
   - $\mathbf{A}$ and $\mathbf{B}$ are low-rank matrices.
   - $\alpha$ is a scaling factor.

4. **Paged Attention**

   Paged attention optimizes memory usage during transformer model attention mechanisms by dividing the attention matrix into smaller, manageable pages.

5. **4-bit NormalFloat (NF4)**

   4-bit NormalFloat (NF4) is a new data type that is information-theoretically optimal for quantizing normally distributed weights.


##### How QLoRA Works

- **Quantization of Pre-trained Model:**

  Parameters are quantized using double quantization:

  $$\mathbf{W}' = \mathbf{Q}_2(\mathbf{Q}_1(\mathbf{W}))$$

- **Application of LoRA:**

  Apply LoRA to the quantized model:

  $$\mathbf{W}_{\text{adapted}}' = \mathbf{W}' + \alpha \mathbf{AB}$$

- **Implementation of Paged Attention:**

  Use paged attention during fine-tuning and inference:

  $$\mathbf{A} = \bigcup_{i=1}^{n} \mathbf{A}_i$$

  where $\mathbf{A}_i$ represents a page of the attention matrix.

##### Benefits of QLoRA

- **Efficiency:**
  - Reduces computational and memory resources.
  - Allows fine-tuning on consumer-grade hardware.

- **Scalability:**
  - Enables fine-tuning of very large models.

- **Performance:**
  - Maintains competitive performance.
  - Preserves accuracy through double quantization and efficient memory usage with paged attention.

##### Practical Applications

- **Domain-Specific Adaptation:** Tailoring large language models for medical, legal, or customer service applications.
- **Resource-Efficient Deployment:** Reducing operational costs by lowering resource requirements for deploying large models.

---

#### 5. Expert-Specialized Fine-Tuning for Sparse Large Language Models
As large language models (LLMs) become increasingly complex, efficient customization techniques are critical. Traditional parameter-efficient fine-tuning (PEFT) methods cater to dense models, while sparse models, particularly those utilizing the Mixture-of-Experts (MoE) architecture, necessitate specialized approaches. This document introduces Expert-Specialized Fine-Tuning (ESFT), a method designed to enhance the tuning efficiency and performance of MoE LLMs.

##### Key Findings

###### Concentration of Expert Activation
- **Observation**: Custom tasks often activate a small, specific set of experts within the model.
- **Implication**: Different tasks tend to activate distinct sets of experts, underscoring the need for task-specific tuning.

###### Efficiency of Expert-Specialized Fine-Tuning (ESFT)
- **Method**: ESFT fine-tunes only the most relevant experts for a given task, keeping others frozen.
- **Result**: This targeted approach improves tuning efficiency and achieves performance comparable to or better than full-parameter fine-tuning.

###### Advantages of MoE Architecture
- **Design**: MoE models with finer-grained experts can select more relevant expert combinations.
- **Benefit**: Improved efficiency and effectiveness in training and inference.


##### Methods

###### Mixture-of-Experts (MoE) for Transformers
- **Architecture**: MoE replaces traditional Feed-Forward Networks (FFNs) with multiple experts. Each token is processed by the most relevant experts based on learned affinity scores.
- **Advancements**: Includes fine-grained segmentation and shared expert isolation, enhancing efficiency through specialization.

###### Probing Task-Specific Expert Specialization
- **Findings**: A small subset of experts is activated for most tasks, and different tasks activate distinct sets of experts.
- **Implications**: Task-specific tuning is essential to maximize the potential of MoE models.

###### Expert-Specialized Fine-Tuning (ESFT)
- **Approach**: Select and fine-tune only the most relevant experts for a given task.
- **Selection Metrics**: Average gate score and token selection ratio are used to identify relevant experts.
- **Benefits**: Reduces computational resources needed for fine-tuning while maintaining or improving performance.

##### Benefits of ESFT

###### Maintaining Expert Specialization
- By updating only relevant experts, ESFT preserves the pre-trained specialization of other experts, preventing performance degradation in non-relevant tasks.

###### Resource Efficiency
- ESFT significantly reduces storage and computational resources, making it ideal for resource-constrained environments.

##### Experimental Evaluation

###### Enhancing Existing Abilities
- **Domains**: Tested on tasks where LLMs had pre-existing proficiency, such as math and code.
- **Results**: Showed significant performance improvements.

###### Adapting to New Tasks
- **Specialized Tasks**: Evaluated on tasks like text-to-JSON intent recognition, text summarization, legal judgment prediction, and low-resource translation.
- **Effectiveness**: ESFT enabled effective adaptation to these new and specialized tasks.

---

#### 6. Sparse Matrix Tuning in Large Language Model Fine-Tuning
Large Language Models (LLMs) are powerful but require significant computational resources for fine-tuning. Traditional methods involve adjusting all model parameters, which is often costly in terms of memory and computation. Parameter-efficient fine-tuning (PEFT) methods like LoRA (Low-Rank Adaptation) aim to reduce these costs but usually fall short in performance compared to full fine-tuning. Sparse Matrix Tuning (SMT) offers a middle ground, aiming to achieve high performance while minimizing resource usage.

##### Sparse Matrix Tuning (SMT) Approach
SMT identifies and updates only the most important sub-matrices within the model's weight matrices, reducing computational and memory costs without sacrificing performance.

###### Intuition Behind SMT
The key idea is that not all parts of the weight matrices are equally important for a given task. By focusing on the most significant sub-matrices, SMT achieves efficient fine-tuning.

##### How SMT Works

###### Identifying Important Sub-matrices
1. **Warm-up Phase**: SMT begins with a warm-up phase of 100 iterations, monitoring the gradients of the weight matrices to identify significant changes. This can be expressed as:

   $$\Delta W_{i,j} = \frac{\partial L}{\partial W_{i,j}}$$

   where $L$ is the loss function and $W_{i,j}$ are the elements of the weight matrices.

2. **Selection Process**: Sub-matrices exhibiting the largest gradient changes are selected for fine-tuning:

   $$S = \{(i,j) \mid |\Delta W_{i,j}| > \theta \}$$

   where $\theta$ is a predefined threshold indicating significant change.

###### Fine-tuning Selected Sub-matrices
1. **Targeted Updates**: Only the selected sub-matrices are updated during fine-tuning, drastically reducing the number of trainable parameters.
2. **Freezing Remaining Parameters**: The rest of the model's parameters remain unchanged, reducing memory and computational costs.

##### Benefits of SMT

1. **Performance**
   - SMT matches or exceeds the performance of full fine-tuning and surpasses other PEFT methods like LoRA and DoRA in various benchmarks.

2. **Efficiency**
   - **Memory Reduction**: GPU memory usage is reduced by 67% compared to full fine-tuning.
   - **Consumer-grade GPU Compatibility**: Allows fine-tuning on GPUs like the NVIDIA RTX 4090.

3. **Consistent Performance**
   - SMT maintains high performance without the performance decline seen in other PEFT methods as the number of trainable parameters increases.

##### Implementation Details

1. **Custom Sparse Linear Layers**
   - These layers compute necessary gradients only for selected sub-matrices, reducing memory and computation requirements for backpropagation and parameter updates.

2. **Optimizer Efficiency**
   - Focusing on selected sub-matrices reduces the memory for storing gradients in optimizers like Adam.

---

#### 7. Representation Finetuning (ReFT)
Representation Finetuning (ReFT) is an innovative technique designed to adapt large language models (LLMs) for specific tasks by modifying their internal representations rather than their parameters. This method aims to balance computational efficiency with high performance, making it an appealing alternative to traditional finetuning approaches.

##### Motivation

###### Challenges with Traditional Finetuning
- **Resource Intensity:** Adjusting the parameters of a large model can be computationally expensive and memory-intensive.
- **Parameter-Efficient Finetuning (PEFT):** While PEFT methods reduce the number of trainable parameters, they still require modifying the model's parameters, which can be cumbersome and less efficient.

###### Advantages of ReFT
- **Focus:** ReFT adjusts the hidden representations (activations) of a frozen base model, offering a resource-efficient way to adapt the model to specific tasks.
- **Efficiency:** By not directly modifying model parameters, ReFT reduces computational and memory overhead.

##### Core Concepts

1. **Frozen Base Model**
   - **Definition:** The model's original parameters remain unchanged during finetuning.
   - **Benefit:** Significantly reduces the computational burden associated with training large models.

2. **Hidden Representations**
   - **Definition:** Intermediate states produced by the model's layers during the forward pass.
   - **Importance:** Contains rich semantic information for task-specific adaptation.

3. **Low-dimensional Subspace**
   - **Definition:** ReFT operates within a constrained subspace of the hidden representations.
   - **Benefit:** Ensures computational efficiency and prevents overfitting by limiting adjustments.

##### Methods

ReFT can be implemented using different techniques to adjust hidden representations. Two notable methods are Low-rank Linear Subspace ReFT (LoReFT) and Direct ReFT (DiReFT).

###### Low-rank Linear Subspace ReFT (LoReFT)

###### Mathematical Formulation

Given a model with hidden representation $H \in \mathbb{R}^{d \times n}$:

- **Transformation:** Apply a low-rank projection $$L :H' = H + UV^T H$$
  - **Low-rank Projection:** Defined as:
    $$L(H) = UV^T$$
    - $U \in \mathbb{R}^{d \times r}$ (low-rank matrix 1)
    - $V \in \mathbb{R}^{r \times d}$ (low-rank matrix 2)
    - $r \ll d$ (rank of the projection)

###### Direct ReFT (DiReFT)

###### Mathematical Formulation

Given a model with hidden representation $H$:

- **Transformation:** Directly adjust $H$ with matrix $W$:
  $$H' = H + W$$
  - $W \in \mathbb{R}^{d \times n}$ (learned adjustment matrix)

##### Implementation Steps

1. **Extract Hidden Representations**
   - During the forward pass, extract the hidden representations $H$ from a specific layer of the frozen base model.

2. **Apply Transformation**
   - **LoReFT:**
     - Define low-rank matrices $U$ and $V$.
     - Compute the transformation: $H' = H + UV^T H$
   - **DiReFT:**
     - Define the adjustment matrix $W$.
     - Apply the transformation: $H' = H + W$

3. **Task-specific Finetuning**
   - Use the modified hidden states $H'$ for the downstream task.
   - Optionally adjust the parameters of the task-specific head if needed.

##### Applications

ReFT is versatile and can be applied to various natural language processing tasks:

- Commonsense Reasoning
- Arithmetic Reasoning
- Instruction-following
- Natural Language Understanding tasks like sentiment analysis, question answering, and text classification.

##### Benefits

- **Efficiency:** By not adjusting the model's parameters, ReFT significantly lowers computational and memory overhead.
- **Performance:** Despite the efficiency, ReFT achieves performance levels comparable to or better than traditional finetuning and other PEFT methods.
- **Flexibility:** ReFT can be easily integrated with existing models, requiring minimal modifications to their architecture.

--- 

## Alignment-Based Fine-Tuning

Alignment-based fine-tuning involves adjusting a large language model (LLM) to ensure its behavior aligns with specific goals, such as ethical guidelines, user preferences, and performance standards. The aim is to create models that generate outputs based on statistical accuracy and desired ethical, safety, and user-specific criteria.

### Types of Alignment Methods

- **RLHF**: Enhancing Language Models with Human Feedback
- **RLAIF**: Leveraging AI Feedback for Training
- **Direct Preference Optimization (DPO)**
- **Identity Preference Optimization (IPO)**
- **Kahneman-Tversky Optimization (KTO)**
- **Odds Ratio Preference Optimization (ORPO)**

---

### RLHF: Enhancing Language Models with Human Feedback

Reinforcement Learning from Human Feedback (RLHF) significantly improves the performance of Large Language Models (LLMs) compared to Supervised Fine-Tuning (SFT) alone. RLHF leverages human feedback to refine and optimize the model’s responses, ensuring they align better with human preferences and expectations.

#### The Value of Human Feedback

- **Complex Human Intuitions**: Human feedback excels in complex and difficult-to-formalize situations.
- **Flexible Dialogues**: Multiple plausible responses for any given prompt vary in quality.

#### Limitations of Demonstration Data

- **Plausibility vs. Quality**: Demonstration data shows plausible responses but not their quality.
- **Quality Evaluation**: RLHF uses a scoring function to evaluate response quality.

#### The RLHF Process

##### Training the Reward Model (RM)

- **Purpose**: The RM scores pairs of (prompt, response) based on quality.
- **Data Collection**: Gather comparison data where labelers decide which of two responses to the same prompt is better.
- **Objective**: Maximize the score difference between winning and losing responses.

##### Optimizing the LLM

- **Goal**: Train the LLM to generate responses that maximize the RM’s scores.
- **Method**: Use reinforcement learning algorithms like Proximal Policy Optimization (PPO).

##### Mathematical Framework

- **Data Format**: (prompt, winning_response, losing_response)
- $s_w = r_{\theta}(x, y_w)$: Reward score for the winning response
- $s_l = r_{\theta}(x, y_l)$: Reward score for the losing response

##### Loss Function

$$\text{Loss} = -\log(\sigma(s_w - s_l))$$

This function encourages the RM to give higher scores to winning responses.

##### Challenges and Solutions in Training the RM

- **Consistent Scoring**: Achieving consistent scoring among different labelers is challenging.
- **Comparison Data**: Use comparison data instead of absolute scores for easier and more reliable labeling.
- **SFT Model as Seed**: Starting RM training with an SFT model as the seed improves performance.
- **Powerful RM**: The RM must be at least as powerful as the LLM it scores.

##### Reinforcement Learning Fine-Tuning

- **KL Divergence**: Use KL divergence to penalize responses that differ significantly from the SFT model’s outputs.
- **Hallucination**: Addressing hallucination by verifying sources and developing better reward functions.

##### Effectiveness of RLHF

- **Overall Performance**: RLHF enhances performance and is generally preferred by human evaluators.
- **Human Feedback and Comparisons**: Improves the model’s ability to generate high-quality, contextually appropriate responses.

---

### RLAIF: Reinforcement Learning from AI Feedback

RLAIF leverages AI-generated feedback instead of human feedback to train large language models (LLMs), aiming to improve scalability, reduce bias, and ensure ethical model behavior.

#### Key Components

- **AI Feedback Agents**: Autonomous AI agents generate feedback on LLM responses, adhering to Constitutional AI principles.
- **Preference Model (PM)**: Evaluates response quality, trained on AI-generated feedback to provide stable training signals.

#### Training Process

- **Generating Harmlessness Dataset**: AI agents generate a dataset evaluated for harmlessness and helpfulness.
- **Fine-tuning SL-CAI Model**: SL-CAI model is fine-tuned using the harmlessness dataset.
- **Training Preference Model**: PM is trained using data from the fine-tuned SL-CAI model.
- **Reinforcement Learning (RL) with PPO**: PPO algorithms adjust the SL-CAI model's policy based on PM evaluations.

#### Advantages of RLAIF

- **Bias Reduction**: AI-generated feedback reduces biases inherent in human datasets.
- **Scalability**: Efficient data generation by AI agents enhances scalability.
- **Ethical and Safe Models**: Adherence to Constitutional AI principles ensures ethical model behavior.
- **Performance Improvement**: Iterative fine-tuning and RL enhance model performance and stability.

---

### Direct Preference Optimization (DPO)

Direct Preference Optimization (DPO) simplifies the training process by optimizing the log-likelihood difference between preferred and non-preferred outputs, bypassing the need for complex reward modeling.

#### Formulation of DPO

DPO fine-tunes a language model policy $\pi_\theta$ to generate outputs $y_w$ preferred over $y_l$, given input $x$. The optimization objective is:

$$\mathcal{L}(\theta) = \mathbb{E}\_{(x, y\_w, y\_l) \sim D} \left[ \log \pi\_\theta(y\_w \mid x) - \log \pi\_\theta(y\_l \mid x) \right]$$

#### Advantages of DPO

- **Simplicity**: Eliminates the need for intermediate reward modeling.
- **Efficiency**: Direct optimization allows for faster convergence and reduced computational overhead.
- **Robustness**: Less sensitive to hyperparameter settings.
- **Performance**: Achieves better or comparable performance to state-of-the-art RLHF methods.

---

### Identity Preference Optimization (IPO)

Identity Preference Optimization (IPO) improves the alignment of language models with human preferences by addressing overconfidence and policy degeneration through identity-based regularization.

#### Motivation

- **Overconfidence**: DPO may result in overconfident reward assignments.
- **Policy Degeneration**: Models may collapse, assigning near-zero probabilities to preferred responses.

#### Core Concept

IPO adds an identity-based regularization term to the optimization objective, preventing overconfidence and maintaining stable policies.

#### Implementation Steps

1. **Collect Preference Data**: Gather annotations where each input prompt $x$ has a preferred response $y_w$ and a non-preferred response $y_\ell$.
2. **Introduce Regularization**: Add an identity-based regularization term to the objective function.
3. **Formulate the Objective Function**:

$$\mathcal{L}*{\text{ipo}}(\pi*\theta; \mathcal{D}*{\text{pref}}) = \mathbb{E}*{(y_w, y_\ell, x) \sim \mathcal{D}*{\text{pref}}} \left[ - \log \sigma \left( \beta \log \frac{\pi*\theta(y_w)}{\pi_\theta(y_\ell)} \cdot \frac{\pi_{\text{ref}}(y_\ell)}{\pi_{\text{ref}}(y_w)} \right) \right] + \lambda \mathcal{R}(\pi_\theta)$$


Where $\sigma$ is the sigmoid function, $\mathcal{R}(\pi_\theta)$ is the regularization term, and $\lambda$ controls the regularization strength.

4. **Train the Model**: Use the modified objective function to ensure robust and stable policies.

#### Benefits of IPO

- **Enhanced Robustness**: Prevents the model from becoming overly confident.
- **Improved Stability**: Regularization maintains stability, preventing policy degeneration.
- **Better Generalization**: Avoids overfitting, improving the model's ability to handle new prompts.

---

### Kahneman-Tversky Optimization (KTO)

Kahneman-Tversky Optimization (KTO) aligns large language models (LLMs) with human feedback by leveraging principles from prospect theory, optimizing model outputs based on binary desirability signals.

#### Background and Motivation

**Prospect Theory**

Prospect theory provides insights into how individuals perceive and evaluate uncertain outcomes, emphasizing loss aversion and subjective biases in probability perception.

#### KTO Methodology

**Derivation of KTO**

KTO leverages the Kahneman-Tversky model to optimize model utility using binary desirability signals.

The KTO loss function is:

$$L_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = E_{x,y \sim D}[w(y)(1 - v_{\text{KTO}}(x,y; \beta))]$$

Where:
$$r_{\text{KTO}}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$
$$z_{\text{ref}} = E_{x' \sim D}[\beta \text{KL}(\pi_\theta(y'|x') \parallel \pi_{\text{ref}}(y'|x'))]$$
$$\sigma(r_{\text{KTO}}(x, y) - z_{\text{ref}})$$ for desirable outputs.
$$\sigma(z_{\text{ref}} - r_{\text{KTO}}(x, y))$$ for undesirable outputs.
$$w(y)$$ weights losses based on desirability ($\lambda_D$ for desirable, $\lambda_U$ for undesirable).

### Implementation Details

KTO estimates the KL divergence term by comparing outputs from the target model and a reference policy.

**Advantages of KTO**

- **Efficient Training**: Incorporates both desirable and undesirable outputs.
- **Aligned Preferences**: Effectively aligns model outputs with human preferences.

---

### Odds Ratio Preference Optimization (ORPO)

Odds Ratio Preference Optimization (ORPO) directly optimizes model outputs by maximizing the log-likelihood of preferred responses over non-preferred ones using the odds ratio principle.

#### Core Concept

ORPO simplifies the training process by focusing on the log-odds of preferred responses, directly reflecting user preferences.

#### Formulation of ORPO

The optimization objective for ORPO is:

$$\mathcal{L}(\theta) = \mathbb{E}\_{(x, y\_w, y\_l) \sim D} \left[ \log \frac{\pi\_\theta(y\_w \mid x)}{\pi\_\theta(y\_l \mid x)} \right]$$

#### Advantages of ORPO

- **Simplicity**: Directly reflects preference comparisons.
- **Efficiency**: Faster convergence and reduced computational overhead.
- **Robustness**: Less sensitive to hyperparameter settings.
- **Performance**: Achieves better or comparable performance to state-of-the-art RLHF methods.

---

### Comparison and Synergies

| Method       | Key Feature                                    | Advantages                             |
|--------------|------------------------------------------------|----------------------------------------|
| **RLHF**     | Human feedback-based                           | High alignment with human preferences  |
| **RLAIF**    | AI-generated feedback, Constitutional AI       | Scalability, ethical behavior          |
| **DPO**      | Direct log-likelihood difference optimization  | Simplicity, efficiency                 |
| **IPO**      | Identity-based regularization                  | Robustness, stability                  |
| **KTO**      | Prospect theory principles                     | Efficient training, aligned preferences|
| **ORPO**     | Log-odds ratio optimization                    | Simplicity, efficiency                 |

--- 

### Alignment Techniques Comparision

| Feature                  | RLHF                                              | RLAIF                                               | DPO                                                | IPO                                                   | KTO                                                  | ORPO                                                 |
|--------------------------|---------------------------------------------------|-----------------------------------------------------|----------------------------------------------------|-------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| **Core Technique**       | Human feedback-based reinforcement learning       | AI-generated feedback                               | Direct optimization of log-likelihood difference   | Identity-based regularization                         | Prospect theory principles                            | Odds ratio optimization                               |
| **Feedback Source**      | Human                                             | AI agents                                           | Human preferences                                  | Human preferences                                     | Human feedback, prospect theory                       | Human preferences                                     |
| **Training Process**     | Train reward model, use PPO for optimization      | AI agents generate feedback, train PM, use PPO      | Optimize log-likelihood difference                 | Regularize to prevent overconfidence                  | Use implicit reward and reference point               | Combine SFT loss with relative ratio loss             |
| **Advantages**           | Aligns with human preferences, improved response quality | Reduces bias, scalable, ethical                     | Simple, efficient, robust, good performance        | Prevents overconfidence, stable policies              | Scalable, practical, robust, handles data imbalance  | Streamlined, efficient, robust, high performance     |
| **Loss Function**        | -log(σ(s_w - s_l))                                | PPO with AI feedback                                | Log-likelihood difference                          | Log-likelihood difference with regularization         | Implicit reward from prospect theory                  | SFT loss + relative ratio loss                        |
| **Scalability**          | Moderate                                          | High                                                | Moderate                                           | Moderate                                              | High                                                 | High                                                 |
| **Implementation Complexity** | High                                       | High                                                | Low                                                | Moderate                                              | Moderate                                             | Moderate                                             |
| **Robustness**           | High                                              | High                                                | High                                               | High                                                  | High                                                 | High                                                 |


## References 
1. **A Survey of Large Language Models** by Wayne Xin Zhao et al. [:link:](https://arxiv.org/pdf/2303.18223)
2. **Attention Is All You Need** by Ashish Vaswani∗ et al. [:link:](https://arxiv.org/pdf/1706.03762)

