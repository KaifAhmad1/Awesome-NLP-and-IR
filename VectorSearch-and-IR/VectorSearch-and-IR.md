# Vector Search: A Comprehensive Overview
---
- **What is a Vector?:** In mathematics, a vector is a quantity defined by both magnitude and direction. Vectors are represented as arrays of numbers, which correspond to coordinates in a multidimensional space. They are foundational in various fields, including physics, engineering, and computer science.
   - Typically represented as $V = [v1, v2, v3, ...., vn]$ where $n$ is the magnitude of the vector in high dimensional space.
   - #### **Basic Properties of Vectors:**
     - **Magnitude:** The length of the vector.
     - **Direction:** The orientation of the vector in space.
  ![Vector Representation in High Dimensional Space](https://github.com/KaifAhmad1/Awesome-NLP-and-IR/blob/main/images/Vector%20Representation.png)
   ### Vector Representation in Machine Learning
    In machine learning and information retrieval, vectors are used to represent data points in a high-dimensional space. This representation is crucial for tasks like similarity search, where the goal is to find data points that are similar to a given query.
    - #### 1. **Text Data:**
       - **Word Embeddings:** Words are mapped to vectors using models like Word2Vec, GloVe, or FastText. These vectors capture semantic meanings, where similar words have similar vectors.
       - **Sentence Embeddings:**  Models like BERT and GPT transform entire sentences or documents into vectors, preserving contextual meaning.
     - #### 2. **Image Data:**
       - **Feature Vectors:** Convolutional Neural Networks (CNNs) are used to extract features from images, which are then represented as vectors.
     - #### 3. **Audio and Video Data:**
       - **Audio Vectors:** Deep learning models like VGGish convert audio signals into vectors that capture the essential characteristics of the sound.
       - **Video Vectors:** Similar to images, videos are processed frame by frame or using 3D CNNs to generate vectors representing the video content.
         
  ---
### Distance Metrics

Distance metrics are used to quantify the similarity or dissimilarity between vectors. Different metrics are suited for different types of data and applications.

#### 1. Dot Product Metric
Measures the similarity between two vectors by taking the dot product of the vectors.
$$v \cdot u = \sum_{i=1}^{n} v_i \cdot u_i$$
- For vectors $V = [1, 2]$ and $U = [3, 4]$, the dot product is $1 \cdot 3 + 2 \cdot 4 = 3 + 8 = 11$.
  ##### Advantages
    - Simple and fast to compute.
    - Useful in various applications, including machine learning and signal processing.
  ##### Limitations
    - Does not provide a bounded similarity score.
    - Can be influenced by the magnitudes of the vectors.

#### 2. Euclidean Distance
Measures the straight-line distance between two points in Euclidean space.
$$d(v, u) = \sqrt{\sum_{i=1}^{n} (v_i - u_i)^2}$$
- For vectors $V = [1, 2]$ and $U = [4, 6]$, the Euclidean distance is $√((4-1)^2 + (6-2)^2) = √(9 + 16) = √25 = 5$.
  ##### Advantages
    - Intuitive and easy to compute.
    - Well-suited for small, low-dimensional datasets.
  ##### Limitations
    - Sensitive to differences in magnitude and scaling.
    - Not suitable for high-dimensional spaces due to the curse of dimensionality, where distances become less meaningful.

#### 3. Manhattan Distance
Measures the distance between two points along axes at right angles, also known as L1 or taxicab distance.
$$d(v, u) = \sum_{i=1}^{n} |v_i - u_i|$$
- For vectors $V = [1, 2]$ and $U = [4, 6]$, the Manhattan distance is $|4-1| + |6-2| = 3 + 4 = 7$.
  ##### Advantages
    - Robust to outliers and useful in grid-based pathfinding problems, such as robotics and game design.
  ##### Limitations
    - Can be less intuitive for non-grid-based data.
    - Sensitive to scale, like Euclidean distance.

#### 4. Cosine Similarity
Measures the cosine of the angle between two vectors, indicating their similarity in terms of direction rather than magnitude.
$$\cos(\theta) = \frac{v \cdot u}{\|v\| \|u\|}$$
- For vectors $V = [1, 2]$ and $U = [2, 3]$, the cosine similarity is 
$$\cos(\theta) = \frac{1 \cdot 2 + 2 \cdot 3}{\sqrt{1^2 + 2^2} \cdot \sqrt{2^2 + 3^2}} = \frac{8}{\sqrt{5} \cdot \sqrt{13}} \approx 0.98$$
  ##### Advantages
    - Useful for high-dimensional data, such as text data represented as word vectors.
    - Ignores magnitude, focusing on the direction of the vectors.
  ##### Limitations
    - Ignores magnitude, which can be a drawback if magnitude differences are important.
    - Requires non-zero vectors to compute.

#### 5. Jaccard Similarity
Measures the similarity between finite sets by considering the size of the intersection divided by the size of the union of the sets.
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
- For sets $A = \{1, 2, 3\}$ and $B = \{2, 3, 4\}$, the Jaccard similarity is 
$$J(A, B) = \frac{| \{2, 3\} |}{| \{1, 2, 3, 4\} |} = \frac{2}{4} = 0.5$$
  ##### Advantages
    - Handles binary or categorical data well.
    - Simple interpretation and calculation.
  ##### Limitations
    - Not suitable for continuous data.
    - Can be less informative for datasets with many common elements.

#### 6. Hamming Distance
Measures the number of positions at which the corresponding elements of two binary vectors are different.
$$H(v, u) = \sum (v_i \neq u_i)$$
- For binary vectors $V = [1, 0, 1]$ and $U = [0, 1, 1]$, the Hamming distance is $(1 ≠ 0) + (0 ≠ 1) + (1 = 1) = 2$.
  ##### Advantages
    - Effective for error detection and correction in binary data.
    - Simple and fast to compute.
  ##### Limitations
    - Only applicable to binary vectors.
    - Not useful for continuous or non-binary categorical data.

#### 7. Earth Mover's Distance (EMD)
Measures the minimum amount of 'work' needed to transform one distribution into another, often used in image retrieval. Also known as the Wasserstein distance.
$$EMD(P, Q) = \inf_{\gamma} \int_{X \times Y} d(x,y) \, d\gamma(x,y)$$
- Given two distributions of points, EMD calculates the cost of moving distributions to match each other. For instance, if distribution $𝑃$ has points $[1, 2]$ and $𝑄$ has points $[2, 3]$, EMD would calculate the minimal transportation cost.
  ##### Advantages
    - Provides a meaningful metric for comparing distributions, taking into account the underlying geometry.
    - Applicable to various types of data, including images and histograms.
  ##### Limitations
    - Computationally intensive, especially for large datasets.
    - Requires solving an optimization problem, which can be complex.

---

### Vector Search Techniques
  Vector search involves finding vectors in a database that are similar to a given query vector. Techniques include:
   - #### 1. **Brute-Force Search:**
      - Computes similarity between the query vector and all vectors in the dataset.
      - Inefficient for large datasets due to high computational cost.
   - #### 2. **k-Nearest Neighbors (k-NN):**
      - Finds the k vectors closest to the query vector.
      - Can be implemented using efficient data structures like KD-Trees or Ball Trees for lower-dimensional data.
   - #### 3. **Approximate Nearest Neighbor (ANN):**
      - Speeds up search by approximating the nearest neighbours.
      - Methods include Locality-Sensitive Hashing (LSH) and Hierarchical Navigable Small World (HNSW) graphs.
- ### **Applications of Vector Search:**
  Vector search is transforming various industries by enabling more accurate and context-aware search functionalities:
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


--- 

### Nearest Neighbor Search

Nearest neighbor search is a fundamental technique used to identify the closest data points to a given query point within a dataset. It is essential in various applications such as recommendation systems, image and video retrieval, and machine learning classification tasks.

- **Example:** In a recommendation system, nearest neighbor search helps find users with similar preferences, enabling the system to suggest products or services that align with a user's tastes. For instance, Netflix recommends movies by identifying viewers with similar viewing habits and suggesting what others with similar preferences have enjoyed.

### High-Dimensional Data

High-dimensional data refers to datasets with a large number of features or dimensions, such as text data represented by word embeddings or image data characterized by pixel values. Analyzing and managing high-dimensional data presents several challenges:

1. **Increased Computational Complexity:** The number of calculations required increases exponentially with the number of dimensions, leading to significant computational costs.
2. **Data Sparsity:** As dimensions increase, data points become sparse, making it difficult to draw meaningful comparisons.
3. **Overfitting:** With a large number of features, models may capture noise rather than underlying patterns, resulting in overfitting.

- **For Example:** In image search, each image can be represented as a high-dimensional vector. Comparing these vectors directly is computationally intensive due to the vast number of dimensions involved.

### Curse of Dimensionality

The curse of dimensionality, a term coined by `Richard Bellman`, describes the various phenomena that arise when analyzing data in high-dimensional spaces. As the number of dimensions increases:

1. **Distance Measures Become Less Meaningful:** In high-dimensional spaces, the distance between data points becomes more uniform, making it difficult to differentiate between the nearest and farthest neighbours.
2. **Volume of Space Increases Exponentially:** The volume of the space grows exponentially with the number of dimensions, causing data points to become sparse and reducing statistical significance.
3. **Increased Noise and Redundancy:** Higher dimensions can introduce more noise and redundant information, complicating the learning process and degrading the performance of algorithms.

- **Example:** Consider a facial recognition system operating in high-dimensional space. The Euclidean distance between facial vectors becomes less effective, necessitating more advanced techniques to accurately measure similarity. This phenomenon illustrates the need for innovative solutions to manage high-dimensional data efficiently.

---

### Linear Search

Linear search is a straightforward method for finding a specific element in a vector (or array) by checking each element sequentially until the desired element is found or the end of the vector is reached. It operates in a vector space, which is essentially a one-dimensional array of elements.

- **Mathematical Explanation:** Given a vector $V = [v_1, v_2, \ldots, v_n]$ and a target element $t$, the linear search algorithm checks each element $v_i$ in $V$ sequentially:
  1. Start from the first element: $i = 1$
  2. Compare $t$ with $v_i$.
  3. If $t = v_i$, the search is successful, and the position $i$ is returned.
  4. If $t \neq v_i$, increment $i$ and repeat steps 2-3 until $i = n$ or $t$ is found.

- **Time Complexity:** Linear $O(n)$
- **Space Complexity:** Constant $O(1)$

#### Advantages:

1. Linear search is straightforward to implement and understand.
2. Linear search does not require the dataset to be sorted or preprocessed in any way.
3. Linear search can be used on any type of dataset, regardless of structure or order.

#### Limitations:

1. Linear search is inefficient for large datasets because it requires checking each element sequentially.
2. For large datasets, linear search can be very slow compared to other search algorithms like KNN search or hash-based searches.

```python
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
--- 

### Dimensionality Reduction
   - Dimensionality reduction is a fundamental technique in data analysis and machine learning, aimed at transforming high-dimensional data into a lower-dimensional representation while preserving its essential characteristics. This process offers several advantages, including enhanced computational efficiency, improved model performance, and better visualization of complex datasets.
   - Reducing dimensions helps address the Curse of Dimensionality by making data more interpretable and patterns more discernible. It also boosts computational efficiency by reducing complexity, leading to faster algorithms. Furthermore, it improves model performance by focusing on relevant features and mitigating overfitting.
   - Dimensionality reduction techniques like PCA and t-SNE facilitate data visualization by projecting high-dimensional data into lower-dimensional spaces, making complex relationships easier to understand.
  
### Principal Component Analysis
PCA is a widely used technique for linear dimensionality reduction. It aims to find the directions, or principal components, in which the data varies the most and projects the data onto these components to obtain a lower-dimensional representation.
  - At its core, PCA seeks to transform high-dimensional data into a lower-dimensional form while preserving the most important information. It achieves this by identifying the directions in which the data varies the most, known as the principal components, and projecting the data onto these components.
#### Mathematical Foundation
   - **Centering the Data:** PCA begins by centering the data, which involves subtracting the mean vector $Xmean$ from each sample.
   - **Covariance Matrix:** Next, it computes the covariance matrix $C$ of the centered data. This matrix quantifies the relationships between different features and how they vary together.
   - **Eigen Decomposition:** PCA then proceeds to compute the eigenvectors and eigenvalues of the covariance matrix. These eigenvectors represent the principal components, and the corresponding eigenvalues indicate the amount of variance explained by each component.

#### Steps in PCA
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

### t-Distributed Stochastic Neighbor Embedding (t-SNE)
t-SNE is a machine learning algorithm primarily used for dimensionality reduction and visualizing high-dimensional data. It is a non-linear technique particularly well-suited for embedding high-dimensional data into a low-dimensional space (typically 2D or 3D) while aiming to preserve the local structure and similarities within the data. Developed by Geoffrey Hinton and Laurens van der Maaten in 2008, t-SNE has gained immense popularity due to its ability to produce high-quality visualizations and uncover hidden patterns and clusters in complex datasets.
 #### Key Concepts
   #### 1. Dimensionality Reduction
   This means reducing the number of variables in the data. t-SNE reduces data from high-dimensional space to a 2D or 3D space, making it easier to plot and visually inspect.
   #### 2. Stochastic Neighbor Embedding
   This idea models the probability distribution of pairs of high-dimensional objects. Nearby points in high-dimensional space remain close in the low-dimensional space, and distant points stay far apart.
   #### 3. t-Distribution
   Unlike linear techniques like PCA (Principal Component Analysis), t-SNE is non-linear. It uses a heavy-tailed t-distribution in the low-dimensional space to prevent points from clumping together.

#### How t-SNE Works
   #### 1. Pairwise Similarities
   t-SNE starts by calculating how similar each pair of points is in the high-dimensional space. It measures the Euclidean distance between points and converts these distances into probabilities that represent similarities.

  The similarity $p_{ij}$ between two points $x_i$ and $x_j$ is calculated as:
  $$p_{ij} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$
  Here, $\sigma_i$ is the variance of the Gaussian distribution centered at $x_i$.
  
  #### 2. Joint Probabilities
   These probabilities are symmetrical to ensure that the similarity between point $A$ and point $B$ is the same as between point B and point A.

  The joint probability $P_{ij}$ is:
  $$P_{ij} = \frac{p_{ij} + p_{ji}}{2N}$$
  Here, $N$ is the number of data points.

  #### 3. Low-Dimensional Mapping
 Points are initially placed randomly in a low-dimensional space. t-SNE then adjusts their positions to minimize the difference between the high-dimensional and low-dimensional similarities.

 #### 4. Gradient Descent
 Positions are adjusted using an optimization method called gradient descent. This minimizes the KL divergence between the two probability distributions (high-dimensional and low-dimensional).
  The Kullback-Leibler divergence $KL(P \parallel Q)$ is:
  $$KL(P \parallel Q) = \sum_{i \neq j} P_{ij} \log\left(\frac{P_{ij}}{Q_{ij}}\right)$$

  Here, $Q_{ij}$ is the similarity between points $y_i$ and $y_j$ in the low-dimensional space, calculated as:
  $$Q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

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

--- 

- ### **Approximate Nearest Neighbor (ANN) Search:**
  Approximate Nearest Neighbor (ANN) search is a technique used to find points in a high-dimensional space that are approximately closest to a given query point. This method is particularly crucial when dealing with large datasets where exact nearest neighbor search becomes computationally infeasible. ANN search balances between accuracy and computational efficiency, making it an invaluable tool in various fields such as machine learning, data mining, and information retrieval.
   - #### 1. **ANN Search in Machine Learning** ANN search is crucial for high-dimensional data tasks, such as:
     - **Feature Matching in Computer Vision:** Identifies similar features across images for tasks like image stitching, object recognition, and 3D reconstruction.
     - **Recommendation Systems:** Recommends items by identifying similar users or items based on behavior or attributes represented as vectors.
     - **Clustering:** Accelerates clustering large datasets by quickly finding approximate clusters, which can then be refined.
   - #### 2. **ANN Search in Data Mining** In data mining, ANN search enhances:
     - **Efficient Data Retrieval:** Quickly finds relevant data points similar to a query, essential for applications like anomaly detection.
     - **Pattern Recognition:** Identifies patterns or associations within large datasets, aiding in market basket analysis and customer segmentation.
   - #### 3. **ANN Search in Information Retrieval** Information retrieval systems use ANN search for:
     - **Semantic Search:** Retrieves documents or information semantically similar to a user's query by representing text data as vectors.
     - **Multimedia Retrieval:** Finds similar images, videos, or audio files based on content rather than metadata, using high-dimensional vectors.

- ### **Trade-Off Between Accuracy and Efficiency in ANN Search**
  In Approximate Nearest Neighbor (ANN) search, balancing accuracy and efficiency is crucial, especially for large-scale and high-dimensional datasets. While the aim is to quickly find the nearest neighbors with high precision, achieving both accuracy and speed is challenging due to computational constraints.
  - **Accuracy vs. Efficiency**
    - **Accuracy:** Ensures the search results closely match the exact nearest neighbors. High accuracy is vital for tasks requiring precise similarity measures but demands extensive computations, making it resource-intensive and slow.
    - **Efficiency:** Focuses on the speed and resource usage of the search. Efficient algorithms deliver quick results and use minimal memory, but they may sacrifice some accuracy by employing approximations and heuristics.
- ### **Importance of Faster Search Methods**
  - #### 1 **Large-Scale Datasets**
     - **Real-Time Processing:** In applications like online search engines, recommendation systems, and real-time analytics, delivering results almost instantaneously is crucial. Efficient ANN search methods enable these systems to provide timely and relevant results without delays.
     - **Scalability:** As datasets grow, the computational burden increases exponentially. Efficient ANN search algorithms ensure the system can handle this growth without a proportional rise in resource requirements, maintaining performance and responsiveness.
  - #### 2 **High-Dimensional Data:**
     - **Reduced Computational Complexity:** Techniques that reduce the number of dimensions or approximate distances help manage the computational load, making it feasible to process high-dimensional data effectively. This is crucial in fields like image and video processing, natural language processing, and genomics.
     - **Handling Sparsity:** High-dimensional spaces often lead to sparse data distributions. Efficient ANN search methods are designed to navigate this sparsity, finding relevant neighbors without exhaustive searches.

- ### **Techniques to Balance Accuracy and Efficiency**
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

--- 

### **Flat Indexing:**
Flat indexing, also referred to as brute-force or exhaustive indexing, entails storing all dataset vectors within a single index structure, typically an array or list. Each vector is assigned a unique identifier or index within this structure. Upon receiving a query vector, the index is sequentially traversed, and the similarity between the query and each dataset vector is computed. This iterative process continues until all vectors are assessed, ultimately identifying the closest matches to the query.
  - #### **How it works:**
     - **Index Construction:** Initially, all dataset vectors are stored in memory or on disk to construct the index.
       -  All dataset vectors $X = \{x_1, x_2, \ldots, x_n\}$ are stored in memory or on disk.
     - **Query Processing:** Upon receiving a query vector, the system systematically compares it with every vector in the index, computing the similarity or distance metric (e.g., Euclidean distance, cosine similarity) between the query and each vector.
     - **Ranking:** As comparisons progress, vectors are ranked based on their similarity to the query, thereby pinpointing the closest matches.
     - **Retrieval:** After evaluating all vectors, the system retrieves either the $top-k$ closest matches or all vectors meeting a specified similarity threshold.

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
--- 

### **Inverted Index** 
An Inverted Index is a data structure used primarily in information retrieval systems, such as search engines, to efficiently map content to its location in a database, document, or set of documents. It enables quick full-text searches by maintaining a mapping from content terms to their occurrences in the dataset.
   - #### **How It Works**
     - **Tokenization:** The process starts with tokenizing the text data. Tokenization involves breaking down text into individual tokens, typically words or terms.
     - **Normalization:** Tokens are often normalized, which may include converting to lowercase, removing punctuation, and applying stemming or lemmatization to reduce words to their base forms.
     - **Index Construction:** Each unique token is stored in the index, along with a list of documents or positions where it appears. This mapping allows for efficient look-up during search queries.
     - **Posting List:** Each token in the index has an associated posting list, which is a list of all documents and positions where the token appears.

   - Example Consider three documents
     - $Document 1:$ `apple banana fruit`
     - $Document 2:$ `banana apple juice`
     - $Document 3:$ `fruit apple orange`
   - The inverted index for these documents would look like this:
     - apple: $[1, 2, 3]$
     - banana: $[1, 2]$
     - fruit: $[1, 3]$
     - juice: $[2]$
     - orange: $[3]$
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

--- 

### **Locality-Sensitive Hashing (LSH)** 
Locality-sensitive hashing (LSH) is a technique used to efficiently find approximate nearest neighbors in high-dimensional data. This method is particularly useful when dealing with large datasets where the exact nearest neighbor search would be too slow. LSH aims to hash similar items into the same buckets with high probability, which makes searching faster.
  - #### **Key Concepts**
    - **Locality Preservation:** LSH ensures that items that are close to each other in high-dimensional space are likely to be in the same bucket after hashing.
    - **Hash Function Family:** LSH uses a set of hash functions $\mathcal{H}$ that have a high probability of assigning similar items to the same bucket and a low probability of assigning dissimilar items to the same bucket.
    - **Approximation:** LSH provides approximate results, which means it finds neighbors that are close enough rather than the exact nearest neighbours.

  - #### **How LSH Works**
    - 1. **Hash Function Selection:** Choose or design hash functions that are locality-sensitive to the chosen similarity metric.
    - 2. **Index Construction:** Apply the hash functions to all items in the dataset, distributing them into buckets.
    - 3. **Query Processing:**
       - Hash the query item using the same hash functions.
       - Retrieve and compare items from the corresponding bucket(s).
       - Use a secondary, more precise similarity measure to rank the retrieved items and find the approximate nearest neighbours.

  - #### **Mathematics of LSH**
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

---

## Quantization

Quantization is a crucial technique in Approximate Nearest Neighbor (ANN) search, particularly when dealing with large and high-dimensional datasets. By approximating data points with a limited set of representative points (centroids), quantization reduces storage requirements and computational complexity, facilitating faster and more efficient similarity searches.

### Key Concepts in Quantization

- **Quantization:** The process of mapping high-dimensional vectors to a finite set of representative points, thereby reducing data complexity.
- **Centroids/Codewords:** Representative points used in the quantization process. Each data point is approximated by the nearest centroid.
  - Consider approximating the value of $\pi$ (pi), which is approximately $3.14159$. Let's use a simple codebook with two centroids: $C_1 = 3.0$ and $C_2 = 3.2$.
    - The nearest centroid to $\pi$ ($3.14159$) is $C_2$ ($3.2$) since $|3.14159 - 3.2| = 0.05841$ is less than $|3.14159 - 3.0| = 0.14159$.
    - Therefore, $\pi$ is approximated by $C_2$.
- **Codebook:** A collection of centroids that are used to approximate the original data points.
- **Quantization Error:** The difference between the original data point and its quantized approximation. Lower quantization error implies higher accuracy in search results.
  - The quantization error is the squared difference between $\pi$ and the centroid $C_2$:
    - Quantization Error = $(3.14159 - 3.2)^2 = (-0.05841)^2 \approx 0.00341$.

Quantization helps manage large datasets by simplifying data representation, which in turn speeds up the process of finding similar data points through approximate nearest neighbor search techniques.

### Types of Quantization

#### 1. Scalar Quantization

Scalar quantization is a technique where each component of a vector is quantized independently, simplifying the data representation process by breaking down the high-dimensional problem into individual dimensions.

- **Example:** Suppose we have a dataset of 2D points $(x, y)$, and we want to quantize each dimension independently. Let's consider quantizing $x$ and $y$ into three levels: $\{1.0, 2.0, 3.0\}$ for $x$ and $\{4.0, 5.0, 6.0\}$ for $y$.
  - **Quantization Process:** Given a point $(2.3, 4.7)$, we quantize $x$ to the nearest level, which is $2.0$, and $y$ to $5.0$. So, the quantized point becomes $(2.0, 5.0)$.
  - **Quantization Error:** To compute the error, we take the sum of squared differences between the original and quantized values:
    - Quantization Error = $(2.3 - 2.0)^2 + (4.7 - 5.0)^2 = 0.09 + 0.09 = 0.18$.

#### 2. Vector Quantization

This technique quantizes the entire vector as a whole rather than its individual components, capturing the correlations between different dimensions of the vector. The data points are mapped to the nearest centroid in a set of predefined centroids (codebook) based on the overall similarity.

- **Example:** Consider the same 2D dataset, but this time, we want to quantize the entire vector as a single entity. Let's have centroids $\{(1.0, 2.0), (3.0, 4.0)\}$.
  - **Quantization Process:** For the point $(2.3, 4.7)$, we find the nearest centroid, which is $(3.0, 4.0)$. Thus, the quantized point becomes $(3.0, 4.0)$.
  - **Quantization Error:** The error is computed as the squared Euclidean distance between the original and quantized vectors:
    - Quantization Error = $(2.3 - 3.0)^2 + (4.7 - 4.0)^2 = 0.49 + 0.49 = 0.98$.

#### 3. Product Quantization

Product quantization is an advanced technique designed to handle very large and high-dimensional datasets efficiently by decomposing the original space into lower-dimensional subspaces.

##### Process

- **Decomposition:** Divide the high-dimensional vector into smaller, non-overlapping sub-vectors.
- **Independent Quantization:** Quantize each sub-vector independently using its own set of centroids.
- **Complexity Reduction:** Break down the high-dimensional quantization problem into several lower-dimensional problems.
- **Centroid Assignment:**
  - Assign each sub-vector a centroid from a sub-codebook.
  - Combine these centroids to represent the original vector.

##### Example

Suppose we have a 4D vector $(x_1, x_2, x_3, x_4)$ and want to perform product quantization by splitting it into two 2D sub-vectors: $(x_1, x_2)$ and $(x_3, x_4)$. Let's use centroids $\{(1.0, 2.0), (3.0, 4.0)\}$ for each sub-vector.

- **Quantization Process:** For the vector $(1.1, 2.2, 3.1, 3.9)$, the sub-vector $(x_1, x_2)$ is closest to $(1.0, 2.0)$, and $(x_3, x_4)$ is closest to $(3.0, 4.0)$. So, the quantized vector becomes $(1.0, 2.0, 3.0, 4.0)$.
- **Quantization Error:** The total error is the sum of errors from quantizing each sub-vector:
  - Quantization Error = $(0.1)^2 + (0.2)^2 + (0.1)^2 + (0.1)^2 = 0.01 + 0.04 + 0.01 + 0.01 = 0.07$.

```python
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

--- 


### **Tree-Based Indexing in ANN:** 
Tree-based indexing techniques are critical for efficiently managing and querying high-dimensional data. These structures organize data points hierarchically, allowing quick search and retrieval operations. The primary types of tree-based indexing methods used in Approximate Nearest Neighbor (ANN) search include K-D Tree, Ball Tree, and R-Tree. Each of these trees has unique characteristics and applications, as detailed below.
  - #### 1. **K-D Tree (K-Dimensional Tree)**
    A K-D Tree is a binary tree that organizes points in a k-dimensional space. It is particularly effective for low-dimensional data but can suffer from inefficiencies as the dimensionality increases.
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
  - #### 2. **Ball Tree**
    A Ball Tree is a binary tree that organizes points in a k-dimensional space by enclosing them in hyperspheres (balls). It is well-suited for high-dimensional data and can handle non-Euclidean distance metrics effectively.
   - #### **Construction**
     - **Ball Creation:** At each level of the tree, data points are grouped into two subsets, each enclosed within a ball. The split is made by selecting a point (usually the centroid) and partitioning the points based on their distances to this point.
     - **Recursive Partitioning:** The process is recursively applied to the subsets, creating a hierarchical structure of balls.
   - #### **Query Processing**
     - **Traversal:** For a given query point, the tree is traversed from the root, checking if the query point lies within a ball.
     - **Pruning:** Branches of the tree are pruned if it's determined that they cannot contain the nearest neighbor, based on the distance to the query point and the radii of the balls.
   - **Balancing** Similar to K-D Trees, Ball Trees require balancing to ensure efficient query times. This involves careful selection of partitioning points to keep the tree's depth minimal.
   - Advantages:
       - 1. EEfficient for higher-dimensional spaces.
       - 2. Handles non-Euclidean distance metrics (e.g., Mahalanobis, Manhattan).
       - 3. Provides exact nearest neighbor search with pruning to speed up queries.
   - Limitations:
      - 1. Construction can be more complex and computationally intensive compared to K-D Trees.
      - 2. May require more memory due to the need to store ball boundaries.
      - 3. Performance can still degrade in very high-dimensional spaces.
``` Python
import numpy as np
from sklearn.neighbors import BallTree

# Sample data
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Build BallTree
tree = BallTree(data, leaf_size=2)

# Query point
query = np.array([2, 3])

# Find nearest neighbor
dist, ind = tree.query([query], k=1)

print("Query point:", query)
print("Nearest neighbor index:", ind[0][0])
print("Nearest neighbor point:", data[ind[0][0]])
print("Distance to nearest neighbor:", dist[0][0])
```

--- 

### **Random Projection in ANN** 
Random projection is a dimensionality reduction technique used to approximate the distances between points in high-dimensional space. It is especially useful in Approximate Nearest Neighbor (ANN) search, where finding exact nearest neighbors in high-dimensional datasets can be computationally infeasible. By reducing the dimensionality while preserving the relative distances, random projection balances accuracy and efficiency.
  - #### **Key Concepts**
     - **Dimensionality Reduction** Random projection reduces the number of dimensions of the dataset, which helps in managing the computational complexity and improving the efficiency of ANN search.
     - **Johnson-Lindenstrauss Lemma** The Johnson-Lindenstrauss lemma is a foundational result that guarantees that a set of points in high-dimensional space can be embedded into a lower-dimensional space such that the distances between the points are approximately preserved.
  - #### **How it works**
     - **Generate Random Projections:**
        - Create a random projection matrix $𝑅$ with dimensions $𝑘×𝑑$ where $𝑑$ is the original dimensionality, and $𝑘$ is the reduced dimensionality.
        - Each element of $𝑅$ is typically drawn from a Gaussian distribution (mean $0$ and variance $1/k$) or a simpler distribution like ${−1,1}$
     - **Project Data:**
        - Multiply the original data matrix $X$ (with dimensions $n×d$) by the projection matrix $𝑅$ to obtain a new matrix $𝑋′$ (with dimensions $n×k$).
        - The transformed data $𝑋′$ now resides in a lower-dimensional space.
     - **Approximate Nearest Neighbor Search:**
        - Use the reduced-dimensional data $𝑋′$ for efficient ANN search.
        - Perform distance calculations and similarity measures in the lower-dimensional space, which is computationally less expensive than in the original high-dimensional space.
   - Advantages:
       - 1. Reduces the computational load by working in a lower-dimensional space.
       - 2. Easy to implement and does not require complex algorithms or data structures.
       - 3. Suitable for large datasets where exact nearest neighbor search is impractical.
   - Limitations:
      - 1. Provides approximate results, which may not be as precise as exact nearest neighbor searches.
      - 2. Performance depends on the choice of reduced dimensionality $𝑘$
``` Python
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import euclidean_distances

data = np.array([
    [1, 2, 3, 4],
    [4, 5, 6, 7],
    [7, 8, 9, 10]
])

# Perform random projection to reduce dimensionality
n_components = 2 
projector = GaussianRandomProjection(n_components=n_components)
reduced_data = projector.fit_transform(data)
print("Original Data:\n", data)
print("Reduced Data:\n", reduced_data)

# Query point (high-dimensional)
query = np.array([1, 1, 1, 1])
reduced_query = projector.transform(query.reshape(1, -1))
print("Reduced Query Point:\n", reduced_query)

distances = euclidean_distances(reduced_query, reduced_data)
nearest_neighbor_index = np.argmin(distances)

# Retrieve the nearest neighbor in the original space
nearest_neighbor = data[nearest_neighbor_index]
print("Nearest Neighbor Index:", nearest_neighbor_index)
print("Nearest Neighbor:", nearest_neighbor)
```
```
Original Data:
 [[ 1  2  3  4]
 [ 4  5  6  7]
 [ 7  8  9 10]]
Reduced Data:
 [[  0.87118251  -1.46763879]
 [  3.73753765  -6.17643136]
 [  6.60389279 -10.88522393]]
Reduced Query Point:
 [[ 0.95545171 -1.56959752]]
Nearest Neighbor Index: 0
Nearest Neighbor: [1 2 3 4]
```

--- 

### **Graph-based Indexing for ANN Search:** 
Graph-based indexing is an advanced technique for performing Approximate Nearest Neighbor (ANN) searches in high-dimensional spaces. This approach constructs a graph where data points are nodes, and edges represent the proximity or similarity between these points. Graph-based indexing efficiently addresses the limitations of traditional methods, such as tree-based or hashing methods, especially in handling high-dimensional data.
  - #### **Key Concepts**
     - **Graph Construction:** Nodes (data points) are connected based on proximity, forming a graph that represents the dataset's structure.
     - **Search Efficiency:** Graph-based methods use the graph's connectivity to navigate quickly to the nearest neighbors, reducing the number of distance computations.
     - **Scalability:** These methods handle large datasets effectively, providing a balance between accuracy and computational efficiency.
  - #### **Advantages**
     - **High Efficiency:** Fast query responses due to optimized traversal of the graph.
     - **Scalability:** Capable of handling large datasets and high-dimensional spaces.
     - **Accuracy:** Often provides better approximations compared to other methods, striking a balance between speed and accuracy.
  - #### 1. **HNSW**
    Hierarchical Navigable Small World (HNSW) is an advanced graph-based algorithm for Approximate Nearest Neighbor (ANN) search, designed to handle high-dimensional and large-scale datasets efficiently. HNSW constructs a multi-layered graph, where each layer contains a subset of nodes connected in a small-world network structure, allowing fast and accurate nearest neighbor searches.
    - #### **Key Concepts**
      - **Small-World Network:** HNSW leverages the properties of small-world networks, where most nodes can be reached from every other by a small number of steps. This significantly reduces search time.
      - **Hierarchical Structure:** The algorithm builds multiple layers of graphs. The top layers are sparsely connected and contain fewer nodes, while the bottom layers are densely connected and contain more nodes.
      - **Efficient Search:** The hierarchical nature allows the algorithm to quickly narrow down the search area by starting from the top layers and progressively moving to the bottom layers, which are more detailed.
    - #### **How it works**
      - #### 1. **Graph Construction:**
         - Layer Creation: Nodes are inserted into multiple layers of the graph. Higher layers contain fewer nodes and serve as navigational shortcuts.
         - Connection Strategy: Each node is connected to a fixed number of nearest neighbors, ensuring a balance between search efficiency and graph sparsity.
      - #### 2. **Query Processing:**
         - Top-Down Search: The search starts at the topmost layer, where the graph is sparse, making it easy to find a rough approximation of the nearest neighbor.
         - Navigating Down Layers: The search continues to the lower layers, where the graph becomes denser, refining the nearest neighbor approximation with each step.
         - Greedy Search: Within each layer, a greedy search algorithm moves from the current node to the closest node in the direction of the query, ensuring a fast traversal.
   - **Advantages:**
       - 1. **High Efficiency:** HNSW provides near real-time query responses even for large and high-dimensional datasets, thanks to its hierarchical structure and small-world properties.
       - 2. **Scalability:** The algorithm scales well with the size of the dataset, maintaining efficient performance as data grows.
       - 3. **High Accuracy:** HNSW achieves high accuracy in ANN search, often comparable to exact methods, but with significantly reduced computational overhead.
   - **Comparison to Other Methods**
       - **Flat Indexing:** HNSW outperforms flat indexing in terms of both speed and memory efficiency, particularly for large and high-dimensional datasets.
       - **Tree-Based Indexing (e.g., K-D Tree, Ball Tree):** HNSW is more efficient in high-dimensional spaces where tree-based methods suffer from the curse of dimensionality.
       - **Locality-Sensitive Hashing (LSH):** While LSH also provides efficient ANN search, HNSW generally offers better accuracy and scalability.
``` Python 
import hnswlib
import numpy as np
data = np.random.rand(10000, 128).astype(np.float32)

# Initialize the index
dim = data.shape[1]
num_elements = data.shape[0]
hnsw_index = hnswlib.Index(space='l2', dim=dim)

hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
hnsw_index.add_items(data)

# Set the query parameters
hnsw_index.set_ef(50)  
query_vector = np.random.rand(1, 128).astype(np.float32)
labels, distances = hnsw_index.knn_query(query_vector, k=10)

print("Nearest neighbors:", labels)
print("Distances:", distances)
```
```
Nearest neighbors: [[2588 7881 4298 1877 5270 6926 4005 1598 3829 9343]]
Distances: [[14.286862 14.975046 15.01863  15.024027 15.081305 15.204177 15.286852
  15.358801 15.378472 15.497097]]
```
  - #### 2. **Vamana Indexing**
    Vamana is a graph-based algorithm designed for efficient Approximate Nearest Neighbor (ANN) search, particularly in high-dimensional spaces. It constructs a navigable small-world graph to enable quick and accurate searches, making it suitable for large-scale datasets.
    - #### **Key Concepts**
      - **Navigable Small-World Graph:** Vamana leverages small-world properties where nodes are connected in such a way that any node can be reached from any other node within a few steps.
      - **Greedy Search:** Utilizes a greedy search approach to traverse the graph, efficiently finding the nearest neighbors by following the closest links at each step.
      - **Graph Construction:** Builds a graph incrementally, ensuring each node is optimally connected to maintain a balance between search efficiency and graph sparsity.
    - #### **How it works**
      - #### 1. **Graph Construction:**
          - Node Insertion: Nodes are added to the graph one by one, each being connected to its nearest neighbors.
          - Randomized Greedy Algorithm: When adding a new node, Vamana uses a randomized greedy algorithm to select connections, ensuring a diverse set of links that improve search efficiency.
          - Graph Pruning: Excessive edges are pruned to maintain a manageable number of connections per node, optimizing both storage and query speed.
      - #### 2. **Query Processing:**
         - Initial Search: Begins at a random or designated entry point in the graph.
         - Greedy Forward Search: Moves through the graph using a greedy approach, always moving to the nearest neighbor closer to the query point.
         - Refinement: Continues until the nearest neighbors are found or no closer nodes can be reached, ensuring an efficient yet thorough search process.
   - **Advantages:**
       - **High Efficiency:** Vamana provides fast query responses due to its optimized graph structure and greedy search mechanism.
       - **Scalability:** Handles large and high-dimensional datasets effectively, maintaining performance as data scales.
       - **High Accuracy:** Delivers accurate nearest neighbor approximations, often comparable to exact methods with significantly less computational overhead.
   - **Comparison to Other Methods**
       - **HNSW:** Vamana is similar to HNSW in leveraging small-world properties but focuses more on maintaining a balanced connection strategy for efficient search.
       - **Tree-Based Methods:** Outperforms tree-based methods like KD-Tree and Ball Tree in high-dimensional spaces, avoiding issues like the curse of dimensionality.
       - **Locality-Sensitive Hashing (LSH):** Offers better accuracy and scalability compared to LSH, with a more structured approach to graph construction and search.
     
--- 

### LSH Forest
Locality-Sensitive Hashing (LSH) Forest is an advanced technique for efficient approximate nearest neighbor search in high-dimensional data. It extends traditional LSH by introducing a dynamic and hierarchical structure.
  - #### **Key Features**
     - **Hierarchical Structure:** Constructs multiple LSH trees, each with hash tables to partition data hierarchically.
     - **Dynamic Hashing:** Adjusts hash functions based on data distribution for better handling of varying data densities.
     - **Bucket Management:** Similar items are hashed into the same buckets, enabling quick neighbor retrieval.
  - #### How It Works
     - ####  1. **Tree Construction:**
        - **Hashing:** Multiple sets of locality-sensitive hash functions partition data into buckets across different trees.
     - #### 2. **Query Processing:**
       - **Hash the Query:** Query points are hashed into buckets.
       - **Retrieve Candidates:** Potential neighbors are gathered from corresponding buckets.
       - **Refinement:** Candidates are ranked to find approximate nearest neighbors
         
--- 

### Composite Indexing in ANN
Composite indexing in Approximate Nearest Neighbor (ANN) search involves combining different indexing techniques to enhance search efficiency and effectiveness in high-dimensional datasets. Here are a few notable composite indexing approaches:
  - **IVF + PQ:** Uses inverted indexing for initial clustering (IVF) and product quantization (PQ) for efficient search within clusters.
  - **LSH + KDTree:** Applies locality-sensitive hashing (LSH) for fast candidate selection and KDTree for accurate nearest neighbor refinement.
  - **HNSW + IVF:** Utilizes hierarchical navigable small world (HNSW) graphs for quick navigation and inverted file (IVF) for detailed search within clusters.

### Comparision between Different Indexing Technique

| **Aspect**                     | **Flat Indexing**                   | **Inverted Index**                | **Locality-Sensitive Hashing (LSH)** | **Product Quantization (PQ)**        | **Vector Quantization (VQ)**         | **Tree-Based Indexing**               | **Graph-Based Indexing**              | **LSH Forest**                        | **Composite Indexing**                |
|--------------------------------|-------------------------------------|-----------------------------------|--------------------------------------|---------------------------------------|---------------------------------------|----------------------------------------|----------------------------------------|---------------------------------------|---------------------------------------|
| **Storage Requirement**        | High                                | Moderate                          | High                                 | Low to moderate                       | Moderate                              | Moderate                               | High                                   | High                                  | Moderate to high                       |
| **Query Time**                 | Slow (O(N))                         | Fast (O(log N))                   | Fast (Sub-linear)                    | Fast (Sub-linear)                     | Fast (Sub-linear)                     | Moderate (O(log N))                    | Fast (Sub-linear)                      | Fast (Sub-linear)                     | Fast (Varies)                          |
| **Accuracy**                   | High (Exact search)                 | Moderate to High                  | Moderate (Approximate search)        | High                                  | Moderate to High                      | High in low dimensions                 | High                                   | Moderate (Approximate search)         | High (Depending on combination)        |
| **Complexity**                 | Low                                 | Moderate                          | High                                 | High                                  | Moderate                              | High                                   | High                                   | High                                  | Very High                              |
| **Scalability**                | Poor                                | Good                              | Good                                 | Excellent                             | Good                                  | Good                                   | Excellent                              | Good                                  | Excellent                              |
| **Implementation Effort**      | Low                                 | Moderate                          | High                                 | High                                  | Moderate                              | High                                   | High                                   | High                                  | Very High                              |
| **Best Dimensional Suitability** | Low to moderate                   | Low to moderate                   | High                                 | High                                  | Moderate                              | Low to moderate                        | High                                   | High                                  | High                                   |
| **Real-World Applications**    | Small datasets, exact matching      | Text search, document retrieval   | High-dimensional data, recommendation systems | Large-scale image and video retrieval | Compression, speech recognition       | Spatial data, geometric search         | Large-scale, high-dimensional search   | High-dimensional approximate search   | Large-scale, multimedia search         |
| **Notable Examples**           | Small e-commerce sites              | Elasticsearch, Apache Lucene      | Image search in social networks, recommender systems | Facebook image search                 | Mobile voice compression              | Geographic Information Systems (GIS)   | Social media content retrieval, search engines | Spotify music recommendation         | Google, Facebook's image and video search |
| **Advantages**                 | Simple, accurate                    | Efficient for sparse data         | Fast approximate search              | Good balance of accuracy and efficiency | Reduces data size, improves search speed | Efficient for low to moderate dimensions | High accuracy and efficiency           | Improved accuracy over single LSH     | Combines strengths of multiple techniques |
| **Disadvantages**              | Slow for large datasets             | Requires manual keyword assignment | Lower accuracy, space-intensive      | Complex to implement                   | Loss of accuracy due to quantization  | Performance degrades in high dimensions | Complex to build and maintain          | High memory usage, complex implementation | Most complex to implement and manage   |
| **Optimization Techniques**    | None                                | Compression, pruning              | Optimized hashing, multi-probe LSH   | Advanced quantization techniques, multi-codebooks | Optimized clustering algorithms        | Balanced trees, advanced partitioning schemes | Efficient graph traversal algorithms, shortcuts | Optimized hashing, multiple hash tables | Combining best practices from individual techniques |
