# 🎬 Deep Learning for Content-Based Movie Recommendation

---

In this project, I built a **content-based movie recommender system** using a neural network. The system learns from movie features (such as genre and release year) and user preferences (such as average ratings by genre) to suggest movies tailored to each user.

The project demonstrates how deep learning can be applied to recommendation systems beyond simple collaborative filtering. Instead of relying only on user-item interactions, the model leverages **movie metadata and engineered user features** to improve prediction accuracy.

---

## 📑 Outline

* [1 - Packages](#1---packages)
* [2 - Movie ratings dataset](#2---movie-ratings-dataset)
* [3 - Content-based filtering with a neural network](#3---content-based-filtering-with-a-neural-network)

  * [3.1 Training Data](#31-training-data)
  * [3.2 Preparing the training data](#32-preparing-the-training-data)
* [4 - Neural Network architecture](#4---neural-network-architecture)
* [5 - Predictions](#5---predictions)

  * [5.1 Predictions for a new user](#51---predictions-for-a-new-user)
  * [5.2 Predictions for an existing user](#52---predictions-for-an-existing-user)
  * [5.3 Finding Similar Items](#53---finding-similar-items)
* [6 - Results & Conclusion](#6---results--conclusion)
* [✅ Solved Exercises](#-solved-exercises)

---

<a name="1---packages"></a>

## 1 - Packages

I used the following main Python packages:

* **NumPy** → numerical computing
* **Pandas** → dataset organization and analysis
* **TensorFlow / Keras** → neural networks
* **scikit-learn** → preprocessing and scaling
* **tabulate** → neat tabular display

---

<a name="2---movie-ratings-dataset"></a>

## 2 - Movie Ratings Dataset

The dataset comes from the [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/latest/) dataset.

* **Original size**: \~9000 movies, 600 users, 100k ratings
* **Reduced size (my project)**:

  * Users: **397**
  * Movies: **847**
  * Ratings: **25,521**
* Each movie has:

  * Title, release year
  * Genres (14 total categories, one-hot encoded)
* Each user has:

  * Per-genre average ratings (engineered feature)
  * Total rating count and rating average

This dataset provides **rich movie-level metadata** but only limited user-level information. I engineered additional user features to improve learning.

---

<a name="3---content-based-filtering-with-a-neural-network"></a>

## 3 - Content-Based Filtering with a Neural Network

Unlike collaborative filtering, which relies solely on user-item rating matrices, this approach integrates **content features**.

The architecture learns:

* A **user feature vector** from user profile data
* A **movie feature vector** from metadata (genres, year, average rating)

The dot product of these vectors predicts the rating a user would give a movie.

---

<a name="31-training-data"></a>

### 3.1 Training Data

* Movie features: release year, one-hot genres, average rating
* User features: per-genre average ratings
* Target: actual rating (0.5 to 5.0)

---

<a name="32-preparing-the-training-data"></a>

### 3.2 Preparing the Training Data

To optimize convergence:

* **StandardScaler** → applied to input features
* **MinMaxScaler (-1, 1)** → applied to ratings

The dataset was then split into **80% training / 20% testing**.

---

<a name="4---neural-network-architecture"></a>

## 4 - Neural Network Architecture

The recommender network consists of **two parallel towers**:

* A **user tower** (dense layers)
* An **item tower** (dense layers)

Each produces a 32-dimensional vector. Their **dot product** gives the predicted rating.

Architecture:

* Dense(256, ReLU)
* Dense(128, ReLU)
* Dense(32, Linear)
* L2 normalization
* Dot product

Optimizer: **Adam (lr=0.01)**
Loss: **Mean Squared Error (MSE)**

---

<a name="5---predictions"></a>

## 5 - Predictions

<a name="51---predictions-for-a-new-user"></a>

### 5.1 Predictions for a New User

I simulated a **fantasy/adventure movie fan**.
The system correctly suggested **fantasy/adventure-heavy movies** such as *Harry Potter* and *The Lord of the Rings*.

---

<a name="52---predictions-for-an-existing-user"></a>

### 5.2 Predictions for an Existing User

For an actual user from the dataset:

* Predictions were usually within ±1 rating point
* Strong genre alignment was captured well
* Edge cases (movies far from genre averages) were harder to predict

---

<a name="53---finding-similar-items"></a>

### 5.3 Finding Similar Items

Using the **movie embeddings** learned by the model, I computed squared distances between movie vectors.

Result:

* Similar genres grouped together
* Example: *Toy Story 3* → closest matches were other Pixar/animation films

---

<a name="6---results--conclusion"></a>

## 6 - Results & Conclusion

I successfully built a **deep learning content-based recommender system** that:

* Learns embeddings for both users and movies
* Uses engineered features + metadata for stronger predictions
* Produces recommendations and similarity-based suggestions

This architecture is flexible and can be extended to:

* Books, music, or shopping recommendations
* Hybrid models (combine collaborative + content features)

---

# ✅ Solved Exercises

### **Exercise 1 — Neural Network Towers**

```python
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs)   # linear output
])

item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs)
])
```

---

### **Exercise 2 — Squared Distance Function**

```python
def sq_dist(a, b):
    """
    Returns the squared distance between two vectors
    """
    d = np.sum(np.square(a - b))
    return d

# ✅ Test Cases
a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
a3 = np.array([0, 1, 0]);       b3 = np.array([1, 0, 0])

print(sq_dist(a1, b1))  # 0.0
print(sq_dist(a2, b2))  # 0.03
print(sq_dist(a3, b3))  # 2.0
```

**Expected Output:**

```
0.0
0.030000000000000054
2.0
```

---
