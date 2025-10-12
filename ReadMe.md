# SENTI-MD: Machine Learning for IMDb Sentiment Analysis

## SENTI-MD - AI-Powered Sentiment Classification

**Problem Statement:**

The core problem is to build a robust **Machine Learning (ML)** model capable of automatically determining the emotional tone (sentiment) expressed in a piece of text, specifically focusing on **IMDb movie reviews**. We aim to accurately classify any given review as either **Positive (1) or Negative (0)**.

---

## 1. Project Summary

Sentiment Analysis is a critical step in understanding audience reception, allowing businesses and filmmakers to quickly gauge public opinion without reading thousands of reviews.

### The Problem

Movie reviews are unstructured text data, making it impossible to analyze them at scale without automation. The challenge is in creating a model that can handle the nuances of human language, including context, to correctly map text features to a binary sentiment label (positive/negative).

### Our Solution: SENTI-MD

SENTI-MD is the classification model and pipeline designed to accurately predict sentiment using foundational Natural Language Processing (NLP) and Machine Learning techniques.

* **Pre-processing Pipeline:** A robust pipeline handles cleaning (HTML tags, punctuation), tokenization, stop word removal, and stemming/lemmatization.
* **Feature Extraction:** The cleaned text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**, optimized using **3-grams**.
* **Classification:** A **Logistic Regression** model is trained on the labeled IMDb dataset to achieve high accuracy in sentiment prediction.

---

## 2. The "Wow" Factor: The Actionable Insight

The model provides a direct answer to the question: **"Is the audience reaction to this movie positive or negative?"**

The model offers an immediate, quantifiable measure of public perception, turning thousands of lines of text into a simple, digestible metric (e.g., an 88.4% Accuracy score), driving quicker decisions for marketing and content strategy.

---

## 3. Technology Stack

* **Languages & Frameworks:** Python, Pandas, NumPy
* **Machine Learning Models:** **Scikit-learn** (Logistic Regression, SVM, Naive Bayes), **NLTK** (Text Preprocessing)
* **Feature Engineering:** **TF-IDF Vectorization** (using 3-grams)
* **Platform & Infrastructure:** The entire project is executed within the **Google Colab** environment.

**Colab Notebook Link:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1seeXX1aCdSp5wTzFrEXODmLhtme2nsI0?usp=sharing)

---

## 4. Key Results & Evaluation

The final model achieved the following performance metrics on the test set:

| Metric | Model | Result |
| :--- | :--- | :--- |
| **Best Classifier** | **Logistic Regression** | **0.884** |
| **Feature Method** | **TF-IDF (3-grams)** | **~0.88** |
| **Accuracy** | **0.884** | **N/A** |
| **F1-Score** | **N/A** | **~0.88** |

**Conclusion:** The **Logistic Regression** model, using **TF-IDF 3-grams**, was the top performer, achieving an accuracy of **88.4%**, validating the chosen approach for text classification.

---

## 5. How to Run the Project

The easiest way to run the project is directly via the Colab link provided above. Alternatively, to replicate the environment:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn nltk
    ```