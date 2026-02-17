# Behavioral Bias Detection in Financial Text  
### Applying Behavioral Finance Theory Using NLP & Machine Learning

---

## Project Motivation

Behavioral finance studies how psychological biases influence financial decision-making. Concepts such as **Overconfidence**, **Loss Aversion**, and **Rational Behavior** are central to the CFA Level I curriculum.

While these biases are typically discussed theoretically, this project attempts to **quantify behavioral bias using Natural Language Processing (NLP)** by classifying financial statements into behavioral categories.

This work demonstrates how behavioral finance theory can be operationalized into a predictive machine learning framework.

---

## Objective

To build and evaluate machine learning models that classify financial sentences into:

- Overconfidence  
- Loss Aversion  
- Rational Behavior  

Using:

- TF-IDF feature engineering  
- Supervised machine learning  
- Stratified K-Fold Cross Validation  
- Macro F1 Score for evaluation  

---

## Dataset

**Source:** Financial PhraseBank  
**Size:** 2,264 financial sentences  

### Original Labels

- Positive  
- Neutral  
- Negative  

### Behavioral Mapping

| Original Sentiment | Behavioral Category |
|-------------------|--------------------|
| Positive          | Overconfidence     |
| Negative          | Loss Aversion      |
| Neutral           | Rational           |

---

## Class Distribution

The dataset is imbalanced:

- Rational ≈ 61%  
- Overconfidence ≈ 25%  
- Loss Aversion ≈ 13%  

Because of this imbalance, **Macro F1 Score** was used instead of accuracy to ensure fair evaluation across all classes.

---

## Methodology

### Text Vectorization

- TF-IDF Vectorizer  
- `ngram_range = (1,2)`  
- `max_features = 10,000`  
- English stopwords removed  

---

### Models Compared

- Logistic Regression  
- Multinomial Naive Bayes  
- Linear Support Vector Machine (Linear SVM)  

All models were evaluated using:

- Stratified 5-Fold Cross Validation  
- Macro F1 Score  

---

## Model Evaluation

### Cross-Validation (5-Fold Macro F1)

| Model                 | Avg Macro F1 |
|-----------------------|--------------|
| Logistic Regression   | ~0.85        |
| Naive Bayes           | ~0.72        |
| Linear SVM            | ~0.85+       |

---

### Test Set Accuracy

| Model                 | Accuracy |
|-----------------------|----------|
| Logistic Regression   | ~0.89    |
| Naive Bayes           | ~0.78    |
| Linear SVM            | ~0.90    |

**Best Performing Classical Model: Linear SVM**

---

## Key Observations

1. Linear models perform well on high-dimensional sparse TF-IDF features.  
2. Naive Bayes underperforms due to feature independence assumptions.  
3. Class imbalance significantly affects minority class recall.  
4. Macro F1 is more reliable than accuracy in imbalanced classification scenarios.  

### Confusion Matrix Insights

The model performs strongly on the majority **Rational** class, achieving high recall.

However, minority classes such as **Loss Aversion** and **Overconfidence** show some misclassification between each other, indicating semantic overlap in financial language.

This suggests that more advanced contextual models (e.g., transformer-based architectures) may better capture nuanced behavioral cues beyond bag-of-words features.

---

## Why Macro F1 Instead of Accuracy?

Since 61% of the dataset belongs to the *Rational* class, a model predicting mostly Rational could achieve high accuracy while ignoring minority classes.

Macro F1 ensures:

- Equal importance to each behavioral category  
- Fair evaluation of minority classes (e.g., Loss Aversion)  

This is critical in financial modeling where rare behavioral signals may carry significant informational value.

---

## Hyperparameter Optimization

GridSearchCV was performed with:

- `ngram_range`: (1,1) and (1,2)  
- `max_features`: 5000, 10000, 15000  
- `C` values: 0.1 to 50  

### Best Parameters

- C = 10  
- ngram_range = (1,2)  
- max_features = 10000  

### Final Classical Model Performance

- Macro F1 = 0.84  
- Accuracy = 0.88  

---

## Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret feature-level contributions.

### Key Findings

- **Loss Aversion** is driven by words indicating declines and negative performance.  
- **Overconfidence** is associated with strong forward-looking growth language.  
- **Rational** class is dominated by neutral financial reporting terminology.  

This confirms that the model captures meaningful behavioral finance patterns rather than random statistical correlations.

---

## Transformer Upgrade: DistilBERT

To improve contextual understanding, a fine-tuned **DistilBERT** model was implemented.

### Model Comparison

| Model                     | Macro F1 Score |
|---------------------------|----------------|
| TF-IDF + Linear SVM       | 0.85           |
| DistilBERT (Fine-tuned)   | 0.96           |

The transformer model significantly outperformed the classical approach, demonstrating the importance of contextual language understanding in detecting behavioral biases.

---

## Tech Stack

- Python  
- pandas  
- scikit-learn  
- TF-IDF Vectorization  
- Stratified K-Fold Cross Validation  
- HuggingFace Transformers  
- PyTorch  
- SHAP  
- Jupyter Notebook  

---

## Financial Relevance

This project bridges:

**Behavioral Finance (CFA Level I)**  
and  
**Machine Learning & NLP**

It demonstrates how theoretical psychological biases can be transformed into quantifiable predictive signals using financial text data.

### Potential Applications

- Earnings call bias detection  
- CEO statement overconfidence tracking  
- Market sentiment anomaly detection  
- Behavioral alpha signal research  

---

## Author

CFA Level I Candidate  
Data Science Undergraduate  
Exploring the intersection of Behavioral Finance and Machine Learning
