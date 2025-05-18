Patent Classification with PatentSBERTa
This repository contains the implementation of a patent classification system that identifies relevant patents using domain-specific embeddings from PatentSBERTa and logistic regression. The project achieves an F1-score of 0.57 on a dataset of 307 patents, significantly outperforming traditional methods like TF-IDF with SVM (F1=0.30–0.40) and generic BERT models (F1=0.15).
Project Overview
The goal of this project is to classify patents as relevant or irrelevant based on technical queries (e.g., "debris shield upper tie plate spring") in a highly imbalanced dataset (5% relevant patents). We leverage PatentSBERTa, a domain-adapted Sentence-BERT model pre-trained on patent texts, combined with logistic regression to handle class imbalance. Our approach demonstrates comparable performance to state-of-the-art models like PatentBERT (F1=0.55–0.65) despite using a small dataset.
Key Features

Domain-specific embeddings: PatentSBERTa captures technical terminology better than generic BERT models.
Robust methodology: Logistic regression with class balancing and train-test splitting (80/20) prevents overfitting.
High performance: Achieves F1=0.57, a 3.8x improvement over baseline BERT (F1=0.15).
Reproducible: Includes scripts and instructions to replicate results.

Results



Approach
Model
Precision
Recall
F1-Score



Approach 2
BERT (bert-base-uncased) + Cosine Similarity
0.08
0.93
0.15


Approach 4
PatentSBERTa + Logistic Regression
0.50
0.67
0.57





Why it works: PatentSBERTa’s bidirectional architecture  and domain-specific pre-training excel in understanding technical jargon, unlike unidirectional models like GPT

Requirements

Python 3.8+
Libraries:pip install pandas numpy scikit-learn sentence-transformers torch


Pre-trained model: AI-Growth-Lab/PatentSBERTa (automatically downloaded via sentence-transformers).




Data Acquisition
To obtain patent data with titles and abstracts:

Use The Lens (lens.org) to export up to 10,000 patents in CSV, including abstracts.
Alternatively, leverage the Open Patent Services (OPS) API from EPO to programmatically retrieve titles and abstracts. See scripts/fetch_data_ops.py for an example (requires EPO registration).
Note: Some abstracts may be unavailable or untranslated for non-English patents.


License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or contributions, please contact:

Email: timur94yasko@gmail.com
GitHub Issues: Open an issue


