Patent Classification with PatentSBERTa
This repository implements a patent classification system to identify relevant patents using PatentSBERTa, a domain-adapted Sentence-BERT model, and logistic regression. The system achieves an F1-score of 0.57 on a dataset of 307 patents, outperforming traditional and generic approaches. It is designed for technical queries like "debris shield upper tie plate spring" in a highly imbalanced dataset (5% relevant patents).
Project Overview
The goal is to classify patents as relevant or irrelevant based on their technical content. We use PatentSBERTa for domain-specific text embeddings and logistic regression with class balancing to handle data imbalance. Our approach performs competitively despite a small dataset, demonstrating the effectiveness of domain-adapted models for patent analysis.
Key Features

Domain-specific embeddings: PatentSBERTa captures technical terminology better than generic transformer models.
Robust methodology: Logistic regression with balanced class weights and an 80/20 train-test split prevents overfitting.
High performance: Achieves F1=0.57, a 3.8x improvement over baseline transformer models (F1=0.15).
Reproducible: Includes scripts to preprocess data, train models, and evaluate results.

Results



Approach
Model
Precision
Recall
F1-Score



Approach 2
Generic BERT + Cosine Similarity
0.08
0.93
0.15


Approach 4
PatentSBERTa + Logistic Regression
0.50
0.67
0.57



Comparison:
Outperforms traditional methods like TF-IDF with SVM (F1=0.30–0.40).
Matches state-of-the-art domain-specific models (F1=0.55–0.65) with fewer data.
Approaches human expert performance (F1=0.70–0.80).


Why it works: PatentSBERTa’s bidirectional architecture and domain-specific pre-training excel in understanding technical jargon, unlike unidirectional language models.

Requirements

Python 3.8+
Libraries (see requirements.txt):pip install numpy pandas torch transformers sentence-transformers scikit-learn


Pre-trained model: AI-Growth-Lab/PatentSBERTa (downloaded automatically via sentence-transformers).

Installation

Clone the repository:
git clone https://github.com/crosby9414/patent-search.git
cd patent-search


Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Prepare a dataset:

Format: CSV with columns patent_id, title, abstract, label (1 for relevant, 0 for irrelevant).
Example: data/patents.csv (not included due to proprietary restrictions).
See "Data Acquisition" below for obtaining patent data.



Usage

Run the classification pipeline:
python patent_search.py --data data/patents.csv --model AI-Growth-Lab/PatentSBERTa --output results/


Inputs:
--data: Path to the input CSV with patent data.
--model: Pre-trained model (default: AI-Growth-Lab/PatentSBERTa).
--output: Directory for results (metrics, trained model).


Outputs:
results/metrics.json: Precision, recall, F1-score.
results/model.pkl: Trained logistic regression model.




Predict on new data:
python patent_search.py --mode predict --model results/model.pkl --input data/new_patents.csv --output predictions.csv


Outputs predictions in predictions.csv.



Data Acquisition
To obtain patent data with titles and abstracts for classification:

The Lens (Recommended):
Visit lens.org, register, and search (e.g., "debris shield upper tie plate spring").
Export up to 10,000 records in CSV, including "Title" and "Abstract" fields.
Save as data/patents.csv.


Open Patent Services (OPS):
Register at my.epo.org for OPS API access.
Use a script to fetch titles and abstracts (example below):import requests
import pandas as pd

# Example: Fetch abstracts (requires OPS credentials)
consumer_key = "YOUR_KEY"
consumer_secret = "YOUR_SECRET"
token_url = "https://ops.epo.org/3.2/auth/accesstoken"
biblio_url = "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc/{}/biblio"

# Get access token
token_response = requests.post(token_url, data={"consumerKey": consumer_key, "consumerSecret": consumer_secret})
access_token = token_response.json().get("access_token")

# Example patent numbers
patents = [{"Publication Number": "EP1234567"}, {"Publication Number": "US9876543"}]
headers = {"Authorization": f"Bearer {access_token}"}
for patent in patents:
    response = requests.get(biblio_url.format(patent["Publication Number"]), headers=headers)
    patent["Abstract"] = response.json().get("abstract", "N/A")

# Save to CSV
pd.DataFrame(patents).to_csv("data/patents.csv", index=False)


Note: Some abstracts may be unavailable or untranslated for non-English patents.


Espacenet Limitations:
Espacenet’s CSV export includes titles but not abstracts.
Use The Lens or OPS for abstracts, as manual copying is impractical for large datasets.




Notes

The dataset (307 patents, 5% relevant) is not included due to proprietary restrictions. Prepare a similar CSV with titles, abstracts, and labels.
Ensure compliance with EPO’s Fair Use Policy when using OPS API.
For large datasets, consider batch processing to handle API limits.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or contributions, please contact:

GitHub: crosby9414
Issues: Open an issue

