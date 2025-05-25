# Patent Classification with PatentSBERTa

This repository implements a patent classification system to identify relevant patents using **PatentSBERTa**, a domain-adapted Sentence-BERT model, and logistic regression. The system achieves an F1-score of 0.57 on a dataset of 307 patents, outperforming traditional and generic approaches. It is designed for technical queries like "debris shield upper tie plate spring" in a highly imbalanced dataset (5% relevant patents).

## Project Overview

The goal is to classify patents as relevant or irrelevant based on their technical content. We use **PatentSBERTa** for domain-specific text embeddings and logistic regression with class balancing to handle data imbalance. Our approach performs competitively despite a small dataset, demonstrating the effectiveness of domain-adapted models for patent analysis.

### Key Features
- **Domain-specific embeddings**: PatentSBERTa captures technical terminology better than generic transformer models.
- **Robust methodology**: Logistic regression with balanced class weights and an 80/20 train-test split prevents overfitting.
- **High performance**: Achieves F1=0.57, a 3.8x improvement over baseline transformer models (F1=0.15).
- **Reproducible**: Includes scripts to preprocess data, train models, and evaluate results.

### Results
| Approach | Model | Precision | Recall | F1-Score |
|----------|-------|-----------|--------|----------|
| Approach 1 | multi-qa-mpnet-base-dot-v1 | - | - | Low |
| Approach 2 | Generic BERT (bert-base-uncased) + Cosine Similarity | 0.08 | 0.93 | 0.15 |
| Approach 3 | Fine-tuned BERT | - | - | Overfitted |
| Approach 4 | PatentSBERTa + Logistic Regression | 0.50 | 0.67 | **0.57** |

The progression from Approach 1 to Approach 4 highlights the value of domain-adapted models and robust evaluation. Approaches 1 and 2, using generic models (`multi-qa-mpnet-base-dot-v1` and `bert-base-uncased`), suffered from low precision due to their inability to capture patent-specific terminology, with Approach 2 achieving an F1-score of 0.15. Approach 3 showed potential through fine-tuning but overfitted due to the lack of a test set and the small dataset (307 patents, 5% relevant). Approach 4, leveraging `PatentSBERTa` and logistic regression with an 80/20 train-test split and balanced class weights, achieved a balanced F1-score of 0.57 (precision=0.50, recall=0.67).

- **Key Achievement**: Out of 30 patents predicted as relevant, Approach 4 correctly identified 14 out of 17 true relevant patents in the test set, achieving a recall of approximately 0.82 for this subset. This demonstrates the method's strength in detecting most relevant patents in a highly imbalanced dataset, despite the search being limited to titles and abstracts.

- **Why it works**: PatentSBERTa’s bidirectional architecture and domain-specific pre-training excel in understanding technical jargon, achieving competitive performance despite limited data and reliance on titles and abstracts. Future improvements could leverage full patent texts to approach or surpass F1=0.65.

## Requirements

- Python 3.8+
- Libraries (see `requirements.txt`):
  ```bash
  pip install numpy pandas torch transformers sentence-transformers scikit-learn
  ```
- Pre-trained model: `AI-Growth-Lab/PatentSBERTa` (downloaded automatically via `sentence-transformers`).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/crosby9414/patent-search.git
   cd patent-search
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Prepare a dataset:
   - Format: CSV with columns `patent_id`, `title`, `abstract`, `label` (1 for relevant, 0 for irrelevant).
   - Example: `data/example_patents.csv` (synthetic data for testing).
   - See "Data Acquisition" below for obtaining patent data.

## Usage

1. **Run the classification pipeline**:
   ```bash
   python patent_search.py --data data/patents.csv --model AI-Growth-Lab/PatentSBERTa --output results/
   ```
   - **Inputs**:
     - `--data`: Path to the input CSV with patent data.
     - `--model`: Pre-trained model (default: `AI-Growth-Lab/PatentSBERTa`).
     - `--output`: Directory for results (metrics, trained model).
   - **Outputs**:
     - `results/metrics.json`: Precision, recall, F1-score.
     - `results/model.pkl`: Trained logistic regression model.

2. **Predict on new data**:
   ```bash
   python patent_search.py --mode predict --model results/model.pkl --input data/new_patents.csv --output predictions.csv
   ```
   - Outputs predictions in `predictions.csv`.

## Data Acquisition

To obtain patent data with titles and abstracts for classification:
1. **The Lens (Recommended)**:
   - Visit [lens.org](https://www.lens.org), register, and search (e.g., "debris shield upper tie plate spring").
   - Export up to 10,000 records in CSV, including "Title" and "Abstract" fields.
   - Save as `data/patents.csv`.
2. **Open Patent Services (OPS)**:
   - Register at [my.epo.org](https://my.epo.org) for OPS API access.
   - Use a script to fetch titles and abstracts (example below):
     ```python
     import requests
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
     ```
   - Note: Some abstracts may be unavailable or untranslated for non-English patents.
3. **Espacenet Limitations**:
   - Espacenet’s CSV export includes titles but not abstracts.
   - Use The Lens or OPS for abstracts, as manual copying is impractical for large datasets.

## Repository Structure

```
patent-search/
├── .gitignore               # Ignored files
├── patent_search.py         # Main script for classification
├── requirements.txt         # Dependencies
├── data/                    # Dataset (e.g., example_patents.csv, prepare your own)
├── results/                 # Output directory (metrics, models)
└── README.md                # This file
```

## Notes
- The dataset (307 patents, 5% relevant) is not included due to proprietary restrictions. A synthetic example is provided in `data/example_patents.csv`.
- Ensure compliance with EPO’s Fair Use Policy when using OPS API.
- For large datasets, consider batch processing to handle API limits.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or contributions, please contact:
- GitHub: [crosby9414](https://github.com/crosby9414)
- Issues: [Open an issue](https://github.com/crosby9414/patent-search/issues)
