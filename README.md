# Scientific Fraud NLP

## How to Run

1. Install the required dependencies.

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Download the most recent version of the Retraction Watch dataset from [Crossref's Retraction Watch data repository](https://gitlab.com/crossref/retraction-watch-data/-/blob/main/retraction_watch.csv), then save it as `retraction_watch.csv` in the project root.

3. Filter and download the set of fraudulent papers to use.

   ```bash
   python3 0_download_fraud_papers.py
   ```

4. Search for and download a control set of comparable non-fraudulent papers on similar topics.

   ```bash
   python3 1_download_control_papers.py
   ```

5. Extract the NLP features from each paper.

   ```bash
   python3 2_extract_features.py
   ```

6. Optional: Run statistics on the extracted NLP features.

   ```bash
   python3 3_run_stats.py
   ```

7. Build the paper embeddings. This only needs to be run once unless the paper data changes.

   ```bash
   python3 4_make_pubmed_embeddings.py
   ```

8. Run the ML pipeline to build the prediction model.

   ```bash
   python3 4_ML_pipeline.py
   ```

## License

The original source code in this repository is licensed under the BSD 3-Clause License.

Third-party content in this repository, such as the full-text articles downloaded from PubMed Central (PMC), is not authored by me and does not fall under this repository's BSD 3-Clause License. It remains under its original copyright and license terms, as listed in the PMC metadata.

Users are responsible for complying with the license terms for each article.
