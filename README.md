# Scientific Fraud NLP

## How to Run

1. Download the project from GitHub, then create a virtual environment.

   ```bash
   git clone https://github.com/xPugnocode/scientific-fraud-nlp.git
   cd scientific-fraud-nlp
   python3 -m venv .venv
   ```

2. Enter the virtual environment and install the required dependencies.

   ```bash
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

3. Optional: Download the most recent version of the Retraction Watch dataset from [Crossref's Retraction Watch data repository](https://gitlab.com/crossref/retraction-watch-data/-/blob/main/retraction_watch.csv) and save it as `retraction_watch.csv` in the project root. Otherwise, the version of the Retraction Watch dataset contained within the repository is the same one used in the paper.

   ```bash
   curl -L "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv" -o retraction_watch.csv
   ```

4. Filter and download the set of fraudulent papers to use.

   ```bash
   python3 0_download_fraud_papers.py
   ```

5. Search for and download a control set of comparable non-fraudulent papers on similar topics.

   ```bash
   python3 1_download_control_papers.py
   ```

6. Extract the NLP features from each paper.

   ```bash
   python3 2_extract_features.py
   ```

7. Optional: Run statistics on the extracted NLP features.

   ```bash
   python3 3_run_stats.py
   ```

8. Build the paper embeddings. This only needs to be run once unless the paper data changes.

   ```bash
   python3 4_make_pubmed_embeddings.py
   ```

9. Run the ML pipeline to build the prediction model.

   ```bash
   python3 4_ML_pipeline.py
   ```

10. Test an individual paper with the best model, the combined RBF SVM. Save the paper body text as a plain text file without the bibliography, then pass the file path and the number of in-body citation markers. 

   ```bash
   python3 5_test_individual_paper.py path/to/paper_body.txt 67
   ```

   The citation number should count how many times references are cited in the body of the paper, not how many references appear in the bibliography.

## Paper

The LaTeX source for the paper is in `paper/`. See `paper/README.md` for instructions on building the PDF.

## License

The original source code in this repository is licensed under the BSD 3-Clause License.

Third-party content in this repository, such as the full-text articles downloaded from PubMed Central (PMC), is not authored by me and does not fall under this repository's BSD 3-Clause License. It remains under its original copyright and license terms, as listed in the PMC metadata.

Users are responsible for complying with the license terms for each article.
