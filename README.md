# scientific-fraud-nlp
## How to Run
1. Install the required dependencies.
```bash
pip install -r requirements.txt
```
2. Download the most recent version of the Retraction Watch dataset [here](https://gitlab.com/crossref/retraction-watch-data/-/blob/main/retraction_watch.csv).
3. Filter and download the set of fraudulent papers to use.
```bash
python3 0_download_fraud_papers.py
```
4. Search for and download a control set of comparable non-fraudulent papers on similar topics.
```bash
python3 1_download_control_papers.py
```

## License
The original source code in this repository is licensed under the BSD 3-Clause License.
Third party content in this repository, such as the full-text articles downloaded from PubMed Central (PMC) are not authored by me and do not fall under this repository's BSD 3-Clause License. They remain under their original copyright and license terms, listed in the PMC metadata. 
Users are responsible for complying with the license terms for each article.