# VAE model
This repository completes the task of the course **Machine Learning**, which requires to transfer the aritistic form to the new Figs.

# About the report
You just need to open `tex/main.pdf`, and you can see the report. To see more, please enter `tex/main.tex` to vie the latex code.

## Running procedure
First download the reliable packages:
```bash
pip install -r requirement.txt
```
Then you need to download the datasets. The COCO daily scene datasets are available at this [page](https://cocodataset.org/) and the Wikiart datasets can be downloaded by running the code:
```bash
python download.py
```
According to the requirements, we just need 10,000 COCO daily scene figures and 8000 Wikiart figures, so we need to select figures via:
```bash
python select_files.py
```
Then the file folder is in the form below
```bash
ex6_VAE/
├── download.py
├── main.py
├── select_files.py
├── val2014/
└── wikiart_subset/
```

Finally, run the code:
```bash
python main.py
```