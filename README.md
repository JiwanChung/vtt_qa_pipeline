# Video QA Pipeline

This repository contains pipelines to conduct video QA with deep learning based models.
It supports image loading, feature extraction, feature caching, training framework, tensorboard logging and more.

We use python3 (3.6.6), and python2 is not supported.
We use PyTorch (1.1.0), though tensorflow-gpu is necessary to launch tensorboard.

## Install

```bash
git clone (this repo)
cd vtt_qa_pipeline/startup
(use python 3.6.6)
pip install -r requirements.txt
python -m nltk.downloader 'punkt'
```

## How to Use

```bash
cd startup
python cli.py train
```

Access the prompted tensorboard port to view basic statistics.

For further configurations, take a look at `startup/config.py` and
[fire][https://github.com/google/python-fire].

## Data Folder Structure

- question for shots only
- images are resized to 224X224 for preprocessing (resnet input size)
- using last layer of resnet50 for feature extraction (base behaviour)
- using glove.6B.300d for pretrained word embedding
- storing image feature cache after feature extraction (for faster dataloading)
- using random splits for train, test, val (8: 1: 1) respectively
- using multiprocessing for faster processing
- using nltk.word_tokenize for tokenization
- You should first run **json_to_jsonl.py** before running the preprocessing code
  to change the question data formats (this procedure may be integrated to the main code upon discussion)

The data folder should be structured as follows:

> ./data
> ./data/QA_train_set_s1s2.json
> ./data/images/s0101/...

The preprocessing command should be the following:

> python json_to_jsonl.py
> python cli.py check_dataloader

## Troubleshooting

See the Troubleshooting page and submit a new issue or contact us if you cannot find an answer.

## Contact Us

To contact us, send an email to jiwanchung@vision.snu.ac.kr
