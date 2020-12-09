# README #

This repository represents our group's project for LING 575 Winter 2020: Analyzing Neural Language Models.

---

Experiment configurations should be added to the 'configs' folder,
e.g. experiment_1a.txt

---

This project requires a conda installation:

  wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
  bash Anaconda3-2019.10-Linux-x86_64.sh


Load the latest conda environment changes from Git into your local repo:

  conda env create -f environment.yml (if 575-project environment does not exist yet)
  conda env update -f environment.yml
  conda activate 575-project

If you modify this environment, update the environment file and push changes to Git:

  conda env export > environment.yml

---

If you have not run this project before, the BertTokenizer and Model must be cached before running on Condor:
  
  From the terminal:
  >>> python
  >>> from transformers import BertTokenizer
  >>> BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  >>> BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
  >>> from transformers import BertForSequenceClassification
  >>> BertForSequenceClassification.from_pretrained("bert-base-uncased")
  >>> BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')

---

To run an experiment:

1. Load and activate the latest conda environment from Git (see above)

2. Submit a condor job, specifying an experiment config name, e.g.:

  condor_submit -a "arguments = experiment_1a" run.cmd

---

To run visualization code:

1. Create a list of strings as input for the model you want to visualize. 
2. Call examine_model() from __main__ with the name of the experiment you want to visualize and the list of strings as the input parameters. 
3. You can view visualizations in the viz/{experiment name} folder

OR

Run viz.py to automatically generate visualizations for all trained models. 