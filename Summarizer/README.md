# 573-summarizer
LING 573 Group Project
Nutshell topic-focused multi-document extractive text summarizer

## Getting Started
Run D4.cmd on condor in order to run the Nutshell text summarizer.

Note that this will take 20-45 minutes as it runs both devtest and evaltest and does compression of parse trees, which is time-intensive.

This will create summary files in the `outputs/D4_devtest/` and `outputs/D4_evaltest/` directories, and the ROUGE results in /results/ directory.

This will run the D4 version of Nutshell, which has all modules (Content Selection, Information Ordering, and Content Realzation) implemented, with the following parameters:

```
dev_path = /dropbox/18-19/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml
eval_path = /dropbox/18-19/573/Data/Documents/evaltest/GuidedSumm11_test_topics.xml 
output_folder = D4
stemming = True
lower = False
idf_type = 'smooth_idf'
tf_type = 'term_frequency'
d = 0.2
intersent_threshold = 0.1
summary_threshold = 0.3
epsilon = 0.04
mle_lambda = 0.6
k = 9
min_sent_len = 3
include_narrative = False
bias_formula = 'rel'
intersent_formula = 'cos'
info_order_type = 'entity'
num_permutations = 5
remove_header = True
remove_parens = True
remove_quotes = True
remove_appos = True
remove_advcl = True
remove_relcl = True
remove_acl = True
```

## Authors
Shannon Ladymon, sladymon@uw.edu

Haley Lepp, hlepp@uw.edu 

Ben Longwill, longwill@uw.edu

Amina Venton, aventon@uw.edu

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) for details

