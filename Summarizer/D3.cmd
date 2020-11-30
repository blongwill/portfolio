universe = vanilla
executable = /usr/bin/python3
getenv = true
#input =
output = cmd_out_3
error = error_3
log = log_3
arguments = src/text_summarizer.py /dropbox/18-19/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml D3 1 0 smooth_idf term_frequency 0.2 0.1 0.3 0.04 0.6 9 3 0 rel cos entity 5
transfer_executable = false
queue
