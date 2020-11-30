universe = vanilla
executable = /opt/python/gpu/python-3.6/bin/python3
getenv = true
#input =
output = cmd_out_D4_1_0_0_0_0_0_0
error = error_D4_1_0_0_0_0_0_0
log = log_D4_1_0_0_0_0_0_0
arguments = src/text_summarizer.py /dropbox/18-19/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml /dropbox/18-19/573/Data/Documents/evaltest/GuidedSumm11_test_topics.xml test_D4_1_0_0_0_0_0_0 both 1 0 smooth_idf term_frequency 0.2 0.1 0.3 0.04 0.6 9 3 0 rel cos entity 5 1 0 0 0 0 0 0
transfer_executable = false
notification = Always
queue
