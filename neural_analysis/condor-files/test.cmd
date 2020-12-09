executable = src/main.sh
getenv = true
arguments = "experiment_test"
output = debug/condor_test/test.condor_output
error = debug/condor_test/test.condor_error
log = debug/condor_test/test.condor_log
notification = complete
request_memory = 4096
request_GPUs = 1
Queue
