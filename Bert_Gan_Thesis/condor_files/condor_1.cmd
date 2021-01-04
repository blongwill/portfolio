executable = src/run_all.sh
getenv = true
arguments = "config_1"
output = debug/config_1.condor_output
error = debug/config_1.condor_error
log = debug/config_1.condor_log
notification = complete
request_memory = 4096
request_GPUs = 1
Queue
