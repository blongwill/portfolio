executable = src/run_all.sh
getenv = true
arguments = "config_18"
output = debug/config_18.condor_output
error = debug/config_18.condor_error
log = debug/config_18.condor_log
notification = complete
request_memory = 4096
request_GPUs = 1
Queue
