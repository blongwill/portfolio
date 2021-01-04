executable = src/run_all.sh
getenv = true
arguments = "config_13"
output = debug/config_13.condor_output
error = debug/config_13.condor_error
log = debug/config_13.condor_log
notification = complete
request_memory = 4096
request_GPUs = 1
Queue
