executable = src/run_all.sh
getenv = true
arguments = "config_20"
output = debug/config_20.condor_output
error = debug/config_20.condor_error
log = debug/config_20.condor_log
notification = complete
request_memory = 4096
request_GPUs = 1
Queue
