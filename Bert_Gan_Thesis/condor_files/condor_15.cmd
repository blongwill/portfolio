executable = src/run_all.sh
getenv = true
arguments = "config_15"
output = debug/config_15.condor_output
error = debug/config_15.condor_error
log = debug/config_15.condor_log
notification = complete
request_memory = 4096
request_GPUs = 1
Queue
