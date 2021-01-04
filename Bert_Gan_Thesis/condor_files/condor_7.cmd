executable = src/run_all.sh
getenv = true
arguments = "config_7"
output = debug/config_7.condor_output
error = debug/config_7.condor_error
log = debug/config_7.condor_log
notification = complete
request_memory = 4096
request_GPUs = 1
Queue
