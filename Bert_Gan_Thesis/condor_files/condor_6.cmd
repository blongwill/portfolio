executable = src/run_all.sh
getenv = true
arguments = "config_6"
output = debug/config_6.condor_output
error = debug/config_6.condor_error
log = debug/config_6.condor_log
notification = complete
request_memory = 4096
request_GPUs = 1
Queue
