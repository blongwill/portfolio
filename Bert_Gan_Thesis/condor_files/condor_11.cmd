executable = src/run_all.sh
getenv = true
arguments = "config_11"
output = debug/config_11.condor_output
error = debug/config_11.condor_error
log = debug/config_11.condor_log
notification = complete
request_memory = 4096
request_GPUs = 1
Queue