#!/bin/bash
input_file=$1
output_file=$2

tr -d '' <$input_file >$output_file
sed -i 's/https\?:\S*/URL/g' $output_file
sed -i 's/@\S\+/USER/g' $output_file

