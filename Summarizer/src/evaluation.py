#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Runs ROUGE evaluation on the summary file results."""


__author__ = 'Haley Lepp'
__email__ = 'hlepp@uw.edu'


import os
import subprocess
from subprocess import Popen, PIPE


def edit_xml_dev(output_folder):
    default_xml = open('src/ROUGE/revised_config.xml', 'r')
    config_file = "src/ROUGE/revised_config_" + output_folder + ".xml"
    new_xml = open(config_file, "w+")    
    if not os.path.exists('outputs/' + output_folder):
        os.mkdir(output_folder)
    for line in default_xml:
        if line == 'outputs/D2\n':
            new_xml.write('outputs/' + output_folder + '\n')
        else:
            new_xml.write(line)
    return config_file


def edit_xml_eval(output_folder):
    default_xml = open('src/ROUGE/eval_config.xml', 'r')
    config_file = 'src/ROUGE/revised_config_' + output_folder \
        + '.xml'
    new_xml = open(config_file, 'w+')
    if not os.path.exists('outputs/' + output_folder):
        os.mkdir(output_folder)
    for line in default_xml:
        if line \
            == '/home/tac/tac2011/Summarization/eval/peers/peers_segmented/peers_A\n':
            new_xml.write('outputs/' + output_folder + '\n')
        elif line == '/dropbox/14-15/573/Data/models/evaltest\n':
            new_xml.write('/dropbox/18-19/573/Data/models/evaltest/\n')
        else:
            new_xml.write(line)
    return config_file


def eval_summary(output_folder, data_type):
    if data_type == 'dev':
        CONFIG_FILE_WITH_PATH = edit_xml_dev(output_folder)
    elif data_type == 'eval':
        CONFIG_FILE_WITH_PATH = edit_xml_eval(output_folder)
    else:
        raise ValueError('First argument must be eval or dev')      
    ROUGE_DATA_DIR = '/dropbox/18-19/573/code/ROUGE/data'
    COMMAND = 'src/ROUGE/ROUGE-1.5.5.pl -e ' + ROUGE_DATA_DIR \
        + ' -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d ' \
        + CONFIG_FILE_WITH_PATH
    stdout = subprocess.check_output(COMMAND.split())
    output_file = "results/" + output_folder +  "_rouge_scores.out"
    results = open(output_file, 'w+')
    results.write(stdout.decode())
    results.close()
