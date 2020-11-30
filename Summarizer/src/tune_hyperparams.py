#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Script to run different hyperparameter settings on text_summarier.py"""

__author__ = 'Shannon Ladymon'
__email__ = 'sladymon@uw.edu'


from text_summarizer import summarize_text, summarize_topics_list
from data_input import get_data


def read_results(output_folder):
    """
    Reads in the rouge score results and prints the R-1 and R-2 values
    """
    results_file = 'results/' + output_folder + '_rouge_scores.out'

    with open(results_file) as f:
        for line in f:
            if "ROUGE-1 Average_R:" in line: 
                split_line = line.split()
                r1_val = float(split_line[3])
            if "ROUGE-2 Average_R:" in line: 
                split_line = line.split()
                r2_val = float(split_line[3])
        print("{}\tR-1={}\tR-2={}".format(output_folder, r1_val, r2_val))


if __name__ == '__main__':

    # Tune parameters on devtest dataset
    input_path = "/dropbox/18-19/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml" 


    ###### Original defaults to start with ######

    # D2 defaults
    stemming_default = False
    lower_default = False
    include_narrative_default = False
    idf_type_default = "smooth_idf"
    tf_type_default = "term_frequency"
    d_default = 0.7
    summary_threshold_default = 0.5
    epsilon_default = 0.1
    min_sent_len_default = 5
    intersent_threshold_default = 0.0

    # 2009 defaults
    mle_lambda_default = 0.6
    k_default = 20
    info_order_type = "chron"
    num_permutations_default = 11

    ###### Options for each parameter to test ######

    # Test for data input for all configurations
    stemming_opts = [False, True] 
    lower_opts = [False, True] 
    idf_type_opts = ["smooth_idf", "probabilistic_idf", "standard_idf"]
    tf_type_opts = ["term_frequency", "log_normalization"]

    # Test for content selection for all configurations
    d_opts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    summary_threshold_opts =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    epsilon_opts = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20] 
    min_sent_len_opts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Test only for cosine (form_D2, form_2005)
    include_narrative_opts = [False, True]
    intersent_threshold_opts =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Test only for gen/norm (form_2009)
    mle_lambda_opts =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k_opts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Test only for enitity
    info_order_type_opts = ["chron", "entity"]
    num_permutations_opts = [5, 10, 15, 20, 25, 30]

    ###### TUNED VALUES ######

    # TUNED INPUT DATA PARAMETER VALUES
    stemming_tuned = True
    lower_tuned = False 
    idf_type_tuned = "smooth_idf" # Note this is slightly lower for config D2 & 2005, but much better for 2009
    tf_type_tuned = "term_frequency"

    # TUNED CONTENT SELECTION PARAMETER VALUES
    d_tuned_dict = {"form_D2": 0.1, "form_2005": 0.2, "form_2009": 0.3}
    summary_threshold_tuned_dict = {"form_D2": 0.4, "form_2005": 0.3, "form_2009": 0.4}
    epsilon_tuned_dict = {"form_D2": 0.02, "form_2005": 0.04, "form_2009": 0.1}
    min_sent_len_tuned_dict = {"form_D2": 3, "form_2005": 3, "form_2009": 4}

    # TUNED CONFIG_D2 and CONFIG_2005 PARAMETER VALUES
    include_narrative_tuned = False
    intersent_threshold_tuned_dict = {"form_D2": 0.0, "form_2005": 0.1, "form_2009": 0.0} 

    # TUNED CONFIG_2009 PARAMETER VALUES
    mle_lambda_tuned = 0.6
    k_tuned = 9

    # TUNED INFO ORDERING PARAMTER VALUES
    num_permutations_tuned_dict = {"form_D2": 5, "form_2005": 5, "form_2009": 20}

    ###### Get the Data #####

    topics = get_data(input_path, stemming_tuned, lower_tuned, idf_type_tuned, tf_type_tuned)


    # Run tests for each of the three formula configurations

    config_D2 = ["form_D2", "cos", "cos"]  # what we ran in D2 
    config_2005 = ["form_2005", "rel", "cos"]  # the formulas from the Otterbacher 2005 paper 
    config_2009 = ["form_2009", "gen", "norm"]  # the formulas from the Otterbacher 2009 paper
    config_list = [config_D2, config_2005, config_2009]

    for config in config_list:
        form = config[0] 
        bias_formula = config[1] 
        intersent_formula = config[2]

        # Get params tuned for each configuration
        d_tuned = d_tuned_dict[form]
        summary_threshold_tuned = summary_threshold_tuned_dict[form]
        epsilon_tuned = epsilon_tuned_dict[form]
        min_sent_len_tuned = min_sent_len_tuned_dict[form]
        intersent_threshold_tuned = intersent_threshold_tuned_dict[form]
        num_permutations_tuned = num_permutations_tuned_dict[form]

        # NOTE: To run tuning, choose the option list to try out, 
        # and replace the variable with the new val
        # in the output_folder and in the summarize_topics_list call
        # Currently, this is set up to test values for info_order_type

        # Run all values to test
        for info_order_val in info_order_type_opts:

            output_folder = "D3_tune_" + form + "_info_order_" + str(info_order_val)

            summarize_topics_list(topics, output_folder, d_tuned, intersent_threshold_tuned, summary_threshold_tuned, epsilon_tuned, mle_lambda_tuned, k_tuned, min_sent_len_tuned, include_narrative_tuned, bias_formula, intersent_formula, info_order_val, num_permutations_tuned)

            read_results(output_folder)

        print()  # Print newline to separate

