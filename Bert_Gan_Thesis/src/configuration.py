import os
import sys


#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Holds Settings object that reads in parameters and required parts

###### Acknowledgments ###############################################################################################################
#This Settings object was built upon from code used in a my group project for UW LING 575 Winter 2020: Analyzing Neural Language Models.
######################################################################################################################################

class Settings:
    debug_folder = '../debug/'
    debug_suffix = '.txt'
    config_folder = '../configs/'
    config_suffix = '.txt'
    datasets_folder = '../datasets/'
    dataset_suffix = '.txt'
    resource_folder = '../rec/'
    output_folder = '../out/'
    models_folder = '../models/'
    pickel_suffix = '.pkl'


    def __init__(self):
        if len(sys.argv) <= 1:
            raise ValueError('Error: must specify an experiment config')
        config_name = sys.argv[1]

        current_folder = os.path.dirname(__file__)

        if len(current_folder):
            current_folder += '/'

        # Verify config file exists.
        config_path = current_folder + Settings.config_folder + config_name + Settings.config_suffix

        if not os.path.exists(config_path):
            raise ValueError(
                'Error: experiment config file not found: {0}'.format(
                    os.path.abspath(config_path)
                )
            )
        # Store config settings as string=>string map.
        # Individual settings can be parsed via getter functions in class Settings.
        self.__settings_map = {}
        with open(config_path, 'r') as f:
            for setting in f.readlines():
                name = setting.split()[0]
                value = setting.split()[1]
                self.__settings_map[name] = value

        # Overwrite the debug output file from prior run, if one exists.
        debug_path = current_folder + Settings.debug_folder + config_name + Settings.debug_suffix
        with open(debug_path, 'a+') as f:
            f.write('Initializing debugger output file...\n')
        # Store debug path, for writing debugger output during execution.
        self.__debug_path = debug_path

        # Store dataset path, for loading training/test data
        self.__dataset_path = current_folder + Settings.datasets_folder

        # Store dataset path, for loading training/test data
        self.__resource_path = current_folder + Settings.resource_folder + config_name + "."

        # Store dataset path, for loading training/test data
        self.__output_path = current_folder + Settings.output_folder + config_name + "."

        self.__models_path = current_folder + Settings.models_folder + config_name + "/"

        if not os.path.exists(self.__models_path):
            os.makedirs(self.get_generator_path())
            os.makedirs(self.get_discriminator_path())



    def write_debug(self, msg: str):
        with open(self.__debug_path, 'a') as f:
            f.write(msg + "\n")
            f.flush()
    def write_result(self, msg: str):
        with open(self.get_eval_results_out_path(), 'a') as f:
            f.write(msg + "\n")
            f.flush()
    def write_train_stat(self, msg: str):
        with open(self.get_train_stats_out_path(), 'a') as f:
            f.write(msg + "\n")
            f.flush()

    # Getter Methods
    def get_dataset_path(self) -> str:
        return self.__dataset_path

    def get_model_type(self) -> str:
        return self.__settings_map['model_type']
    def get_generator_path(self) -> str:
        return self.__models_path + self.__settings_map['generator_path']
    def get_discriminator_path(self) -> str:
        return self.__models_path + self.__settings_map['discriminator_path']

    def get_num_labels(self) -> int:
        return int(self.__settings_map['num_labels'])

    def get_batch_size(self) -> int:
        return int(self.__settings_map['batch_size'])

    def get_train_epochs(self) -> int:
        return int(self.__settings_map['train_epochs'])

    def get_random_state(self) -> int:
        return int(self.__settings_map['random_state'])
    def get_sample_size(self) -> int:
        return int(self.__settings_map['sample_size'])
    def get_num_batches(self) -> int:
        return int(self.__settings_map['num_batches'])
    def get_min_sample_len(self) -> int:
        return int(self.__settings_map['min_sample_len'])
    def get_test_size_ratio(self) -> float:
        return float(self.__settings_map['test_size_ratio'])

    def get_raw_wiki_path(self) -> str:
        return self.__dataset_path + str(self.__settings_map['raw_wiki_path'] + self.dataset_suffix)
    def get_raw_tbc_path(self) -> str:
        return self.__dataset_path + str(self.__settings_map['raw_tbc_path']+ self.dataset_suffix)
    def get_train_inputs_path(self) -> str:
        return self.__resource_path + str(self.__settings_map['train_inputs_path'])
    def get_validation_inputs_path(self) -> str:
        return self.__resource_path + str(self.__settings_map['validation_inputs_path'])

    def get_proc_wiki_path(self) -> str:
        return self.__resource_path + str(self.__settings_map['proc_wiki_path']) + self.pickel_suffix
    def get_proc_tbc_path(self) -> str:
        return self.__resource_path + str(self.__settings_map['proc_tbc_path']) + self.pickel_suffix

    def get_bert_prelim_out_path(self) -> str:
        return self.__output_path + str(self.__settings_map['bert_prelim_out_path'])
    def get_gpt_prelim_out_path(self) -> str:
        return self.__output_path + str(self.__settings_map['gpt_prelim_out_path'])
    def get_eval_results_out_path(self) -> str:
        return self.__output_path + str(self.__settings_map['eval_results_out_path'])
    def get_train_stats_out_path(self) -> str:
        return self.__output_path + str(self.__settings_map['train_stats_out_path'])


    def get_bert_eval_out_path(self) -> str:
        return self.__output_path + str(self.__settings_map['bert_eval_out_path'])
    def get_bert_train_out_path(self) -> str:
        return self.__output_path + str(self.__settings_map['bert_train_out_path'])

    def get_bert_valid_out_path(self) -> str:
        return self.__output_path + str(self.__settings_map['bert_valid_out_path'])

    def get_num_eval_samples(self) -> int:
        return int(self.__settings_map['num_eval_samples'])

    def get_eval_batch_size(self) -> int:
        return int(self.__settings_map['eval_batch_size'])
    def get_eval_top_k(self) -> int:
        return int(self.__settings_map['eval_top_k'])
    def get_eval_temp(self) -> float:
        return float(self.__settings_map['eval_temp'])

    def get_eval_burnin(self) -> int:
        return int(self.__settings_map['eval_burnin'])

    def get_eval_gen_mode_key(self) -> str:
        return str(self.__settings_map['eval_gen_mode_key'])

    def get_eval_sample(self) -> bool:
        return bool(self.__settings_map['eval_sample'])

    def get_eval_max_iter(self) -> int:
        return int(self.__settings_map['eval_max_iter'])

    def get_eval_seed_text(self) -> str:
        return str(self.__settings_map['eval_seed_text'])

    def get_bleu_max_n(self) -> int:
        return int(self.__settings_map['bleu_max_n'])


#Object instantiated when imported
settings=Settings()

