#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Module designed to instantiate and train a GAN object by providing configuration file
#train.py can be run from the base directory by calling ./src/train.py config_file_name (not including .txt extension)

#Dependencies
from utility_models import device
from configuration import settings
from data_collection import create_smart_batches
from gan_model import GAN

def main():

    #Instantiates GAN and attaches to GPU for training
    gan = GAN().to(device)

    #Context Manager for training and validation dataset files
    with open(settings.get_train_inputs_path()) as train_f, open(settings.get_validation_inputs_path()) as validation_f :

        #Uses the creates smart_batches from data_collection.py to create batches of equal size given the training and validation datasets
        train_inputs=create_smart_batches([x.split() for x in train_f.read().splitlines()], settings.get_batch_size())
        validation_inputs=create_smart_batches([x.split() for x in validation_f.read().splitlines()], settings.get_batch_size())

        #Begins the Gan training
        gan.train_gan(train_inputs,validation_inputs)

        #Saves the models to specified file paths for later evaluation
        gan.generator.map1.save_pretrained(settings.get_generator_path())
        gan.discriminator.map1.save_pretrained(settings.get_discriminator_path())

if __name__ == "__main__":
    main()
