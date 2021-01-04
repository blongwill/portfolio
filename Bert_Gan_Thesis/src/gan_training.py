#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#File holds methods related to the training loop for GAN
#Can be used by calling .train_gan(training_dataloader, validation_dataloader) on GAN object

###### Acknowledgments ##########################
#-function train_gan flat_accuracy and format_time code was modified from:
#       @misc{mccormick_ryan_2019, title={BERT Fine-Tuning Tutorial with PyTorch},
#       url={https://mccormickml.com/2019/07/22/BERT-fine-tuning}, author={McCormick, Chris and Ryan, Nick}, year={2019}, month={Jul}}
#-function train_gan code was also influenced from:
#       @misc{dc_gan, title={DCGAN TUTORIAL},
#       url={https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html}, author={Inkawhich, Nathan}, year={2017}}
###############################


#Imported libraries
import torch
import datetime
from utility_models import settings, tokenizer, device
import numpy as np
import random
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import time


#Set random seeds all to same value for reproducibility
seed_val = settings.get_random_state()
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


#method that contains the training loop. Requires training and validation lists casted to dataloader objects.
## In later version this should be corrected
def train_gan(self, training_dataloader: DataLoader, validation_dataloader: DataLoader):
    settings.write_debug("Begin Training")

    self.initialize_training(training_dataloader)

    settings.write_debug("========================================")
    settings.write_debug("               Training                 ")
    settings.write_debug("========================================")

    # Training loop -- Iterates over full training dataset
    for epoch_i in range(0, self.epochs):

        settings.write_debug("")
        settings.write_debug('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
        settings.write_debug('Training...')

        # Initializes time variable to measure duration of training
        t0 = time.time()

        # Reset the loss variables for each epoch
        discriminator_total_loss = 0
        generator_total_loss = 0
        gen_update_step = 0

        ########################################################
        # Get one batch of real samples from dataset at a time
        #######################################################
        for step, batch in enumerate(training_dataloader):


            ########################
            # Generate Labels
            ########################
            #   Set labels using random float values for label smoothing
            #       - Real labels are range [.9,1] instead of just 1
            #       - Fake labels are range [0,.1] instead of just 0
            #       - Helps to prevent mode collapse by keeping a moving target
            self.real_labels = torch.tensor([random.uniform(.9, 1)] * settings.get_batch_size(),
                                            requires_grad=False).unsqueeze(-1).to(device)
            self.false_labels = torch.tensor([random.uniform(0, .1)] * settings.get_batch_size(),
                                             requires_grad=False).unsqueeze(-1).to(device)

            settings.write_debug("batch:" + str(step) + "/" + str(len(training_dataloader)))

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                settings.write_debug(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(training_dataloader), elapsed))


            ###########################
            # Discriminator Network Training
            #############################
                # - Real samples only to maximize log(D(x)) + log(1 - D(G(z)))
            self.discriminator.train()
            self.generator.train()
            self.discriminator.requries_grad = True

            #clears previously accumulated gradients by setting to all zeros
            self.discriminator.zero_grad()

            #attaches batch to GPU
            batch = batch.to(device)

            ## Trains with batch of samples from the dataset -- "real"
            D_x,  discriminator_real_loss = self.discriminator.train_discriminator(batch, self.real_labels)  ### Adds the correct list of labels to the batch

            ############################################################################
            ## Generate a "fake" batch of samples similar in size and length to the "real"
            ##########################################################################
            n_samples = batch_size = settings.get_batch_size()
            sample_lens = ((batch != 0).sum(dim=-1) - 2).tolist()  # Subtract 2 to account for CLS and final SEP
            max_len = len(batch[0]) - 2  # settings.get_sample_size()
            top_k = 10  ### *** Don't Use 1 here because it breaks computation ***
            temperature = 1.0
            generation_mode = "training"
            burnin = 0
            sample = True
            max_iter = 1

            # Choose the prefix context
            seed_text = "[CLS]".split()
            generated_input_ids = self.generator.generate(n_samples, seed_text=seed_text, batch_size=batch_size,
                                                          max_len=max_len, sample_lens=sample_lens,
                                                          generation_mode=generation_mode,
                                                          sample=sample, top_k=top_k, temperature=temperature,
                                                          burnin=burnin,
                                                          max_iter=max_iter)

            bert_sents = [self.generator.detokenize(tokenizer.convert_ids_to_tokens(sent.tolist())).split() for sent in generated_input_ids]
            with open(settings.get_bert_train_out_path(), 'a+') as f:
                [f.write('[%d/%d][%d/%d]\t'% (epoch_i, self.epochs, index, len(bert_sents)) + " ".join(sent) + "\n") for index, sent in enumerate(bert_sents)]

            D_G_z1, discriminator_fake_loss = self.discriminator.train_discriminator(generated_input_ids.clone().detach(), self.false_labels)  ### Adds the correct list of labels to the batch


            discriminator_combined_loss = discriminator_fake_loss + discriminator_real_loss

            discriminator_total_loss += discriminator_combined_loss


            self.discriminator.optimizer.step()
            # Update the learning rate.
            self.discriminator_scheduler.step()


            ###########################
            # Discriminator Network Training
            #############################
                # - Generated samples only to maximizes log(D(G(z)))

            #Save gpu memory by untracking discriminator gradients
            self.discriminator.requries_grad = False

            ## Train with generated batch
            D_G_z2, generator_loss = self.generator.train_generator(generated_input_ids, self.real_labels, self.discriminator)

            #Call step to optimizer to update weights
            #Step to scheduler modifies learning rate
            self.generator.optimizer.step()
            self.generator_scheduler.step()
            generator_total_loss += generator_loss

            #counter
            gen_update_step += 1


            #Clear gradients after update, Detach and empty cache to save on memory
            self.generator.zero_grad()
            generated_input_ids.detach()
            del generated_input_ids
            torch.cuda.empty_cache()

            # Output training stats
            settings.write_train_stat('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\n'% (epoch_i, self.epochs, step, len(training_dataloader),
                         discriminator_combined_loss, generator_loss, D_x, D_G_z1, D_G_z2))

        # Calculate the average loss over the training data.
        discriminator_avg_train_loss = discriminator_total_loss / (len(training_dataloader))
        generator_avg_train_loss = generator_total_loss / gen_update_step


        ### output statistics to file
        settings.write_train_stat("\n")
        settings.write_train_stat("  Average Discriminator training loss: {0:.2f}".format(discriminator_avg_train_loss))
        settings.write_train_stat("  Average Generator training loss: {0:.2f}".format(generator_avg_train_loss))
        settings.write_train_stat("  Training epcoh took: {:}\n".format(format_time(time.time() - t0)))



        settings.write_debug("========================================")
        settings.write_debug("              Validation                ")
        settings.write_debug("========================================")
        # After training epoch, measure our accuracy using mixture of validation set from real dataset and more generated samples to match

        settings.write_debug("")
        settings.write_debug("Running Validation...")

        t0 = time.time()

        #eval changes behavior of dropout layers
        self.generator.eval()
        self.discriminator.eval()

        # reset variables
        eval_loss, eval_accuracy = 0., 0.
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            with torch.no_grad():

                ## Generate an all-fake batch

                sample_lens = ((batch != 0).sum(dim=-1) - 2).tolist()  # Subtract 2 to account for CLS and final SEP

                n_samples = batch_size = settings.get_batch_size()
                max_len = len(batch[0]) - 2  # settings.get_sample_size()
                top_k = 100  ### Using 1 here seems to break computation for some reason
                temperature = 1.0
                generation_mode = "evaluation"
                burnin = 250
                sample = True
                max_iter = 500
                seed_text = "[CLS]".split()
                generated_input_ids = self.generator.generate(n_samples, seed_text=seed_text, batch_size=batch_size,
                                                              max_len=max_len,
                                                              generation_mode=generation_mode,
                                                              sample=sample, top_k=top_k, temperature=temperature,
                                                              burnin=burnin,
                                                              max_iter=max_iter, sample_lens=sample_lens)

                validation_sents = [self.generator.detokenize(tokenizer.convert_ids_to_tokens(sent.tolist())).split() for sent in generated_input_ids]
                with open(settings.get_bert_valid_out_path(), 'a+') as f:
                    [f.write('[%d/%d][%d/%d]\t' % (epoch_i, self.epochs, index, len(validation_sents)) + " ".join(sent) + "\n") for index, sent in enumerate(validation_sents)]



                batch = torch.cat((batch, generated_input_ids)).to(device)
                labels = torch.cat((self.real_labels, self.false_labels))

                #Matches labels with samples and shuffles them accordingly
                batch = list(zip(batch, labels))
                random.shuffle(batch)
                batch, labels = zip(*batch)

                logits, _ = self.discriminator(torch.stack(batch))


                print("Validation logits", flush=True)
                print(logits, flush=True)
                # Calculate the acc for this batch of mixed real and generated validation

                tmp_eval_accuracy = self._flat_accuracy(preds=logits, labels=torch.stack(labels))

                # save total acc
                eval_accuracy += tmp_eval_accuracy

                # total number of validation batches run
                nb_eval_steps += 1

                print(eval_accuracy / nb_eval_steps)

        # Output acc
        settings.write_train_stat("\n")
        settings.write_train_stat("  Validation Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        settings.write_train_stat("  Validation took: {:}\n".format(format_time(time.time() - t0)))

        #Put both models back into training mode now that validation has ended
        self.generator.train()
        self.discriminator.train()

    settings.write_debug("")
    settings.write_debug("Training complete!")

    # Function to calculate the accuracy of our predictions vs labels

def _flat_accuracy(preds, labels):
    #round the predictions and and flatten them to one vector
    pred_flat = torch.abs(torch.round(torch.sigmoid(preds.float()))).flatten()
    labels_flat = torch.abs(torch.round(labels)).flatten()
    return torch.sum(pred_flat == labels_flat).item() / len(labels_flat)

def initialize_training(self, training_dataloader):
    # Initializes variables for training session

    self.epochs = settings.get_train_epochs()

    total_steps_discriminator = len(
        training_dataloader) * self.epochs  # Only will update once for LR for REal + Fake

    total_steps_generator = len(
        training_dataloader) * self.epochs  # Only for generated sentences which matches the number of real sentences

    self.discriminator_scheduler = get_linear_schedule_with_warmup(self.discriminator.optimizer,
                                                                   num_warmup_steps=0,
                                                                   # Default value in run_glue.py
                                                                   num_training_steps=total_steps_discriminator)

    self.generator_scheduler = get_linear_schedule_with_warmup(self.generator.optimizer,
                                                               num_warmup_steps=0,  # Default value in run_glue.py
                                                               num_training_steps=total_steps_generator)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
