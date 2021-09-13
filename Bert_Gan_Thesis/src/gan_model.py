#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Benny Longwill
#12/20/20
#For use as part of BERT GAN thesis project
#Class file holds Generator and Discriminator subclasses as part of GAN Class
#A new GAN is instantiated using class name (GAN) followed by open and close parens (i.e., GAN())
#Separately Generator/Discriminator objects have the possiblity of being initialized with a previous model file (i.e., Generator(model_file_path:str)) -- Mostly used for eval

'''
###### Acknowledgments ##########################

- The following methods were modified from Wang & Cho (2019)
            __generate_step : added gumbel softmax functionality for backpropogation which changed the sampling procedure
            __get_init_text : samples samples of variable length within a batch and completed ranom token sequence functionality
            generate : modified by changing the generation procedures
             __training_generation : modified in order to allow for signel pass training generation and also for backpropogation of gradients. Also added generation of a group of indicies of a single pass.

- The following methods were applied directly from Wang & Cho (2019) due to reproducibility of results:
           _eval_generate_step, __evaluation_generation, tokenize_batch, untokenize_batch, and detokenize

       @inproceedings{wang2019bert,
        #title = "{BERT} has a Mouth, and It Must Speak: {BERT} as a {M}arkov Random Field Language Model",
        #author = "Wang, Alex  and  Cho, Kyunghyun", month = jun, year = "2019"
        #https://github.com/nyu-dl/bert-gen/blob/master/bert-babble.ipynb

###############################
'''
#Dependenciest
import torch.nn as nn
from utility_models import settings, tokenizer, device
import numpy as np
import random
from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM, AdamW
import math, time
import gc
import torch

# Set the seed value all over the place to make this reproducible.
seed_val = settings.get_random_state()
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#GAN object contains a generator and discriminator for encapsulation of modules
class GAN(nn.Module):

    def __init__(self, gen_file_path:str=None, disc_file_path:str=None):
        super(GAN, self).__init__()
        settings.write_debug("Initializing GAN")

        self.generator = GAN.Generator(gen_file_path)
        self.discriminator = GAN.Discriminator(disc_file_path)

    #Import object attributes from file
    from gan_training import initialize_training, train_gan, format_time, _flat_accuracy

    #Generator objects has the possiblity of being initialized with a model file
    class Generator(nn.Module):
        def __init__(self, path=None):
            super(GAN.Generator, self).__init__()
            settings.write_debug("Initializing generator")

            if path is not None:
                settings.write_debug("Getting model from file")
                self.map1 = BertForMaskedLM.from_pretrained(path)
            else:
                self.map1 = BertForMaskedLM.from_pretrained(settings.get_model_type(), output_attentions=False,
                                                        output_hidden_states=True)


            #Loss function and optimizer used for generation -- Attributes bc this could be changed to be different from discriminator
            self.loss_fct = nn.BCEWithLogitsLoss()
            self.optimizer = AdamW(self.parameters(),
                                   lr=2e-6,
                                   eps=10e-4
                                   )
        ### Forward pass on generator is called
        def forward(self, batch, labels, discriminator):

            #Puts the discriminator into evaluation mode only for getting logits here
            discriminator.eval()
            discriminator_logits, _ = discriminator(batch)
            discriminator.train()

            #### Reparameterization Trick used to work around the discrete labels going into discriminator
            loss = self.loss_fct((discriminator_logits - batch.sum(-1).unsqueeze(-1)).detach() + batch.sum(-1).unsqueeze(-1), labels)

            return discriminator_logits.mean().item(), loss

        @staticmethod
        def tokenize_batch(batch):
            return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

        @staticmethod
        def untokenize_batch(batch, tokenizer: BertTokenizer):
            return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

        @staticmethod
        def detokenize(sent):
            """ Roughly detokenizes (mainly undoes wordpiece) """

            new_sent = []
            for i, tok in enumerate(sent[1:-1]):
                if tok.startswith("##") and len(new_sent) > 0:
                    new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
                elif tokenizer.pad_token not in tok and tokenizer.sep_token not in tok:
                    new_sent.append(tok)

            return " ".join(new_sent)

        #This method was modified by removing the for loop and allowing for training generation from bert in a single pass
        #Additionally the gumbel softmax was implemented in order to sample in a differentiable procedure
        def __generate_step(self, bert_output: torch.tensor, gen_idx, temperature=None, top_k=0, sample=False,
                            return_list=True):
            #Generates a singl word token from bert propability distribution at gen_idx
            #Temperature controls the uniformity of the given probability distribution
            #Topk is the list of top probabilities in the distribution

            logits = bert_output[:, gen_idx]

            if temperature is not None:
                logits = logits / temperature
            if top_k > 0:
                ###### topk=1 is the same as using argmax. Caution! Breaks computation bc non-differentiable
                #Gumbel Softmax is implemented to allow for sampling with backprop
                kth_vals, kth_idx = torch.topk(logits, top_k, dim=-1)
                gumbel_reps = torch.nn.functional.gumbel_softmax(logits=kth_vals, hard=True)
                idx = torch.sum(torch.mul(gumbel_reps, kth_idx), dim=-1)

            elif sample:
                gumbel_reps = torch.nn.functional.gumbel_softmax(logits=logits, hard=True)
                idx = torch.sum(torch.mul(gumbel_reps, torch.tensor(range(0, len(tokenizer.vocab))).to(device)), dim=-1)
            else:
                #Argmax does not allow for backpropogation
                idx = torch.argmax(logits, dim=-1)
                settings.write_debug("ERRORRRRR!!!!!!")

            logits.detach()
            del logits
            gumbel_reps.detach()
            del gumbel_reps

            gc.collect()

            return idx.tolist() if return_list else idx

        #This method was modified to include variable smaple lengths within a batch
        def __get_init_text(self, seed_text, max_len, batch_size=1, rand_init=False, sample_lens=None):
            #Creates starting vector of either masked tokens or random words tokens
            #Sample lens is a list of sequence lengths for the batch to allow for variable length sequences that require padding
            if sample_lens:
                batch = [
                    seed_text + [tokenizer.mask_token] * sample_len + [tokenizer.sep_token] + [tokenizer.pad_token] * (
                                max_len - sample_len) for sample_len in sample_lens]
            else:
                batch = [seed_text + [tokenizer.mask_token] * max_len + [tokenizer.sep_token] for _ in
                         range(batch_size)]

            batch = GAN.Generator.tokenize_batch(batch)

            if rand_init:
                for batch_idx in range(batch_size):
                    batch[batch_idx][len(seed_text):max_len + 1] = random.sample(range(0, len(tokenizer.vocab)),
                                                                                 max_len)

            return batch

        @staticmethod
        def _eval_generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
            """ Generate a word from from out[gen_idx]

            args:
                - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
                - gen_idx (int): location for which to generate for
                - top_k (int): if >0, only sample from the top k most probable words
                - sample (Bool): if True, sample from full distribution. Overridden by top_k
            """
            logits = out[:, gen_idx]
            if temperature is not None:
                logits = logits / temperature
            if top_k > 0:
                kth_vals, kth_idx = logits.topk(top_k, dim=-1)
                dist = torch.distributions.categorical.Categorical(logits=kth_vals)
                idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
            elif sample:
                dist = torch.distributions.categorical.Categorical(logits=logits)
                idx = dist.sample().squeeze(-1)
            else:
                idx = torch.argmax(logits, dim=-1)
            return idx.tolist() if return_list else idx

        def __evaluation_generation(self, seed_text, batch_size=10, max_len=15, top_k=0, temperature=None,
                                    max_iter=300, burnin=200, sample_lens=None,
                                    cuda=False, print_every=10, verbose=True):
            """ Generate for one random position at a timestep

            args:
                - burnin: during burn-in period, sample from full distribution; afterwards take argmax
            """
            seed_len = len(seed_text)
            batch = torch.tensor(
                self.__get_init_text(seed_text, max_len, batch_size, rand_init=False, sample_lens=sample_lens),
                requires_grad=False)

            for ii in range(max_iter):
                kk = np.random.randint(0, max_len)
                for jj in range(batch_size):
                    if sample_lens is None or (sample_lens and seed_len + kk <= sample_lens[
                        jj]):  #### If the index is less than the max len
                        batch[jj][seed_len + kk] = tokenizer.mask_token_id
                inp = torch.as_tensor(batch).to(device)
                out = self.map1(inp)[0]
                topk = top_k if (ii >= burnin) else 0
                idxs = GAN.Generator._eval_generate_step(out, gen_idx=seed_len + kk, top_k=topk,
                                                         temperature=temperature, sample=(ii < burnin))

                for jj in range(batch_size):
                    if sample_lens is None or (sample_lens and seed_len + kk <= sample_lens[
                        jj]):  #### If the index is less than the max len
                        batch[jj][seed_len + kk] = idxs[jj]

            return batch

        #This method was modified such that the a loop was removed in order to allow for signel pass training generation.
        #This method also had modifications that allow for backpropogation of gradients
        def __training_generation(self, seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300,
                                  burnin=200, sample_lens=None, print_every=10, verbose=True):
            #This method in the final project samples all tokens at once from the distribution created from just a single pass through bert

            seed_len = len(seed_text)

            noise_input = torch.tensor(
                self.__get_init_text(seed_text, max_len, batch_size, rand_init=False, sample_lens=sample_lens),
                requires_grad=True, dtype=torch.float, device=device)

            batch = noise_input.clone()

            #the number of groups that re-use the same probility distribution from one pass through BERt
            #In the case of the current study, the whole vector (each index) is unmasked from a single pass
            num_groups = 1

            for ii in range(max_iter):

                raw_idxs = np.array(range(0, max_len))
                random.shuffle(raw_idxs)
                rand_idxs = np.array_split(raw_idxs, num_groups)

                #A group represents 1 or more randomly selected indicies from the input vector
                for group in rand_idxs:

                    out = self.map1(batch.long())[0]

                    for kk in group:

                        for jj in range(batch_size):
                            if sample_lens is None or (sample_lens and seed_len + kk <= sample_lens[
                                jj]):  #### If the index is less than the max len
                                batch[jj][seed_len + kk] = tokenizer.mask_token_id

                        topk = top_k if (ii >= burnin) else 0
                        idxs = self.__generate_step(out, gen_idx=seed_len + kk, top_k=topk, temperature=temperature,
                                                    sample=(ii < burnin), return_list=False)

                        for jj in range(batch_size):
                            # batch[jj].index_fill_(-1, torch.tensor(seed_len + kk).to(device), idxs[jj])
                            if sample_lens is None or (sample_lens and seed_len + kk <= sample_lens[
                                jj]):  #### If the index is less than the max len
                                batch[jj][seed_len + kk] = idxs[jj]

                    out.detach()
                    del out

            return batch

        #This method was modified to include only two generation procedures and to allow for backpropogation of gradients
        def generate(self, n_samples, seed_text="[CLS]", batch_size=10, max_len=25, sample_lens=None,
                     generation_mode="parallel-sequential",
                     sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500, print_every=1):
            # main generation function to call
            settings.write_debug("Generating Sentences")
            sentences = []
            n_batches = math.ceil(n_samples / batch_size)
            start_time = time.time()
            for batch_n in range(n_batches):

                if generation_mode == "evaluation":
                    batch = self.__evaluation_generation(seed_text, batch_size=batch_size, max_len=max_len,
                                                         top_k=top_k,
                                                         temperature=temperature, burnin=burnin,
                                                         max_iter=max_iter, sample_lens=sample_lens, verbose=False)

                elif generation_mode == "training":
                    batch = self.__training_generation(seed_text, batch_size=batch_size, max_len=max_len,
                                                       top_k=top_k,
                                                       temperature=temperature, burnin=burnin,
                                                       max_iter=max_iter, sample_lens=sample_lens, verbose=False)

                if (batch_n + 1) % print_every == 0:
                    settings.write_debug(
                        "Finished generating batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
                    start_time = time.time()

                sentences += batch
                del batch
                torch.cuda.empty_cache()

            settings.write_debug("Returning Generated Sentences")
            return torch.stack(sentences)

        def train_generator(self, batch, labels, discriminator):
            settings.write_debug("Begin training generator")

            output, loss = self(batch, labels, discriminator)

            generator_loss = loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            settings.write_debug("Finish training generator")
            return output, generator_loss

    class Discriminator(nn.Module):

        def __init__(self, path=None):
            super(GAN.Discriminator, self).__init__()
            settings.write_debug("Initializing discriminator object")

            if path is not None:
                settings.write_debug("Getting model from file")
                self.map1 = BertForSequenceClassification.from_pretrained(path)
            else:
                # Load BertForSequenceClassification, the pretrained BERT model with a single
                # linear classification layer on top.
                self.map1 = BertForSequenceClassification.from_pretrained(
                    settings.get_model_type(),  # Use the 12-layer BERT model, with an uncased vocab.
                    # num_labels=settings.get_num_labels(),  # The number of output labels--2 for binary classification.
                    num_labels=1,
                    # You can increase this for multi-class tasks.
                    output_attentions=False,  # Whether the model returns attentions weights.
                    output_hidden_states=False,  # Whether the model returns all hidden-states.
                )

            #Discriminator loss and optimizer are attributes of this object so that they can be different Generator objects if necessary
            self.loss_f = nn.BCEWithLogitsLoss()
            self.optimizer = AdamW(self.parameters(),
                                   lr=2e-5,
                                   eps=1e-8
                                   )

        def forward(self, batch, labels=None):
            output = self.map1(input_ids=batch.long())[0]

            if batch is not None:
                loss = self.loss_f(output, labels)
            else:
                loss=None

            return output, loss

        def train_discriminator(self, batch, labels):
            settings.write_debug("Begin Train Discriminator")

            output, loss = self(batch, labels)

            #Backward pass to accumulate the gradients
            loss.backward()
            #Clipts gradients that rise above 1.0 to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            settings.write_debug("Ending Train Discriminator")
            return output.mean().item() , loss
