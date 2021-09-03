def __generate_step(self, out: torch.tensor, gen_idx, temperature=None, top_k=0, sample=False,
                    return_list=True):
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

        ###### I think something is happening in gumbel softmax that breaks computation graph if topk==1
        kth_vals, kth_idx = torch.topk(logits, top_k, dim=-1)
        gumbel_reps = torch.nn.functional.gumbel_softmax(logits=kth_vals, hard=True)
        idx = torch.sum(torch.mul(gumbel_reps, kth_idx), dim=-1)

        # del kth_vals, kth_idx
        # torch.cuda.empty_cache()
    elif sample:
        gumbel_reps = torch.nn.functional.gumbel_softmax(logits=logits, hard=True)
        idx = torch.sum(torch.mul(gumbel_reps, torch.tensor(range(0, len(tokenizer.vocab))).to(device)), dim=-1)
    else:
        idx = torch.argmax(logits, dim=-1)
        print("ERRORRRRR!!!!!!")

    logits.detach()
    del logits
    gumbel_reps.detach()
    del gumbel_reps
    # torch.cuda.ipc_collect()
    # torch.cuda.empty_cache()
    gc.collect()

    return idx.tolist() if return_list else idx


'''
def __get_init_text(self, seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [tokenizer.mask_token] * max_len + [tokenizer.sep_token] for _ in range(batch_size)]
    # if rand_init:
    #    for ii in range(max_len):
    #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))

    return GAN.Generator.tokenize_batch(batch, tokenizer)
'''


def __get_init_text(self, seed_text, max_len, batch_size=1, rand_init=False, sample_lens=None):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """

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


def __training_generation(self, seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300,
                          burnin=200, sample_lens=None, print_every=10, verbose=True):
    """ Generate for one random position at a timestep

    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = len(seed_text)

    noise_input = torch.tensor(
        self.__get_init_text(seed_text, max_len, batch_size, rand_init=False, sample_lens=sample_lens),
        requires_grad=True, dtype=torch.float, device=device)
    # print(noise_input.dtype)
    # print(noise_input.requires_grad)
    # print(noise_input.is_leaf)

    batch = noise_input.clone()
    # batch=torch.tensor(self.__get_init_text(seed_text, max_len, batch_size)).to(device)
    # batch=torch.tensor(self.__get_init_text(seed_text, max_len, batch_size), device=device)

    # num_groups = 5
    num_groups = 1

    for ii in range(max_iter):

        raw_idxs = np.array(range(0, max_len))
        random.shuffle(raw_idxs)
        rand_idxs = np.array_split(raw_idxs, num_groups)

        for group in rand_idxs:

            '''
            with torch.no_grad():
                embeddings_out = self.embedding_layer(batch.long())
            bert_embeds = (embeddings_out - batch.unsqueeze(-1)).detach() + batch.unsqueeze(-1)
            '''

            out = self.map1(batch.long())[0]
            # out = self.map1(inputs_embeds=bert_embeds)[0]

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

                # del idxs
                # torch.cuda.empty_cache()
            out.detach()
            del out
            # torch.cuda.ipc_collect()
            # torch.cuda.empty_cache()
            # gc.collect()

    return batch


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