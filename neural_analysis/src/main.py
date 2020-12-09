
import os, sys
import pandas as pd  # type: ignore
from typing import List, Tuple
from transformers import BertTokenizer, BertModel , BertForSequenceClassification, AdamW, BertConfig,get_linear_schedule_with_warmup  # type: ignore

if not 'bertviz_repo' in sys.path:
    sys.path += ['../bertviz_repo']

from bertviz import head_view  # type: ignore
from bertviz.neuron_view import show  # type: ignore
from bertviz import model_view  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import f1_score as compute_f1_score  # type: ignore
#from transformers import BertForSequenceClassification, AdamW, BertConfig  # type: ignore
#from transformers import get_linear_schedule_with_warmup  # type: ignore
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np  # type: ignore
import time
import datetime
import random
import allennlp  # type: ignore
import torch
from IPython.display import display, HTML  # type: ignore


debug_folder = '../debug/'
debug_suffix = '.txt'
config_folder = '../configs/'
config_suffix = '.txt'
datasets_folder = '../datasets/'
dataset_suffix = '.txt'

# call_html method is for visualizations
# https://colab.research.google.com/drive/1PEHWRHrvxQvYr9NFRC-E_fr3xDq1htCj#scrollTo=Mv6H9QK9yLLegv12
def call_html():
  import IPython  # type: ignore
  return IPython.core.display.HTML('''
        <script src="../require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: 'https://cdnjs.cloudflare.com/ajax/libs/jquery/3.4.1/jquery.min',
            },
          });
        </script>
        ''')

# Store settings from config file here
class Settings:
    def __init__(self, config_name, append_existing_debug_file):

        current_folder = os.path.dirname(__file__)
        if len(current_folder):
            current_folder += '/'

        # Verify config file exists.
        config_path = current_folder + config_folder + config_name + config_suffix
        if not os.path.exists(config_path):
            raise ValueError(
                'Error: experiment config file not found: {0}'.format(
                    os.path.abspath(config_path)
                )
            )
        # Store config settings as string=>string map.
        # Individual settings can be parsed via getter functions in class Settings.
        self.__settings_map = {}
        self.__settings_map['config_name'] = config_name
        with open(config_path, 'r') as f:
            for setting in f.readlines():
                name = setting.split()[0]
                value = setting.split()[1]
                self.__settings_map[name] = value

        # Store dataset path, for loading training/test data
        # Split dataset value on commas if we have more than one
        dataset_list = self.__settings_map['dataset'].split(',')
        all_dataset_paths = []
        for dataset in dataset_list:
            all_dataset_paths.append(current_folder  \
                + datasets_folder  \
                + dataset \
                + dataset_suffix)
        self.__dataset_path = all_dataset_paths

        # Overwrite the debug output file from prior run, if one exists.
        debug_path = current_folder + debug_folder + config_name + debug_suffix
        if not append_existing_debug_file:
            with open(debug_path, 'w') as f:
                f.write('Initializing debugger output file...\n')
                # Log all experiment settings to debug output:
                f.write('###\n')
                f.write('Experiment config settings:\n')
                for setting in self.__settings_map:
                    value = self.__settings_map[setting]
                    f.write('{}: {}\n'.format(setting, value))
                f.write('###\n')
                f.write('DATASETS: {}\n'.format(str(self.__dataset_path)))
        # Store debug path, for writing debugger output during execution.
        self.__debug_path = debug_path


    def write_debug(self, msg: str):
        with open(self.__debug_path, 'a') as f:
            f.write(msg + '\n')

    # Getter Methods
    def get_dataset_path(self) -> str:
        return self.__dataset_path
    def get_random_seed(self) -> int:
        return int(self.__settings_map['random_seed'])
    def get_model_type(self) -> str:
        return self.__settings_map['model_type']
    def get_num_classifier_labels(self) -> int:
        return int(self.__settings_map['num_classifier_labels'])
    def get_output_attentions(self) -> bool:
        return bool(self.__settings_map['output_attentions'])
    def get_output_hidden_states(self) -> bool:
        return bool(self.__settings_map['output_hidden_states'])
    def get_batch_size(self) -> int:
        return int(self.__settings_map['batch_size'])
    def get_learning_rate(self) -> float:
        return float(self.__settings_map['learning_rate'])
    def get_num_training_epochs(self) -> int:
        return int(self.__settings_map['num_training_epochs'])
    def get_epsilon(self) -> float:
        return float(self.__settings_map['epsilon'])
    def get_optimizer_name(self) ->  str:
        return self.__settings_map['optimizer']
    def get_num_warmup_steps(self) ->  int:
        return int(self.__settings_map['num_warmup_steps'])
    def get_max_tokens_length(self) -> int:
        return int(self.__settings_map['max_tokens_length'])
    def get_config_name(self) -> str:
        return self.__settings_map['config_name']

def get_optimizer(optimizer_name:str, model_paramaters, settings:Settings):
    if optimizer_name == 'AdamW':
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model_paramaters,
                          lr=settings.get_learning_rate(),  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=settings.get_epsilon()  # args.adam_epsilon  - default is 1e-8.
         )

    if optimizer:
        return optimizer
    else:
        raise ValueError('Error: optimizer not initailized')


# Output: an array of token ids
def encode_sentences(settings:Settings, tokenizer: BertTokenizer, sentences: list) -> Tuple[list,list]:
    # Use the pretrained BERT transfer model
    # return as an array of token id's
    # converts tokens to id's and includes CLS and SEP
    # can be converted back with convert_ids_to_tokens
    encoding_dict = tokenizer.batch_encode_plus(
        sentences,
        pad_to_max_length=True,
        max_length=settings.get_max_tokens_length(),
        add_special_tokens=True,
    )
    return encoding_dict['input_ids'], encoding_dict['attention_mask']


# Output: an array of features, which come from a pretrained BERT model
def featurize_tokens(token_ids: List[int]) -> List[torch.Tensor]:
    # Use the pretrained BERT transfer model
    #model needs a tensor as input
    model.eval()        #put the model in evaluation mode (may or may not need, comment if not needed)

    #seg_ids = [1] * len([i for i in token_ids if i > 103])     #remove  CLS, SEP, MASK
    seg_ids = [1] * len(token_ids)

    #tensor-ized input arrs for next one
    seg_tens = torch.tensor([seg_ids])
    tok_tens = torch.tensor([token_ids])

    with torch.no_grad():   #will save the model computation. HELLUVA LOT.
        encoded_layers = model(tok_tens, seg_tens)[2] # this returns the tuple of all the encoded layers

    tok_embeddings = torch.stack(encoded_layers, dim=0) #the word vectors
    # Remove dimension 1, the "batches".
    tok_embeddings = torch.squeeze(tok_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    tok_embeddings = tok_embeddings.permute(1,0,2) 

    catvecs_tok_out = []
    for tok in tok_embeddings:
        catvec = torch.cat((tok[-1], tok[-2], tok[-3], tok[-4]), dim=0)
        catvecs_tok_out.append(catvec)
    return catvecs_tok_out


# Output: a tuple of a list of sentences and their corresponding list of labels ([sentences], [labels])
def get_dataset(settings: Settings) -> Tuple[list,list]:
    settings.write_debug('Starting load dataset')
    # Load the dataset into a pandas dataframe
    dataset_path = settings.get_dataset_path()
    df = pd.DataFrame()
    for datum in dataset_path:
        df = df.append(pd.read_csv(
            datum,
            delimiter='\t',
            header=None,
            names=['sentence', 'label'],
            quoting=3,
        ))
    df = df[df['sentence'].notnull()]
    label_lookup = {'normal': 0, 'abusive': 1, 'hate': 2, 'spam':3}
    df['num_label'] = df['label'].apply(lambda x: label_lookup.get(x, 0))
    sentences = df.sentence.values
    labels = df.num_label.values
    #labels = torch.tensor(df.num_label.values)
    if not len(sentences) == len(labels):
        raise ValueError(
            'Error: number of sentences and number of labels must be the same'
        )
    '''
    feature_tensors = []
    for i in range(0, len(sentences)):
        sentence = sentences[i][:512] # truncating to the max sequence the model can handle
        encoding = encode_sentence(sentence)
        featurized = featurize_tokens(token_ids)
        feature_tensors.append(featurized)
    settings.write_debug('Total dataset size: {0}'.format(len(feature_tensors)))
    settings.write_debug('First data sentence: {0}'.format(
        sentences[0].encode('utf-8').decode('latin-1')
    ))
    '''

    settings.write_debug('First data label: {0}'.format(labels[0]))
    settings.write_debug('Finished load dataset')
    #return feature_tensors, labels
    return sentences, labels


# Output: Dataloader objects for validation and training
def split_data(settings: Settings, input_ids:list, labels:list, attention_masks:list) -> Tuple[DataLoader,DataLoader]:
    settings.write_debug('Starting split dataset')
    # TODO implement split dataset
    settings.write_debug('Input IDs dimension-1 size: {0}'.format(len(input_ids)))
    settings.write_debug('Input IDs dimension-2 size: {0}'.format(len(input_ids[0])))
    settings.write_debug('Input Labels dimension-1 size: {0}'.format(len(labels)))
    settings.write_debug('Input Masks dimension-1 size: {0}'.format(len(attention_masks)))
    settings.write_debug('Input Masks dimension-2 size: {0}'.format(len(attention_masks[0])))
    ########### sklearn.model_selection for selection ###########################################################################

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                           random_state=2018, test_size=0.1)

    ######################################################################################

    settings.write_debug('Train inputs dimension-1 size: {0}'.format(len(train_inputs)))
    settings.write_debug('Train inputs dimension-2 size: {0}'.format(len(train_inputs[0])))

    settings.write_debug('Train labels dimension-1 size: {0}'.format(len(train_labels)))

    settings.write_debug('Train masks dimension-1 size: {0}'.format(len(train_masks)))
    settings.write_debug('Train masks dimension-2 size: {0}'.format(len(train_masks[0])))

    #Todo Convert these input_ids, labels, and attention masks to tensor objects
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    
    # Create the DataLoader for our validation set
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=settings.get_batch_size())

    # Todo Do the same for the training set
    # Create the DataLoader for our validation set
    training_data = TensorDataset(train_inputs, train_masks, train_labels)
    training_sampler = SequentialSampler(training_data)
    training_dataloader = DataLoader(training_data, sampler=training_sampler,
                                       batch_size=settings.get_batch_size())

    settings.write_debug('Finished split dataset')
    return (training_dataloader,validation_dataloader)

def init_model(settings:Settings):
    settings.write_debug('Starting to Load BertForSequenceClassification')
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        settings.get_model_type(),  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=settings.get_num_classifier_labels(),  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=settings.get_output_attentions(),  # Whether the model returns attentions weights.
        output_hidden_states=settings.get_output_attentions(),  # Whether the model returns all hidden-states.
    )
    # Tell pytorch to run this model on the GPU.
    model.cuda()
    settings.write_debug('Finished initializing model')
    return model

def get_gpu(settings:Settings):
    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        settings.write_debug('There are %d GPU(s) available.' % torch.cuda.device_count())

        settings.write_debug('We will use the GPU:' + torch.cuda.get_device_name(0))

    # If not...
    else:
        settings.write_debug('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device


def train_model(device:torch.device, model:BertForSequenceClassification , settings: Settings, train_dataloader:DataLoader):
    settings.write_debug('Starting train hate speech model')

    optimizer = get_optimizer(settings.get_optimizer_name(), model.parameters(), settings)

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = settings.get_num_training_epochs()

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=settings.get_num_warmup_steps(),  # Default value in run_glue.py
                                                num_training_steps=total_steps)


    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    random_seed = settings.get_random_seed()

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        settings.write_debug("")
        settings.write_debug('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        settings.write_debug('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                settings.write_debug('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        settings.write_debug("")
        settings.write_debug("  Average training loss: {0:.2f}".format(avg_train_loss))
        settings.write_debug("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    settings.write_debug("")
    settings.write_debug("Training complete!")

    

    settings.write_debug('Finished train hate speech model')


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_f1(settings, preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    labels = list(range(0, settings.get_num_classifier_labels()))
    f1 = compute_f1_score(labels_flat, pred_flat, labels, average='weighted', zero_division=0)
    return f1
    

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def test_model(
    settings: Settings,
    device: torch.device,
    model: BertForSequenceClassification,
    evaluation_dataloader:DataLoader,
    dataset_type: str
):
    # Test model on the given input dataset.
    # We can also (???) examine the BERT feature values for sentences which are
    # associated with hatespeech, versus sentences which are not associated
    # with hatespeech.
    settings.write_debug('Starting evaluation: {0} data'.format(dataset_type))
    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in evaluation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        tmp_f1 = flat_f1(settings, logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        eval_f1 += tmp_f1

        # Track the number of batches
        nb_eval_steps += 1

    settings.write_debug("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    settings.write_debug("  F1: {0:.2f}".format(eval_f1 / nb_eval_steps))
    settings.write_debug("  Validation took: {:}".format(format_time(time.time() - t0)))
    settings.write_debug('Finished evaluation: {0} data'.format(dataset_type))


def examine_model(experiment, labeltype, inputs = None):
    saved_dir = "models/saved_{}".format(experiment)
    model = torch.load('{}/hate_speech_model_trained.pt'.format(saved_dir))
    settings = Settings(experiment, True)
    settings.write_debug('Starting visualization of trained model')

    if inputs == None:
        inputs = ["The cat sat on the dog"]

    tokenizer = load_bert_tokenizer(settings, True)
    model_type = settings.get_model_type()

    # examine neurons during a given input
    # call_html()
    # show(model, 'bert', tokenizer, input_data)

    for input_data in inputs:

        inputs = tokenizer.encode_plus(input_data, return_tensors='pt', add_special_tokens=True)
        device = get_gpu(settings)
        token_type_ids = inputs['token_type_ids'].to(device)
        input_ids = inputs['input_ids'].to(device)
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        input_id_list = input_ids[0].tolist() # Batch index 0
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    
        main_html = call_html()

        # examine whole model during a given input
        model_html1, model_js2, model_js3 = model_view(attention, tokens)

        # examine just the heads during a given input
        head_html1, head_js2, head_js3 = head_view(attention, tokens)

        config_name = settings.get_config_name()
        os.makedirs("viz/{}/{}/{}".format(config_name, labeltype, input_data), exist_ok=True)

        with open("viz/{}/{}/{}/model_vizualization.html".format(config_name, labeltype, input_data), 'w+') as f:
            f.write("{} \n <script>{} \n {}</script> \n {} \n"
                .format(main_html.data, model_js2.data, model_js3.data, model_html1.data))

        with open("viz/{}/{}/{}/head_vizualization.html".format(config_name, labeltype, input_data), 'w+') as f:
            f.write("{} \n <script>{} \n {}</script> \n {} \n"
                .format(main_html.data, head_js2.data, head_js3.data, head_html1.data))

    
        # Which BERT features were weighted the most? The least?
        settings.write_debug('Finished visualization of trained model given input {}'.format(input_data))
        settings.write_debug('HTML file saved to viz/{}/{}'.format(config_name, input_data))

# Load the BERT tokenizer.
def load_bert_tokenizer(settings:Settings, do_lower_case:bool ) -> BertTokenizer:
    settings.write_debug('Loading BERT tokenizer...')
    tokenizer=BertTokenizer.from_pretrained(settings.get_model_type(), do_lower_case=do_lower_case)

    return tokenizer

def prepare_data(settings:Settings) -> Tuple[list,list,list]:

    # Takes dataset directory string adn Returns tuple of (sentences , labels)
    sentences, labels = get_dataset(settings)

    ### Takes model type string and do lowercase boolean to initate the bert tokenizer
    tokenizer = load_bert_tokenizer(settings, True)

    input_ids, attention_masks=encode_sentences(settings, tokenizer,sentences)

    return input_ids, labels, attention_masks

def run_experiment(settings: Settings):
    settings.write_debug('Starting experiment execution')

    device=get_gpu(settings)

    input_ids, labels, attention_masks = prepare_data(settings)

    # Split the labels : requires input ids, labels, and attention masks ###############
    train_dataloader, validation_dataloader = split_data(settings, input_ids,labels,attention_masks)

    settings.write_debug("Made it here")

    
    # Initialize bert model with task
    hate_speech_model = init_model(settings)

    # train model on training data + fine tune on validation data
    # Requires  train_dataloader:DataLoader, validation_dataloader:DataLoader objects as input parameters
    train_model(device, hate_speech_model, settings, train_dataloader)

    # evaluate model on test data
    test_model(settings, device, hate_speech_model, validation_dataloader, 'test')

    experiment = settings.get_config_name()

    os.makedirs('models/saved_{}'.format(experiment), exist_ok=True)

    # save model
    torch.save(hate_speech_model, 'models/saved_{}/hate_speech_model_trained.pt'.format(experiment)) 
    
    settings.write_debug('Finished experiment execution')


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError('Error: must specify an experiment config')
    config_name = sys.argv[1]
    settings = Settings(config_name, False)
    run_experiment(settings)

