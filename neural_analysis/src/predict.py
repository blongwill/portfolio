import os
import argparse
import csv
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from main import get_gpu, prepare_data, split_data, load_bert_tokenizer, Settings


def record_score_information(settings, true, hyp):
	settings.write_debug("normal:0, abusive:1, hate:2, spam:3")
	settings.write_debug(classification_report(true, hyp))	
	
	confusion = confusion_matrix(true, hyp)
	settings.write_debug("Confusion matrix:\n{}".format(str(confusion)))


def predict(device, model, predict_dataloader):
	model.eval() 

	preds = list()
	for batch in predict_dataloader:
		batch = tuple(t.to(device) for t in batch)
		b_input_ids, b_input_mask, b_labels = batch
	
		with torch.no_grad():
			outputs = model(b_input_ids, token_type_ids=None, attention_mask = b_input_mask)
		
			logits = outputs[0]
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to("cpu").numpy()
		
			preds.extend(np.argmax(logits, axis=1).flatten())
	return preds


def predict_model(experiment, output_dir):
	"""Writes the predictions of a given dataset file."""
	saved_dir = "/home2/preetmhn/clms/ling_575_nlms/models/saved_{}".format(experiment)
	model = torch.load('{}/hate_speech_model_trained.pt'.format(saved_dir))
	settings = Settings(experiment, True)
	
	# get gpu
	device = get_gpu(settings)
	
	# get data, split with the same random seed as in training
	input_ids, labels, attention_masks = prepare_data(settings)
	_, validation_inputs, _, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
	_, validation_dataloader = split_data(settings, input_ids, labels, attention_masks)
	
	# make predictions and write to file
	settings.write_debug("Getting model predictions")
	preds = predict(device, model, validation_dataloader)

	# load tokenizer for the decoding
	tokenizer = load_bert_tokenizer(settings, True)
	
	# write to file
	settings.write_debug("Writing model predictions")
	output_file = os.path.join(output_dir, experiment + '_pred.txt')
	out = csv.writer(open(output_file, 'w+', encoding='utf-8'), delimiter='\t')
	out.writerow(['input', 'true', 'pred'])
	for i in range(len(preds)):
		tokens = tokenizer.decode(input_ids[i], skip_special_tokens=True)
		out.writerow([tokens, labels[i], preds[i]])
	
	# write scores
	settings.write_debug("Getting test evaluation")
	record_score_information(settings, validation_labels, preds)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("experiment", help="Name of experiment")
	parser.add_argument("output_dir", help="Output directory for predictions")
	args = parser.parse_args()
	predict_model(args.experiment, args.output_dir)
