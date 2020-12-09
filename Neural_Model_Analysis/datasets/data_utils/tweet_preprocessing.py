"""
Script to preprocess tweet datasets in English, Indonesian, and Arabic.

Extends tweepy preprocessing and methods from Dilan K Jayasekara
https://towardsdatascience.com/extracting-twitter-data-pre-processing-and-sentiment-analysis-using-python-3-0-7192bd8b47cf

URLS are removed by default

Can optionally remove: stopword, punctuation, emoji/emoticon/smiley.

"""


import csv
import unicodedata as ud
import re
import argparse
import pandas as pd
from nltk.corpus import stopwords
import preprocessor as p


def get_stopwords(args):
	language = 'english'
	if args.indonesian:
		language = 'indonesian'
	if args.arabic:
		language = 'arabic'
	return set(stopwords.words(language))


def get_emoticons():
	emoticons_happy = set([
	':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
	':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
	'=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
	'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
	'<3'
	])
	
	emoticons_sad = set([
	':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
	':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
	':c', ':{', '>:\\', ';('
	])
	
	return emoticons_happy.union(emoticons_sad)


def remove_emoji(tweet):
	emoji_pattern = re.compile("["
		 u"\U0001F600-\U0001F64F"  # emoticons
		 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
		 u"\U0001F680-\U0001F6FF"  # transport & map symbols
		 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
		 u"\U00002702-\U000027B0"
		 u"\U000024C2-\U0001F251"
		 "]+", flags=re.UNICODE)
	return emoji_pattern.sub(r'', tweet)
	

def remove_punctuation(s, args):
	s = [c for c in s if not ud.category(c).startswith('P')]
	return ''.join(s)
	

def filter_labels(s):
	special = re.compile(r'USER|URL')
	return special.sub('', s)


def clean_tweet(tweet, args):
	stopwords = get_stopwords(args)
	emoticons = get_emoticons()
	
	# remove by default
	options = [p.OPT.URL]
		
	#o optionally remove emoji
	if args.emoji:
		options.append(p.OPT.EMOJI)
		options.append(p.OPT.SMILEY)

	p.set_options(*options)
	tweet = p.clean(tweet)	
	
	# remove newline chars in tweet
	tweet = re.sub(r'\\n', '', tweet)
	
	# remove labels for indonesian data to match english
	if args.indonesian:
		tweet = filter_labels(tweet) 
	
	# remove emojis from tweet
	if args.emoji:
		tweet = remove_emoji(tweet)
	
	# remove punctuation if specified
	if args.punctuation:
		tweet = remove_punctuation(tweet, args)
	
	# split for further processing
	tokens = tweet.split()

	filtered_tweet = list()
	for t in tokens:	
		remove = False
			
		# optionally remove stopwords
		if args.stopwords and remove_punctuation(t.lower(), args) in stopwords:
			remove = True
		
		# optionally remove emoticons
		if args.emoji and t in emoticons:
			remove = True
			
		if not remove:
			filtered_tweet.append(t)

	tweet = ' '.join(filtered_tweet)
		
	return tweet


def main(args):
	df = pd.read_csv(
		args.dataset,
        delimiter='\t',
        header=None,
        names=['sentence', 'label'],
		quotechar='\0'
    )

	df['cleaned_sentence'] = df['sentence'].apply(lambda x: clean_tweet(x, args))

	df = df[['cleaned_sentence', 'label']]
	df.to_csv(
		args.output, 
		sep='\t',
		index=False,
		header=False,
		quoting=csv.QUOTE_NONE
	)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	
	# files
	parser.add_argument("dataset", help="Data file tsv for cleaning")
	parser.add_argument("output", help="output file with cleaned data")
	
	# languages
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-i", "--indonesian", help="indonesian language (default is English)", action='store_true')
	group.add_argument("-a", "--arabic", help="arabic language (default is English)", action='store_true')
	
	# options
	parser.add_argument("-s", "--stopwords", help="filter stopwords", action='store_true')
	parser.add_argument("-p", "--punctuation", help="remove punctuation", action='store_true')
	parser.add_argument("-e", "--emoji", help="remove emoji, emoticons, and smileys", action='store_true')
	
	args = parser.parse_args()
	main(args)
