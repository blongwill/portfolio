from main import examine_model, Settings
import os

if __name__ == "__main__":
	tweets_to_viz = dict()
	for tweet_file in os.listdir('visualization_tweets'):
		filename = tweet_file.split('.')[0]
		print(filename)
		with open('visualization_tweets/{}'.format(tweet_file)) as f:
			tweets = f.readlines()
		tweets_to_viz[filename] = tweets


	for f in os.listdir('configs'):
		filename = f.split('.')[0]
		print(filename)
		if "arabic" in filename or "all" in filename:
			examine_model(filename, "arabic_abusive", tweets_to_viz["arabic_abusive"])
			examine_model(filename, "arabic_hate", tweets_to_viz["arabic_hate"])
			examine_model(filename, "arabic_normal", tweets_to_viz["arabic_normal"])
		if "english" in filename or "all" in filename:
			examine_model(filename, "english_abusive", tweets_to_viz["english_abusive"])
			examine_model(filename, "english_hate", tweets_to_viz["english_hate"])
			examine_model(filename, "english_normal", tweets_to_viz["english_normal"])
		if "indonesian" in filename or "all" in filename:
			examine_model(filename, "indonesian_abusive", tweets_to_viz["indonesian_abusive"])
			examine_model(filename, "indonesian_hate", tweets_to_viz["indonesian_hate"])
			examine_model(filename, "indonesian_normal", tweets_to_viz["indonesian_normal"])

