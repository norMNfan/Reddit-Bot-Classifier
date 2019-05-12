import pymongo
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
import scattertext as st


class Classifier:
	def classify_all(self):
		classify_all()

	def classify_user(self, username):
		classify_user(username)

	def create_visualizations(self):
		create_interactive_visualizations()
		create_number_of_comments_and_posts_pie_chart()
		create_cake_day_histogram()
		create_time_of_day_for_comment_histogram()


def classify_all():

	X_comments, X_comments_sub, Y_comments, Y_comments_sub = get_comment_data_from_mongo_for_all()
	X_posts, X_posts_sub, Y_posts, Y_posts_sub = get_post_data_from_mongo_for_all()

	print('------------------------------------------------------------')
	print_post_numbers(X_posts, Y_posts)
	print_comment_numbers(X_comments, Y_comments)
	print('------------------------------------------------------------')

	# Post subreddit
	print('post subreddit')
	classify(X_posts_sub, Y_posts_sub, stem=False)
	print('------------------------------------------------------------')

	# Comment subreddit
	print('comment subreddit')
	classify(X_comments_sub, Y_comments_sub, stem=False)
	print('------------------------------------------------------------')

	# Posts
	print('posts')
	classify(X_posts, Y_posts)
	print('------------------------------------------------------------')

	# Comments
	print('comments')
	classify(X_comments, Y_comments)
	print('------------------------------------------------------------')


def classify(X, Y, stem=True):

	stemmer = SnowballStemmer('english')

	if stem:
		X = [' '.join([stemmer.stem(word) for word in text.split(' ')]) for text in X]

	# Split train/test data
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=31)

	# Able to add different classifiers to this list to compare their results
	classifiers = [ExtraTreesClassifier()]

	for classifier in classifiers:

		print(classifier)

		# Create pipeline: word vectorizer => create tfidf model => Run classifier
		text_clf = Pipeline([
			('vect', CountVectorizer(lowercase=True, strip_accents='ascii', stop_words='english')), 
			('tfidf', TfidfTransformer()), 
			('clf', classifier),])

		# Train model
		text_clf.fit(X_train, Y_train)

		# Predict test data
		Y_pred = text_clf.predict(X_test)
		print(confusion_matrix(Y_pred, Y_test))
		print('accuracy: ' + str(accuracy_score(Y_test, Y_pred)))
		print(classification_report(Y_pred, Y_test))

		# Predict train data
		#predicted_train = text_clf.predict(X_train)
		#print('Train: ')
		#acc, precision, recall, f1 = evaluate(Y_train, predicted_train)
		#print("Evalution: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))
		#print(confusion_matrix(predicted_train, Y_train))
		#print(classification_report(predicted_train, Y_train))


def classify_users(X, Y, X_user, Y_user):

	if len(X_user) < 1 or len(Y_user) < 1:
		return

	# Create pipeline: word vectorizer => create tfidf model => Run NB classifier
	text_clf = Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0))
	])

	# Train model
	text_clf.fit(X, Y)

	# Predict train data
	predicted_user = text_clf.predict(X_user)
	print('user accuracy: ' + str(np.mean(predicted_user == Y_user)))
	counter = Counter(predicted_user)
	print(counter)
	if counter[1] > counter[0]:
		print('bot')
	else:
		print('user')


def classify_user(username):

	X_comments, X_comments_sub, Y_comments, Y_comments_sub = get_comment_data_from_mongo_for_all()
	X_posts, X_posts_sub, Y_posts, Y_posts_sub = get_post_data_from_mongo_for_all()

	X_comments_user, X_comments_sub_user, Y_comments_user, Y_comments_sub_user = get_comment_data_from_mongo_for_user(
		username)
	X_posts_user, X_posts_sub_user, Y_posts_user, Y_posts_sub_user = get_post_data_from_mongo_for_user(username)

	print('------------------------------------------------------------')
	print_post_numbers(X_posts, Y_posts)
	print_comment_numbers(X_comments, Y_comments)
	print('------------------------------------------------------------')

	# Post subreddit
	print('post subreddit')
	classify_users(X_posts_sub, Y_posts_sub, X_posts_sub_user, Y_posts_user)
	print('------------------------------------------------------------')

	# Comment subreddit
	print('comment subreddit')
	classify_users(X_comments_sub, Y_comments_sub, X_comments_sub_user, Y_comments_user)
	print('------------------------------------------------------------')

	# Posts
	print('posts')
	classify_users(X_posts, Y_posts, X_posts_user, Y_posts_user)
	print('------------------------------------------------------------')

	# Comments
	print('comments')
	classify_users(X_comments, Y_comments, X_comments_user, Y_comments_user)
	print('------------------------------------------------------------')


def get_post_data_from_mongo_for_user(username):
	return get_post_data_from_mongo(username)


def get_comment_data_from_mongo_for_user(username):
	return get_comment_data_from_mongo(username)


def get_post_data_from_mongo_for_all():
	return get_post_data_from_mongo('')


def get_comment_data_from_mongo_for_all():
	return get_comment_data_from_mongo('')


def print_comment_numbers(X_comments, Y_comments):
	print('num comments: ' + str(len(X_comments)))
	print('\tnum bot comments: ' + str(Y_comments.count(1)))
	print('\tnum user comments: ' + str(Y_comments.count(0)))


def print_post_numbers(X_posts, Y_posts):
	print('num posts: ' + str(len(X_posts)))
	print('\tnum bot posts: ' + str(Y_posts.count(1)))
	print('\tnum user posts: ' + str(Y_posts.count(0)))


def get_comment_data_from_mongo(username):
	X_comments, Y_comments, X_comments_sub, Y_comments_sub = [], [], [], []
	# Get data from database
	for doc in get_redditor_collection().find({}):

		if doc['username'] == username:
			continue

		# Skip is account contains no comment and no posts
		if len(doc['comments']) < 1:
			continue

		X_comments_sub_obj = ''

		for comment in doc['comments']:
			X_comments_sub_obj = X_comments_sub_obj + comment['subreddit'] + ' '
			X_comments.append(comment['body'])
			if doc['is_bot']:
				Y_comments.append(1)
			else:
				Y_comments.append(0)

		# Add subreddit data
		X_comments_sub.append(X_comments_sub_obj)
		if doc['is_bot']:
			Y_comments_sub.append(1)
		else:
			Y_comments_sub.append(0)
	return X_comments, X_comments_sub, Y_comments, Y_comments_sub


def get_post_data_from_mongo(username):
	X_posts, Y_posts, X_posts_sub, Y_posts_sub = [], [], [], []
	# Get data from database
	for doc in get_redditor_collection().find({}):

		if doc['username'] == username:
			continue

		# Skip is account contains no comment and no posts
		if len(doc['posts']) < 1:
			continue

		X_posts_sub_obj = ''

		for post in doc['posts']:
			X_posts_sub_obj = X_posts_sub_obj + post['subreddit'] + ' '
			X_posts.append(post['title'])
			if doc['is_bot']:
				Y_posts.append(1)
			else:
				Y_posts.append(0)

		X_posts_sub.append(X_posts_sub_obj)
		if doc['is_bot']:
			Y_posts_sub.append(1)
		else:
			Y_posts_sub.append(0)
	return X_posts, X_posts_sub, Y_posts, Y_posts_sub


def create_interactive_visualizations():

	X_comments, Y_comments, X_posts, Y_posts = [], [], [], []
	X_comments_sub, X_posts_sub, Y_comments_sub, Y_posts_sub = [], [], [], []

	# Get data from database
	for doc in get_redditor_collection().find({}):
		X_comments_sub_obj = ''
		X_posts_sub_obj = ''

		if len(doc['comments']) < 1 and len(doc['posts']) < 1:
			continue

		# Add comment data
		for comment in doc['comments']:
			X_comments_sub_obj = X_comments_sub_obj + comment['subreddit'] + ' '
			X_comments.append(comment['body'])
			if doc['is_bot']:
				Y_comments.append(1)
			else:
				Y_comments.append(0)

		# Add post data
		for post in doc['posts']:
			X_posts_sub_obj = X_posts_sub_obj + post['subreddit'] + ' '
			X_posts.append(post['title'])
			if doc['is_bot']:
				Y_posts.append(1)
			else:
				Y_posts.append(0)

		# Add subreddit data
		X_comments_sub.append(X_comments_sub_obj)
		X_posts_sub.append(X_posts_sub_obj)
		if doc['is_bot']:
			Y_comments_sub.append(1)
			Y_posts_sub.append(1)
		else:
			Y_comments_sub.append(0)
			Y_posts_sub.append(0)

	data = np.empty([len(X_comments), 2], dtype=object)
	data[:, 0] = Y_comments
	data[:, 1] = X_comments

	for d in data:
		if d[0] == 0:
			d[0] = 'normal'
		else:
			d[0] = 'bot'

	df = pd.DataFrame({'label': data[:, 0], 'text':data[:, 1]})
	print(df)

	corpus = (st.CorpusFromPandas(df, category_col='label', text_col='text', nlp=st.whitespace_nlp_with_sentences)
		.build()
		.get_unigram_corpus()
		.compact(st.ClassPercentageCompactor(term_count=2, term_ranker=st.OncePerDocFrequencyRanker)))

	html = st.produce_characteristic_explorer(
		corpus,
		category='normal',
		category_name='Normal',
		not_category_name='Bot'
	)
	open('comment_text_chart.html', 'wb').write(html.encode('utf-8'))


def create_number_of_comments_and_posts_pie_chart():

	X_comments, X_comments_sub, Y_comments, Y_comments_sub = get_comment_data_from_mongo_for_all()
	X_posts, X_posts_sub, Y_posts, Y_posts_sub = get_post_data_from_mongo_for_all()

	# Create post pie chart
	labels = 'Normal', 'Bot'
	sizes = [Y_posts.count(0), Y_posts.count(1)]
	colors = ['blue', 'red']
	explode = (0, 0.1)
	plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
	plt.axis('equal')
	plt.show()

	# Create comment pie chart
	labels = 'Normal', 'Bot'
	sizes = [Y_comments.count(0), Y_comments.count(1)]
	colors = ['blue', 'red']
	explode = (0, 0.1)
	plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
	plt.axis('equal')
	plt.show()


def create_cake_day_histogram():

	bot_cake_days, normal_cake_days = [], []

	# Get data from database
	for doc in get_redditor_collection().find({}):
		if doc['is_bot']:
			t = (datetime.datetime.fromtimestamp(doc['cake_day']),)
			bot_cake_days.append(t)
		else:
			t = (datetime.datetime.fromtimestamp(doc['cake_day']),)
			normal_cake_days.append(t)

	labels = ['date']

	df_bots = pd.DataFrame.from_records(bot_cake_days, columns=labels)
	df_normal = pd.DataFrame.from_records(normal_cake_days, columns=labels)

	data = np.array([['', 'Num'], ['Bots', len(bot_cake_days)], ['Users', len(normal_cake_days)]])

	print(pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:]))

	df = pd.DataFrame({'Number of accounts': ['Bot', 'Normal'], 'val':[len(bot_cake_days), len(normal_cake_days)]})
	df.plot.bar(x='Number of accounts', y='val', rot=0)

	df_bots["date"] = df_bots["date"].astype("datetime64")
	df_bots.groupby([df_bots["date"].dt.year, df_bots["date"].dt.month]).count().plot(kind="bar")

	df_normal["date"] = df_normal["date"].astype("datetime64")
	df_normal.groupby([df_normal["date"].dt.year, df_normal["date"].dt.month]).count().plot(kind="bar")

	print(df_bots)
	print(df_normal)

	visualize(df_bots)
	visualize(df_normal)


def create_time_of_day_for_comment_histogram():

	bot_comments, normal_comments = [], []

	# Get data from database
	for doc in get_redditor_collection().find({}):
		if doc['is_bot']:
			for comment in doc['comments']:
				t = (datetime.datetime.fromtimestamp(comment['created_utc']),)
				bot_comments.append(t)
		else:
			for comment in doc['comments']:
				t = (datetime.datetime.fromtimestamp(comment['created_utc']),)
				normal_comments.append(t)

	labels = ['date']

	df_bots = pd.DataFrame.from_records(bot_comments, columns=labels)
	df_normal = pd.DataFrame.from_records(normal_comments, columns=labels)

	visualize(df_bots, title='bot users')
	visualize(df_normal, title='normal users')


def get_redditor_collection():
	client = pymongo.MongoClient('mongodb://localhost:27017')
	db = client['redditors']
	collection = db['redditors']
	client.close()
	return collection


# Creates histogram of a dataframe
def visualize(df, column_name='date', color='#494949', title=''):

	plt.figure(figsize=(20, 10))
	ax = (df[column_name].groupby(df[column_name].dt.hour)
	                     .count()).plot(kind="bar", color=color)
	ax.set_facecolor('#eeeeee')
	ax.set_xlabel("hour of the day")
	ax.set_ylabel("count")
	ax.set_title(title)
	plt.show()
	plt.savefig('visuals/' + title + '.png')