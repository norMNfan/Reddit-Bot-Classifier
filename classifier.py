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
from sklearn.naive_bayes import GaussianNB
import scattertext as st

X_comments, Y_comments, X_posts, Y_posts = [], [], [], []
X_comments_sub, X_posts_sub, Y_comments_sub, Y_posts_sub = [], [], [], []

# Classifies the documents and prints the results
def classify(X, Y, stem=True):

	# If you need to cut the data down in size
	# X, Z1, Y, Z2= train_test_split(X, Y, test_size=0.90, random_state=31)

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
		#acc, precision, recall, f1 = evaluate(Y_test, Y_pred)
		#print("Evalution: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))
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

# Classifies a user and prints the results
def classify_user(X, Y, X_user, Y_user):

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

# Starting point of classifying a user
def classify_user(username):

	X_comments, Y_comments, X_posts, Y_posts = [], [], [], []
	X_comments_sub, X_posts_sub, Y_comments_sub, Y_posts_sub = [], [], [], []

	# Get data from database
	for doc in get_redditor_collection().find({}):
		if doc['username'] != username:
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

	X_comments_user, Y_comments_user, X_posts_user, Y_posts_user, X_comments_sub_user, X_posts_sub_user = [], [], [], [], [], []

	# Get data from database
	for doc in get_redditor_collection().find({'username':username}):
		X_comments_sub_obj = ''
		X_posts_sub_obj = ''

		if len(doc['comments']) < 1 and len(doc['posts']) < 1:
			continue

		# Add comment data
		for comment in doc['comments']:
			X_comments_sub_obj = X_comments_sub_obj + comment['subreddit'] + ' '
			X_comments_user.append(comment['body'])
			if doc['is_bot']:
				Y_comments_user.append(1)
			else:
				Y_comments_user.append(0)

		# Add post data
		for post in doc['posts']:
			X_posts_sub_obj = X_posts_sub_obj + post['subreddit'] + ' '
			X_posts_user.append(post['title'])
			if doc['is_bot']:
				Y_posts_user.append(1)
			else:
				Y_posts_user.append(0)

		# Add subreddit data
		X_comments_sub_user.append(X_comments_sub_obj)
		X_posts_sub_user.append(X_posts_sub_obj)
		if doc['is_bot']:
			Y_comments_user.append(1)
			Y_posts_user.append(1)
		else:
			Y_comments_user.append(0)
			Y_posts_user.append(0)

	print('------------------------------------------------------------')
	print('num comments: ' + str(len(X_comments)))
	print('\tnum bot comments: ' + str(Y_comments_user.count(1)))
	print('\tnum user comments: ' + str(Y_comments_user.count(0)))

	print('num posts: ' + str(len(X_posts)))
	print('\tnum bot posts: ' + str(Y_posts_user.count(1)))
	print('\tnum user posts: ' + str(Y_posts_user.count(0)))
	print('------------------------------------------------------------')

	# Post subreddit
	print('post subreddit')
	classify_user(X_posts_sub, Y_posts_sub, X_posts_sub_user, Y_posts_user)
	print('------------------------------------------------------------')

	# Comment subreddit
	print('comment subreddit')
	classify_user(X_comments_sub, Y_comments_sub, X_comments_sub_user, Y_comments_user)
	print('------------------------------------------------------------')

	# Posts
	print('posts')
	classify_user(X_posts, Y_posts, X_posts_user, Y_posts_user)
	print('------------------------------------------------------------')

	# Comments
	print('comments')
	classify_user(X_comments, Y_comments, X_comments_user, Y_comments_user)
	print('------------------------------------------------------------')

# Starting point of classifying all the documents
def classify_all():

	X_comments, Y_comments, X_posts, Y_posts = [], [], [], []
	X_comments_sub, X_posts_sub, Y_comments_sub, Y_posts_sub = [], [], [], []

	# Get data from database
	for doc in get_redditor_collection().find({}):

		# Skip is account contains no comment and no posts
		if len(doc['comments']) < 1 and len(doc['posts']) < 1:
			continue

		X_comments_sub_obj = get_comment_subreddit_data(X_comments, Y_comments, doc)
		X_posts_sub_obj = get_post_subreddit_data(X_posts, Y_posts, doc)

		# Add subreddit data
		X_comments_sub.append(X_comments_sub_obj)
		X_posts_sub.append(X_posts_sub_obj)
		if doc['is_bot']:
			Y_comments_sub.append(1)
			Y_posts_sub.append(1)
		else:
			Y_comments_sub.append(0)
			Y_posts_sub.append(0)
	
	print('------------------------------------------------------------')
	print('num posts: ' + str(len(X_posts)))
	print('\tnum bot posts: ' + str(Y_posts.count(1)))
	print('\tnum user posts: ' + str(Y_posts.count(0)))

	print('num comments: ' + str(len(X_comments)))
	print('\tnum bot comments: ' + str(Y_comments.count(1)))
	print('\tnum user comments: ' + str(Y_comments.count(0)))
	print('------------------------------------------------------------')

	# Create metric graphs
	#create_metric_graphs(X_comments, Y_comments, X_posts, Y_posts)

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


def get_post_subreddit_data(X_posts, Y_posts, doc):
	X_posts_sub_obj = ''
	for post in doc['posts']:
		X_posts_sub_obj = X_posts_sub_obj + post['subreddit'] + ' '
		X_posts.append(post['title'])
		if doc['is_bot']:
			Y_posts.append(1)
		else:
			Y_posts.append(0)
	return X_posts_sub_obj


def get_comment_subreddit_data(X_comments, Y_comments, doc):
	X_comments_sub_obj = ''
	for comment in doc['comments']:
		X_comments_sub_obj = X_comments_sub_obj + comment['subreddit'] + ' '
		X_comments.append(comment['body'])
		if doc['is_bot']:
			Y_comments.append(1)
		else:
			Y_comments.append(0)
	return X_comments_sub_obj


# Creates the interactive visualization
def create_visualizations():

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
	data[:,0] = Y_comments
	data[:,1] = X_comments
	for d in data:
		if d[0] == 0:
			d[0] = 'normal'
		else:
			d[0] = 'bot'

	df = pd.DataFrame({'label':data[:,0],'text':data[:,1]})
	print(df)

	corpus = (st.CorpusFromPandas(df, category_col='label', text_col='text', nlp=st.whitespace_nlp_with_sentences)
          .build()
          .get_unigram_corpus()
          .compact(st.ClassPercentageCompactor(term_count=2,
                                               term_ranker=st.OncePerDocFrequencyRanker)))
	html = st.produce_characteristic_explorer(
		corpus,
		category='normal',
		category_name='Normal',
		not_category_name='Bot'
	)
	open('comment_text_chart.html', 'wb').write(html.encode('utf-8'))

# Create pie chart of number of normal and bot users
def create_metric_graphs(X_comments, Y_comments, X_posts, Y_posts):

	# Create post pie chart
	labels = 'Normal', 'Bot'
	sizes = [Y_posts.count(0), Y_posts.count(1)]
	colors = ['blue', 'red']
	explode = (0, 0.1)
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
	plt.axis('equal')
	plt.show()

	# Create comment pie chart
	labels = 'Normal', 'Bot'
	sizes = [Y_comments.count(0), Y_comments.count(1)]
	colors = ['blue', 'red']
	explode = (0, 0.1)
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
	plt.axis('equal')
	plt.show()

# Create histogram of cake day distributions
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

	data = np.array([['','Num'], ['Bots',len(bot_cake_days)], ['Users',len(normal_cake_days)]])

	print(pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:]))

	df = pd.DataFrame({'Number of accounts':['Bot', 'Normal'], 'val':[len(bot_cake_days), len(normal_cake_days)]})
	df.plot.bar(x='Number of accounts', y='val', rot=0)

	df_bots["date"] = df_bots["date"].astype("datetime64")
	df_bots.groupby([df_bots["date"].dt.year, df_bots["date"].dt.month]).count().plot(kind="bar")

	df_normal["date"] = df_normal["date"].astype("datetime64")
	df_normal.groupby([df_normal["date"].dt.year, df_normal["date"].dt.month]).count().plot(kind="bar")

	print(df_bots)
	print(df_normal)

	visualize(df_bots)
	visualize(df_normal)

# Create histogram of time of day of comments
def create_comment_histogram():

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
	"""
	Visualize a dataframe with a date column.

	Parameters
	----------
	df : Pandas dataframe
	column_name : str
	    Column to visualize
	color : str
	title : str
	"""
	plt.figure(figsize=(20, 10))
	ax = (df[column_name].groupby(df[column_name].dt.hour)
	                     .count()).plot(kind="bar", color=color)
	ax.set_facecolor('#eeeeee')
	ax.set_xlabel("hour of the day")
	ax.set_ylabel("count")
	ax.set_title(title)
	plt.show()
	# plt.savefig('visuals/' + title + '.png')

# Classify coments based on dates (unsuccessful)
def classify_comment_dates():

	X, Y = [], []

	# Get data from database
	for doc in get_redditor_collection().find({}):
		if doc['is_bot']:
			for comment in doc['posts']:
				X.append(comment['created_utc'])
				Y.append(1)
		else:
			for comment in doc['posts']:
				X.append(comment['created_utc'])
				Y.append(0)

	X = np.array(X).reshape(-1,1)
	Y = np.array(Y).reshape(-1,1)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=31)

	# Create pipeline: word vectorizer => create tfidf model => Run NB classifier
	clf = GaussianNB()

	# Train model
	clf.fit(X_train, Y_train)

	# Predict test data
	Y_pred = clf.predict(X_test)
	print(confusion_matrix(Y_pred, Y_test))
	print('accuracy: ' + str(accuracy_score(Y_test, Y_pred)))
	print(classification_report(Y_pred, Y_test))

# Create histogram of comment karma
def avg_comment_account():

	bot_comments, normal_comments = [], []

	# Get data from database
	for doc in get_redditor_collection().find({}):
		if doc['is_bot']:
			if len(doc['comments']) > 0:
				t = (len(doc['comments']),)
				bot_comments.append(t)
		else:
			if len(doc['comments']) > 0:
				t = (len(doc['comments']),)
				normal_comments.append(t)

	n_bins = 50

	fig, axes = plt.subplots(nrows=2, ncols=2)
	ax0, ax1, ax2, ax3 = axes.flatten()

	# Make a multiple-histogram of data-sets with different length.
	x_multi = [bot_comments, normal_comments]
	print(x_multi)
	ax3.hist(x_multi, n_bins, histtype='bar')
	ax3.set_title('different sample sizes')

	fig.tight_layout()
	plt.show()

	"""
	labels = ['bot']

	df_bots = pd.DataFrame.from_records(bot_comments, columns=labels)
	df_normal = pd.DataFrame.from_records(normal_comments, columns=labels)

	df = pd.DataFrame.from_records(bot_comments, columns=labels)
	df['normal'] = normal_comments

	df.plot.hist(rot=0, bins=50)
	df_normal.plot.hist(rot=0, bins=50)

	plt.show()
	"""

start = time.time()

classify_all()

print(time.time() - start)