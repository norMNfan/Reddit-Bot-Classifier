import re, datetime, time, praw, datetime, json, csv
from items import Redditor
import pymongo
import urllib.request

reddit = praw.Reddit('bot1', user_agent='bot1 user agent')

# Scrape use information and store it in mongodb
def scrape_users(usernames, is_bot):

	# Filter users that are already in database
	already_scraped = get_usernames_from_database()
	authors = []
	for username in usernames:
		if username not in already_scraped:
			authors.append(username)

	# Loop through authors
	for author in authors:

		print(author)
		if author == 'AutoModerator':
			continue

		redditor = Redditor()

		# try/except because some users have been deleted
		try:
			user = reddit.redditor(author)

			# Get user metadata
			redditor['username'] = author
			redditor['post_karma'] = user.link_karma
			redditor['comment_karma'] = user.comment_karma
			redditor['cake_day'] = user.created_utc
			redditor['is_bot'] = is_bot

		# Account has been deleted
		except:
			continue

		# Get user comments
		comment_counter = 500
		last_utc = '1428624000'
		comments = []

		# Loop through all comments between 1428624000 (4/10/2015) to 1523318400 (4/10/18)
		# These are the dates that the bots were active
		while comment_counter == 500:
			comment_counter = 0

			# Get the comment data from pushshift api
			url = 'https://api.pushshift.io/reddit/comment/search?after=' + last_utc + '&before=1523318400&size=500&author=' + author
			webURL = urllib.request.urlopen(url)
			contents = webURL.read()
			encoding = webURL.info().get_content_charset('utf-8')
			data = json.loads(contents.decode(encoding))

			# Add each comment to the user
			for comment in data['data']:

				# In case the subreddit has been deleted
				try:
					subreddit = comment['subreddit']
				except:
					subreddit = ''

				comment_object = {
					'body':comment['body'],
					'created_utc':comment['created_utc'],
					'score':comment['score'],
					'subreddit':subreddit
				}
				comments.append(comment_object)
				comment_counter = comment_counter + 1
				last_utc = str(comment['created_utc']+1)

		print('num comments: ' + str(len(comments)))
		redditor['comments'] = comments

		# Get user posts
		post_counter = 500
		last_utc = '1428624000'
		posts= []

		# Loop through all posts between 1428624000 (4/10/2015) to 1523318400 (4/10/18)
		# These are the dates that the bots were active
		while post_counter == 500:
			post_counter = 0

			url = 'https://api.pushshift.io/reddit/submission/search?after=' + last_utc + '&before=1523318400&size=500&author=' + author
			webURL = urllib.request.urlopen(url)
			contents = webURL.read()
			encoding = webURL.info().get_content_charset('utf-8')
			data = json.loads(contents.decode(encoding))

			# Add each post to the user
			for post in data['data']:

				# In case the subreddit has been deleted
				try:
					subreddit = post['subreddit']
				except:
					subreddit = ''

				post_object = {
					'created_utc':post['created_utc'],
					'num_comments':post['num_comments'],
					'over_18':post['over_18'],
					'score':post['score'],
					'subreddit':subreddit,
					'title':post['title']
				}

				# Some posts don't have selftext
				try:
					post_object['selftext'] = post['selftext']
				except KeyError:
					post_object['selftext'] = ''

				posts.append(post_object)
				post_counter = post_counter + 1
				last_utc = str(post['created_utc']+1)

		print('num posts: ' + str(len(posts)))
		redditor['posts'] = posts

		# Insert user into db
		client = pymongo.MongoClient('mongodb://localhost:27017')
		db = client['redditors']
		collection = db['redditors']
		collection.update({'username':redditor['username']}, dict(redditor), upsert=True)
		client.close()

# Get all usernames already in database
# This is so we skip these usernames when scraping later
def get_usernames_from_database():

	client = pymongo.MongoClient('mongodb://localhost:27017')
	db = client['redditors']
	collection = db['redditors']
	documents = collection.find({},{'username':True})
	client.close()

	return [document['username'] for document in documents]
    
# Get the names of all the bots
# These names are taken from the file: bot_accounts.txt
def get_bot_names():
	# read bot file
	with open('bot_accounts.txt', 'r') as dat:
		lines = dat.readlines()

	bot_usernames = []

	for line in lines:
		bot_usernames.append(line.split('\t')[0].split('/')[1])

	# get reddit users by name
	bot_redditors = []

	for name in bot_usernames:
		bot_redditors.append(reddit.redditor(name).name)

	bot_redditors.reverse()
	
	return bot_redditors

# Get account names from random reddit users
# We search all comments starting from 4/10/2015 0:00:00 (UTC)
def get_account_names(limit=500):

	last_utc = '1428624000'
	authors = []

	while len(authors) < limit:

		#  Get a lsit of usernames
		url = 'https://api.pushshift.io/reddit/comment/search?after=' + last_utc + '&before=1523318400&size=500'
		webURL = urllib.request.urlopen(url)
		contents = webURL.read()
		encoding = webURL.info().get_content_charset('utf-8')
		data = json.loads(contents.decode(encoding))

		# Collect usernames
		last_utc = ''
		for comment in data['data']:
			authors.append(comment['author'])
			last_utc = str(comment['created_utc']+1)

	return authors[:limit]

# Scrape bot and user accounts and insert their data into the db
def scrape_accounts():

	MAX_ACCOUNTS = 500

	bot_names = get_bot_names()
	account_names = get_account_names(MAX_ACCOUNTS)

	scrape_users(account_names, False)
	scrape_users(bot_names, True)

scrape_accounts()