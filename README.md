# Reddit-Bot-Classifier

This project classifies accounts on Reddit as bots or regular users.

I originally created this project for an independent study project during my last semester of University but since then have adapted it to be used by others.

There are two main files used in this project:

### user_scraper.py

This file is used to scrape user data from reddit and insert it into mongo db.


### classifier.py

This file performs all of the classification algorithms and creates all of the visualizations for the data.

## How to use this code

##### 1. Clone this repository

##### 2. Download [mongodb](https://www.mongodb.com/download-center/community)

##### 3. Create Reddit account and then create a [developer application](https://www.reddit.com/prefs/apps)
Step by step guide [here](https://github.com/reddit-archive/reddit/wiki/OAuth2)

##### 4. Create a [praw.ini](https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html) file with your reddit account credentials
Pu the praw.ini file in your main directory. It should look something like this:

[bot1]

client_id=XXXXXXXXXXXXXX

client_secret=XXXXXXXXXXXXXXXXXXXXXXXXXXX

password=XXXXXXXXXX

username=XXXXXXXXXX

##### 5. Run the user_scraper.py

##### 6. Run the classifier_all() in classifier.py