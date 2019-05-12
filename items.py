class Redditor:

	username = ''
	post_karma = 0
	comment_karma = 0
	cake_day = 0
	is_bot = False
	comments = []
	posts = []

	def Redditor(self):
		self.username = ''
		self.post_karma = 0
		self.comment_karma = 0
		self.cake_day = 0
		self.is_bot = False
		self.comments = []
		self.posts = []

	def __setitem__(self, key, value):
		self[key] = value

