import unittest

from user_scraper import get_bot_usernames
from user_scraper import get_usernames_currently_in_db


class Tests(unittest.TestCase):

	def test_get_bot_names(self):
		self.assertEqual(len(get_bot_usernames()), 939, "There should be 939 bot account names")

	def test_get_regular_names(self):
		self.assertEqual(len(get_usernames_currently_in_db()), 1343, "There should be 1343 regular account names")


if __name__ == '__main__':
	unittest.main()