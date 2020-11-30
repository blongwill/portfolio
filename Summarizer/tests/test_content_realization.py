#!opt/python-3.6/bin/python3

import unittest
import sys
sys.path.append("../src")
from content_realization import realize_content

class TestContentRealization(unittest.TestCase):

    def test_realize_content(self):

        # TODO: fix to actually test
        value = 5
        self.assertEqual(value, 5)


if __name__ == '__main__':
	unittest.main()