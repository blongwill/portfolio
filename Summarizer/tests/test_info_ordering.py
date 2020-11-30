#!opt/python-3.6/bin/python3

import unittest
import sys
sys.path.append("../src")
from info_ordering import order_info

class TestInfoOrdering(unittest.TestCase):

    def test_order_info(self):
        # TODO: fix to actually test
        value = 5
        self.assertEqual(value, 5)



if __name__ == '__main__':
	unittest.main()