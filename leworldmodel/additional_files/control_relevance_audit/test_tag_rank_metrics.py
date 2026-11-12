import unittest
from tag_rank_metrics import compare


class TagRankingTests(unittest.TestCase):
    def test_common_shift_cancels(self):
        r = compare([[1, 2, 2], [101, 102, 102]], [4, 1, 5])
        self.assertEqual((r["flips"], r["eligible"], r["regrets"]), (0, 2, [3, 3]))

    def test_reversal_has_physical_consequence(self):
        r = compare([[1, 2], [3, 0]], [1, 9])
        self.assertEqual((r["flips"], r["regrets"]), (1, [0, 8]))
