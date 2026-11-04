import unittest
from attribute_candidate_errors import audit_decision, audit_report


def fixture(learned, oracle):
    return {'turn': 0, 'learned_cost': learned, 'oracle_cost': oracle,
            'candidate_state_traces': [[[0, 0, x, 0, 0, 0, 0, 0, 0, 0]] for x in (30, 5)]}


class AttributionTests(unittest.TestCase):
    def test_prediction_error_rescued(self):
        r = audit_decision(fixture([0, 1], [1, 0]), [0]*7)
        self.assertTrue(r['prediction_changes_choice_beyond_tie'])
        self.assertTrue(r['oracle_physically_dominates_learned'])
        self.assertFalse(r['real_endpoint_latent_choice_physically_dominated'])

    def test_score_error_with_exact_prediction(self):
        r = audit_decision(fixture([0, 1], [0, 1]), [0]*7)
        self.assertFalse(r['prediction_changes_choice_beyond_tie'])
        self.assertTrue(r['real_endpoint_latent_choice_physically_dominated'])

    def test_tradeoff_is_not_dominance(self):
        d = fixture([0, 1], [0, 1])
        d['candidate_state_traces'][1][0][4] = .5
        self.assertFalse(audit_decision(d, [0]*7)['real_endpoint_latent_choice_physically_dominated'])

    def test_ties_and_nonfinite(self):
        self.assertFalse(audit_decision(fixture([0, 1], [1e-8, 0]), [0]*7)['prediction_changes_choice_beyond_tie'])
        with self.assertRaises(ValueError):
            audit_decision(fixture([float('nan'), 0], [0, 1]), [0]*7)

    def test_only_matched_learned_history(self):
        r = audit_report({'queries': [{'episode': 7, 'goal': [0]*7, 'arms': {
            'learned': {'decisions': [fixture([0, 1], [1, 0])]},
            'oracle': {'decisions': [fixture([0, 1], [0, 1])]}}}]})
        self.assertEqual(r['summary']['decisions'], 1)
        self.assertEqual(r['unique_episodes'], 1)


if __name__ == '__main__':
    unittest.main()
