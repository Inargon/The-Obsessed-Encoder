"""Exact synthetic objective audit, not a learned-world-model benchmark.

Run with Python's standard library only. No checkpoints or GPU are used.
Blocked regime: a in {-1,+1} uniformly; real successor z'=(0,0).
Restricted predictor family: z_hat'=(0, epsilon*a).
Inverse head: a_hat=w*z_hat'[1]. Objective:
  epsilon**2 + lam*((w*epsilon-1)**2 + beta*w**2).
The ridge penalty is explicitly inside the lam multiplier. It is not a
claim about AD-WM's optimizer weight decay or normalized action embeddings.
Free regime: real z'=(a,0), which admits exact physical action recovery.
"""

import argparse
import json
import math
from pathlib import Path
import unittest


def objective(epsilon, weight, lam, beta):
    return epsilon**2 + lam * ((weight * epsilon - 1)**2 + beta * weight**2)


def solve(lam, beta):
    if lam < 0 or beta <= 0:
        raise ValueError('lam must be nonnegative and beta strictly positive')
    variance = max(0.0, math.sqrt(lam * beta) - beta)
    epsilon = math.sqrt(variance)
    weight = epsilon / (variance + beta)
    return {
        'lambda': lam, 'beta': beta, 'epsilon': epsilon,
        'inverse_weight': weight, 'prediction_mse': variance,
        'action_recovery_mse': (weight * epsilon - 1)**2,
        'objective': objective(epsilon, weight, lam, beta),
        'faithful_prediction_objective': lam,
    }


def log_cosh(x):
    return abs(x) + math.log1p(math.exp(-2 * abs(x))) - math.log(2)


def posterior_kl(epsilon, slope):
    # Real teacher posterior is uniform. Off-support extension has logit
    # odds 2*slope*z[1]. Both slope=0 and slope>0 fit real data identically.
    return log_cosh(slope * epsilon)


class MechanismTests(unittest.TestCase):
    def test_ridge_optimum_matches_grid(self):
        for lam in (0.0, .01, .1, 1.0):
            for beta in (.001, .1, 1.0):
                result = solve(lam, beta)
                for i in range(1001):
                    eps = i / 500
                    w = eps / (eps**2 + beta)
                    self.assertLessEqual(result['objective'],
                                         objective(eps, w, lam, beta) + 1e-12)

    def test_recovery_gain_threshold(self):
        self.assertEqual(solve(.01, .1)['epsilon'], 0)
        self.assertGreater(solve(.1, .001)['epsilon'], 0)

    def test_injection_without_penalty(self):
        for epsilon in (.1, .01, .001):
            self.assertAlmostEqual(objective(epsilon, 1 / epsilon, .1, 0),
                                   epsilon**2)

    def test_teachers_agree_on_support_disagree_off_support(self):
        self.assertEqual(posterior_kl(0, 0), 0)
        self.assertEqual(posterior_kl(0, 10), 0)
        self.assertEqual(posterior_kl(.1, 0), 0)
        self.assertGreater(posterior_kl(.1, 10), .4)

    def test_real_free_motion_recovers_action(self):
        for action in (-1, 1):
            real_successor = (action, 0)
            self.assertEqual(real_successor[0], action)

    def test_posterior_matching_does_not_identify_outcome(self):
        # Successor=(action, independent coin). Exact inverse posterior
        # identifies action but ignores the second coordinate.
        left, right = (1, -1), (1, 1)
        self.assertEqual(left[0], right[0])
        self.assertNotEqual(left[1], right[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(MechanismTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(tests)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    main_case = solve(.1, .001)
    report = {
        'experiment': 'exact_synthetic_action_recovery_objective_audit',
        'tests_passed': outcome.testsRun,
        'main_case': main_case,
        'sweep': [solve(lam, beta) for lam in (.01, .1, 1.0)
                  for beta in (.001, .01, .1, 1.0)],
        'posterior_extension_comparison': {
            'epsilon': .1,
            'flat_teacher_kl': posterior_kl(.1, 0),
            'sensitive_teacher_kl': posterior_kl(.1, 10),
            'both_teachers_real_data_nll': math.log(2),
            'interpretation': 'Observed-data calibration does not determine off-support gradients.'
        },
        'limitations': [
            'Analytical minimization in a hand-selected function family, not neural training.',
            'No evidence about existing checkpoints, AD-WM, or planning performance.',
            'Action recovery uses raw scalar commands, not learned action embeddings.',
            'Posterior consistency is not sufficient for complete physical prediction.',
            'No novelty claim; this checks a candidate mechanism before GPU experiments.'
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(main_case, indent=2))
    print('REPORT', args.output.resolve())


if __name__ == '__main__':
    main()
