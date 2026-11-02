"""Read saved oracle branches without Torch, a simulator, or GPU access."""
import argparse
import csv
import json
import math
from pathlib import Path


def error(state, goal):
    angle = state[4] - goal[4]
    return math.hypot(state[2]-goal[2], state[3]-goal[3]), abs(math.atan2(math.sin(angle), math.cos(angle)))


def compare(decision, goal, scorer):
    traces = decision['candidate_state_traces']
    costs = decision[scorer + '_cost']
    if len(costs) != len(traces) or len(traces) < 2:
        raise ValueError('Candidate count mismatch')
    if not all(math.isfinite(x) for x in costs):
        raise ValueError('Nonfinite score')
    for trace in traces:
        if not trace or any(len(s) != 10 or not all(math.isfinite(x) for x in s) for s in trace):
            raise ValueError('Invalid full-state trace')
    selected = min(range(len(costs)), key=costs.__getitem__)
    chosen = traces[selected][-1]
    chosen_error = error(chosen, goal)
    rows = []
    for i, trace in enumerate(traces):
        state = trace[-1]
        xy, angle = error(state, goal)
        dx, da = chosen_error[0]-xy, chosen_error[1]-angle
        dominates = dx >= -1e-5 and da >= -1e-7 and (dx > 1e-5 or da > 1e-7)
        rows.append(dict(candidate=i, selected=selected, dominates=dominates,
            # Descriptive thresholds, not tuned success criteria.
            material=dominates and (dx >= 1.0 or da >= .02),
            gain_xy_px=dx, gain_angle_rad=da, xy_error=xy, angle_error=angle,
            score_gap=costs[i]-costs[selected],
            agent_x=state[0], agent_y=state[1], agent_vx=state[5], agent_vy=state[6],
            agent_endpoint_separation_px=math.hypot(state[0]-chosen[0], state[1]-chosen[1]),
            agent_block_center_distance_px=math.hypot(state[0]-state[2], state[1]-state[3]),
            block_vx=state[7], block_vy=state[8], block_omega=state[9]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    all_rows = []
    for qi, query in enumerate(report['queries']):
        for arm, result in query['arms'].items():
            for scorer in ('learned', 'oracle'):
                dominated = material = 0
                for decision in result['decisions']:
                    rows = compare(decision, query['goal'], scorer)
                    dominated += any(r['dominates'] for r in rows)
                    material += any(r['material'] for r in rows)
                    for row in rows:
                        all_rows.append(dict(query=qi, trajectory=arm, turn=decision['turn'], scorer=scorer, **row))
                    # Print all dominating candidates, avoiding a cherry-picked representative.
                    witnesses = [r for r in rows if r['dominates']]
                    print('RANKING_GAPS', qi, arm, scorer, decision['turn'], json.dumps(witnesses))
                print('RANKING_SUMMARY', qi, arm, scorer, json.dumps(dict(
                    decisions=len(result['decisions']), dominated=dominated, material=material)))
    if not all_rows:
        raise ValueError('No decision records to analyze')
    output = args.report.parent / 'candidate-ranking-detail.csv'
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    print('LIMITATION: material = Pareto dominance plus >=1px OR >=0.02rad gain; descriptive only.')
    print('LIMITATION: agent-to-block center distance is not contact or future controllability.')
    print('LIMITATION: decisions on diverged arm trajectories are not independent or same-state paired comparisons.')
    print('RANKING_DETAIL_COMPLETE', output)


if __name__ == '__main__':
    main()
