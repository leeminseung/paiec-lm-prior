"""Split the gap between the two platform submissions into "the draw was easier" and "the predictor was better".

Inputs are the per-pair Brier scores the platform printed for round 1 (fixed prior 0.575) and round 2 (Sonnet prior).
floor = mean over pairs of r(1-r), the Brier of an oracle that knows each pair's success rate r and nothing about
single items. Round 1 predicts the constant 0.575 at budget 0, so r = (0.575^2 - B0) / 0.15 exactly. Round 2's budget-0
predictions vary by item, so its r comes from B31 instead: r(1-r) = 0.955 * B31 (the ratio floor/B31 seen in round 1),
which leaves r or 1-r; the branch is taken from round 1's pairs on the same benchmark, and both extremes are reported.
The round-1 predictor's expected score on round 2's pairs is simulated: x ~ Bin(b, r) own successes, other subjects on
the same benchmark pooled the same way, Brier = r(1-r) + E[(p - r)^2].

Usage: python -I platform_draws.py
"""
import itertools

import numpy as np

W = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
BUDGETS = [0, 1, 3, 7, 15, 31]
M, ALPHA, A0 = 0.575, 4.0, 4.0
R1 = {  # subject: (benchmark, ALC, [B0, B1, B3, B7, B15, B31]) as printed by the platform
    "221473": ("119137", 0.154668, [0.206534, 0.152748, 0.154753, 0.144532, 0.146134, 0.143815]),
    "238893": ("830060", 0.209701, [0.293125, 0.231600, 0.193673, 0.190000, 0.188061, 0.197216]),
    "294931": ("342307", 0.269818, [0.273125, 0.312933, 0.244150, 0.272259, 0.261597, 0.243176]),
    "382927": ("119137", 0.251933, [0.237898, 0.248093, 0.262086, 0.248297, 0.256020, 0.252444]),
    "614225": ("893575", 0.194875, [0.216080, 0.191236, 0.180538, 0.190496, 0.209023, 0.190085]),
    "732864": ("213669", 0.138411, [0.320280, 0.217117, 0.131605, 0.083844, 0.066922, 0.064850]),
    "765104": ("800730", 0.191043, [0.319514, 0.217526, 0.226478, 0.168972, 0.134896, 0.095179]),
    "809941": ("869474", 0.252892, [0.244911, 0.244910, 0.246914, 0.278128, 0.247093, 0.249919]),
    "895786": ("869474", 0.258125, [0.265168, 0.254347, 0.245966, 0.286610, 0.246199, 0.249839])}
R2 = {
    "243003": ("800730", 0.039498, [0.047213, 0.039807, 0.042767, 0.035882, 0.037578, 0.035702]),
    "269692": ("800730", 0.111002, [0.104547, 0.105647, 0.112856, 0.113139, 0.113829, 0.114529]),
    "423661": ("342307", 0.268297, [0.262787, 0.297250, 0.273752, 0.260178, 0.246501, 0.264818]),
    "431933": ("830060", 0.123589, [0.113183, 0.111362, 0.117695, 0.131007, 0.135758, 0.131059]),
    "471957": ("869474", 0.265123, [0.340291, 0.266801, 0.261806, 0.257031, 0.252589, 0.234489]),
    "707538": ("213669", 0.017738, [0.056628, 0.034403, 0.016267, 0.007095, 0.002247, 0.000723]),
    "723524": ("893575", 0.060515, [0.100955, 0.078107, 0.054991, 0.044236, 0.045733, 0.058059]),
    "801207": ("830060", 0.214180, [0.188515, 0.211486, 0.203138, 0.238829, 0.217214, 0.211949]),
    "941033": ("119137", 0.226977, [0.211650, 0.222899, 0.244900, 0.217083, 0.228812, 0.230731])}


def expected_round1_alc(pairs, sims=20000, seed=0):
    """pairs: {subject: (benchmark, r)} -> {subject: expected ALC of the round-1 predictor}."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, (bench, r) in pairs.items():
        peers = [rp for sp, (bp, rp) in pairs.items() if bp == bench and sp != s]
        alc = 0.0
        for w, b in zip(W, BUDGETS):
            x = rng.binomial(b, r, sims)
            S = sum(rng.binomial(b, rp, sims) for rp in peers) if peers else np.zeros(sims)
            center = (S + A0 * M) / (b * len(peers) + A0)
            p = (x + ALPHA * center) / (b + ALPHA)
            alc += w * (r * (1 - r) + np.mean((p - r) ** 2))
        out[s] = alc
    return out


def main():
    r1 = {s: (b, (M * M - c[0]) / (2 * M - 1)) for s, (b, _, c) in R1.items()}
    floor1 = np.mean([r * (1 - r) for _, r in r1.values()])
    exp1 = expected_round1_alc(r1)
    print(f"round 1 draw: score {np.mean([v[1] for v in R1.values()]):.4f}, floor {floor1:.4f},"
          f" expected score of the round-1 predictor on it {np.mean(list(exp1.values())):.4f} (simulation check)")
    bench_r1 = {}
    for b, r in r1.values():
        bench_r1.setdefault(b, []).append(r)
    low = {}
    for s, (b, _, c) in R2.items():
        q = min(0.25, 0.955 * c[-1])
        low[s] = (b, (1 - (1 - 4 * q) ** 0.5) / 2)
    branch = {s: (b, r if abs(r - np.mean(bench_r1[b])) <= abs(1 - r - np.mean(bench_r1[b])) else 1 - r)
              for s, (b, r) in low.items()}
    floor2 = np.mean([r * (1 - r) for _, r in branch.values()])
    score2 = np.mean([v[1] for v in R2.values()])
    print(f"round 2 draw: score {score2:.4f}, floor about {floor2:.4f}")
    print("  estimated pair success rates:", {s: round(r, 2) for s, (b, r) in branch.items()})
    exp = expected_round1_alc(branch)
    alcs = []
    for flips in itertools.product([0, 1], repeat=len(low)):
        pairs = {s: (b, 1 - r if f else r) for (s, (b, r)), f in zip(low.items(), flips)}
        alcs.append(np.mean(list(expected_round1_alc(pairs, sims=4000).values())))
    print(f"  round-1 predictor's expected score on round 2's pairs: {np.mean(list(exp.values())):.4f}"
          f" (branches from round 1's benchmarks); over all {len(alcs)} branch choices {min(alcs):.4f} to {max(alcs):.4f}")
    print(f"  per pair, round-1 predictor expected vs round 2 actual:")
    for s, (b, r) in branch.items():
        print(f"    {s} {b} r~{r:.2f}: {exp[s]:.3f} vs {R2[s][1]:.3f}")


if __name__ == "__main__":
    main()
