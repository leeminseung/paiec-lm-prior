"""Count-based predictors that also use other systems' revealed results on the same benchmark.

- j2: IvanYamasaki/paiec-yamasaki "J2" as published (lines/J_robusto/submit_J2_min/model.py):
      center = (S_others + 4 * 0.5) / (N_others + 4); p = (s + 2 * center) / (n + 2)
- shrunk_peers: our shrinkage baseline with its training-fitted prior m and strength alpha, whose center is first
      pulled toward the other systems' revealed rate on the same benchmark (the J2 pooling idea):
      center = (S_others + a0 * m) / (N_others + a0); p = (s + alpha * center) / (n + alpha)
s, n = the pair's own successes and revealed results; S_others, N_others = those of every other system on the same
benchmark_id. A system is matched by its whole subject dict, as the platform's subject identity is.
Both read only (input, labeled).
"""


def _counts(subject, item, labeled):
    bench = item.get("benchmark_id")
    s = n = so = no = 0
    for (sub, it), y in labeled or []:
        if it.get("benchmark_id") != bench:
            continue
        if sub == subject:
            s, n = s + int(y), n + 1
        else:
            so, no = so + int(y), no + 1
    return s, n, so, no


def make_j2(a=2.0, a0=4.0):
    def predict(inp, labeled):
        s, n, so, no = _counts(inp[0], inp[1], labeled)
        center = (so + a0 * 0.5) / (no + a0)
        return (s + a * center) / (n + a)
    return predict


def make_shrunk_peers(m, alpha, a0=4.0):
    def predict(inp, labeled):
        s, n, so, no = _counts(inp[0], inp[1], labeled)
        center = (so + a0 * m) / (no + a0)
        return (s + alpha * center) / (n + alpha)
    return predict
