"""CPU-only GRPO lab: one-token categorical answers, no KL or reward model."""

import argparse
import math
import random
import statistics


# One independent logit row per prompt; answer index is the generated token.
TASKS = (
    (17, 6, (102, 112, 96, 100)),
    (13, 7, (81, 91, 97, 89)),
    (19, 4, (72, 74, 78, 76)),
    (16, 8, (118, 124, 128, 132)),
)
CLIP_EPS = 0.2


def softmax(logits):
    peak = max(logits)
    weights = [math.exp(z - peak) for z in logits]
    total = sum(weights)
    return [w / total for w in weights]


def group_advantages(rewards):
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards)
    return [(r - mean) / std if std else 0.0 for r in rewards]


def clipped_term(ratio, advantage):
    clipped_ratio = min(1 + CLIP_EPS, max(1 - CLIP_EPS, ratio))
    raw = ratio * advantage
    clipped = clipped_ratio * advantage
    # At an exact boundary, use the unclipped branch's one-sided derivative.
    return min(raw, clipped), raw > clipped


def rollout(logits, task, group_size, rng):
    a, b, answers = task
    old_p = softmax(logits)
    sampled = rng.choices(range(len(answers)), weights=old_p, k=group_size)
    rewards = [int(answers[token] == a * b) for token in sampled]
    batch = [(token, math.log(old_p[token]), advantage)
             for token, advantage in zip(sampled, group_advantages(rewards))]
    return batch, rewards


def objective_and_grad(logits, batch):
    """Mean clipped objective and its analytic gradient w.r.t. the logits."""
    p = softmax(logits)
    grad = [0.0] * len(logits)
    total = clipped_count = 0
    for token, old_logp, advantage in batch:
        ratio = math.exp(math.log(p[token]) - old_logp)
        value, clipped = clipped_term(ratio, advantage)
        total += value
        clipped_count += clipped
        if not clipped:
            # d log p[token] / d logit[j] = indicator(j == token) - p[j].
            for j, prob in enumerate(p):
                grad[j] += advantage * ratio * ((j == token) - prob)
    n = len(batch)
    return total / n, [g / n for g in grad], clipped_count


def expected_reward(policy):
    # Exact expectation on the training prompts, not held-out accuracy.
    return statistics.mean(softmax(row)[answers.index(a * b)]
                           for row, (a, b, answers) in zip(policy, TASKS))


def train(steps, group_size, updates_per_batch, seed):
    rng = random.Random(seed)
    policy = [[0.0] * len(answers) for _, _, answers in TASKS]
    print(f"Initial expected reward: {expected_reward(policy):.3f}")
    print("step  sampled_reward  mixed_groups  clipped_terms  gradient_norm  expected_reward")
    for step in range(1, steps + 1):
        # Generate all groups before updating any policy row (one snapshot).
        rollouts = [rollout(row, task, group_size, rng)
                    for row, task in zip(policy, TASKS)]
        mean_reward = statistics.mean(r for _, rewards in rollouts for r in rewards)
        mixed = sum(len(set(rewards)) > 1 for _, rewards in rollouts)
        clipped = 0
        for update in range(updates_per_batch):
            grad_sq = 0.0
            for row, (batch, _) in zip(policy, rollouts):
                _, grad, count = objective_and_grad(row, batch)
                clipped += count
                grad_sq += sum(g * g for g in grad)
                for j, g in enumerate(grad):
                    # Gradient ascent; equivalent to descent on negative objective.
                    row[j] += 0.5 * g / len(policy)
            if update == 0:
                first_grad_norm = math.sqrt(grad_sq) / len(policy)
        denom = len(TASKS) * group_size * updates_per_batch
        print(f"{step:>4}  {mean_reward:>14.3f}  {mixed:>5}/{len(TASKS)}"
              f"  {clipped / denom:>13.1%}  {first_grad_norm:>13.4f}"
              f"  {expected_reward(policy):>15.3f}")


def self_test():
    # Fixed fixture from Appendix H; training itself uses random rollouts.
    assert group_advantages([1, 0, 1, 0]) == [1, -1, 1, -1]
    for rewards in ([0] * 4, [1] * 4, [1]):
        advantages = group_advantages(rewards)
        batch = [(0, math.log(0.25), a) for a in advantages]
        assert objective_and_grad([0.0] * 4, batch) == (0, [0.0] * 4, 0)
    assert clipped_term(1.30, 1) == (1.20, True)
    assert clipped_term(0.70, -1) == (-0.80, True)
    assert clipped_term(0.70, 1) == (0.70, False)
    assert clipped_term(1.30, -1) == (-1.30, False)
    batch = [(token, math.log(0.25), a)
             for token, a in enumerate(group_advantages([1, 0, 1, 0]))]
    for row, want_clipped in (([0.0] * 4, 0), ([0.8, -0.8, 0.0, 0.0], 2)):
        _, grad, clipped = objective_and_grad(row, batch)
        assert clipped == want_clipped and any(grad)
        for j, analytic in enumerate(grad):
            plus, minus = row[:], row[:]
            h = 1e-5
            plus[j] += h
            minus[j] -= h
            numerical = (objective_and_grad(plus, batch)[0]
                         - objective_and_grad(minus, batch)[0]) / (2 * h)
            assert math.isclose(analytic, numerical, abs_tol=1e-8)
    print("PASS: mixed/equal rewards, clipping in both directions, and finite-difference gradients")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--updates-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.group_size, args.updates_per_batch) < 1:
        parser.error("steps, group size, and updates per batch must be positive")
    if args.self_test:
        self_test()
    else:
        train(args.steps, args.group_size, args.updates_per_batch, args.seed)


if __name__ == "__main__":
    main()
