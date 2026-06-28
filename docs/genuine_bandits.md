## Why Every Environment Here Is a Genuine Bandit

Most "reinforcement learning for industry" repositories contain a quiet bug that
isn't in the code — it's in the **problem definition**. They model something as a
multi-armed bandit when it is really a Markov Decision Process (MDP). This
section explains the distinction, why it matters, and how `bandit-risk` enforces
it.

### What defines a genuine bandit

A multi-armed bandit has one defining property:

> **The reward depends only on the action you take — never on history, elapsed
> time, or any evolving state.**

Formally, pulling arm *a* returns a reward drawn from a fixed distribution with
mean *μₐ*. That distribution does not change based on what you did last step, how
many steps remain, or any accumulated condition. Each decision is independent.

### Why the distinction matters

The moment reward depends on a state that *evolves in response to your actions*,
you no longer have a bandit — you have an MDP:

```
Bandit:   reward = f(action)
MDP:      reward = f(action, state),   and state changes over time
```

This is not pedantry. Bandit algorithms (ε-greedy, UCB1, Thompson Sampling) and
their regret guarantees are **derived under the stationarity assumption**. Apply
them to a problem with hidden evolving state and the guarantees silently become
invalid: the regret bounds you cite no longer hold, and the agent can converge to
the wrong policy while appearing to work. The code runs; the conclusions are
fiction.

### Why many repositories accidentally implement MDPs

The slip is easy to make because many real problems *feel* like "pick the best
option repeatedly." But if choosing option A today changes what option B is worth
tomorrow, state is leaking into the reward. Common accidental-MDP patterns:

- **Reward depends on time remaining** (e.g. a deadline bonus). Now reward is a
  function of the step index, not just the action.
- **Reward depends on how often an arm was already pulled** (e.g. diminishing
  returns, forced-exploration bonuses). Now reward is a function of history.
- **Pulling an arm changes the environment** (e.g. maintenance lowers future
  failure rates). Now there is genuine state transition.

Each of these turns a clean stationary bandit into a non-stationary or sequential
problem requiring different tools.

### How this repository validates every environment

Every environment in `bandit-risk` must pass an explicit **integrity test**
before it is included:

1. State the decision in one sentence.
2. Ask: *does the reward distribution of any arm change based on prior actions,
   step count, or accumulated state?*
3. If **no** → genuine bandit, include it.
4. If **yes** → it is an MDP/POMDP, exclude it (or defer it to a later library
   designed for sequential problems).

This test is applied in the docstring of each environment and recorded in the
project's design notes.

### Why invalid environments are intentionally excluded

Two environments originally planned for Month 1 were **cut** after failing the
test:

- **Maintenance scheduling** — choosing to defer maintenance changes the asset's
  future condition. That is state transition → an MDP. Deferred to a later
  tabular-RL library.
- **Sensor incident sequencing** — the value of inspecting a sensor depends on
  which sensors were already inspected. That is history dependence → a POMDP.
  Deferred to a later safe-RL library.

A "MEES deadline reward bonus" was also designed and then deliberately removed
from `ESGBanditEnv`, because making reward depend on steps-remaining would have
converted a clean bandit into a non-stationary one. Deadline-aware behaviour
belongs in the *agent's policy*, not the environment's reward.

Excluding these isn't a limitation — it's the point. A library that only contains
correctly classified problems is one whose results you can trust.

### Validated environments

| Environment | Decision | Arms | Best arm | Bandit? | Why |
|---|---|---|---|---|---|
| `InspectionBanditEnv` | Which inspection strategy to deploy | 5 | Arm 0 — Full structural (μ=0.70) | ✅ Yes | Each survey is independent; no evolving state |
| `ESGBanditEnv` | Which ESG retrofit to recommend | 6 | Arm 2 — Heat pump ASHP (μ=0.71) | ✅ Yes | Single-shot recommendation; reward is stationary |
| Maintenance scheduling | When to service an asset | — | — | ❌ No (MDP) | Deferring maintenance changes future asset state |
| Sensor incident sequencing | Order to inspect sensors | — | — | ❌ No (POMDP) | Reward depends on which sensors already inspected |

---

## Architecture

`bandit-risk` separates three concerns so each can evolve independently:

- **Agents** (`bandit_risk/agents/`) — decision algorithms. An agent receives
  only `n_arms` and rewards; it never sees arm labels or true means. The same
  agent works unchanged across every environment.
- **Environments** (`bandit_risk/envs/`) — the problems. All domain semantics
  (what each arm means, its reward distribution) live here. Environments expose a
  public regret API (`true_mean`, `optimal_mean`) so experiments never reach into
  private state.
- **Utilities** (`bandit_risk/utils/`) — shared, reusable code such as
  `plot_regret`, so experiment scripts contain only experiment logic.

This mirrors the agent–environment interface at the heart of reinforcement
learning, and means adding a new algorithm or a new problem is a localized change.

## Learning progression

The library is built as a curriculum. Each week adds one algorithm of increasing
sophistication, evaluated on the same kind of regret plot so improvements are
directly comparable:

```
ε-greedy  →  UCB1  →  Thompson Sampling  →  LinUCB (contextual)
 random      optimism    Bayesian             uses context features
 explore     under       posterior            for personalised
             uncertainty  sampling             decisions
```

## Project philosophy

1. **Correctness before cleverness.** Only model problems that are genuinely
   what they claim to be.
2. **Everything tested.** Each module ships with its own test file.
3. **Built in public.** Progress is shared openly as a learning record.
4. **Honest claims.** No overstated novelty; results are reported as measured.
