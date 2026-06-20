# Mathematical Foundations — Multi-Armed Bandits

> Audience: future contributors, students, and developers who want to understand
> the mathematics underlying `bandit-risk`. This document explains the theory
> from first principles, not the implementation. It is written at a graduate
> level but aims to remain accessible to a motivated beginner. Notation follows
> Sutton & Barto and Lattimore & Szepesvári where possible.

---

## 1. What is a Multi-Armed Bandit?

### Formal definition

A **stochastic multi-armed bandit** is a tuple $(\mathcal{A}, \{P_a\}_{a \in \mathcal{A}})$ where:

- $\mathcal{A} = \{1, 2, \dots, K\}$ is a finite set of $K$ **actions** (arms).
- For each arm $a$, $P_a$ is a fixed probability distribution over rewards, with
  mean $\mu_a = \mathbb{E}_{R \sim P_a}[R]$.

The name comes from a gambler facing $K$ slot machines ("one-armed bandits"),
each with an unknown payout rate, who must decide which arms to pull.

### Interaction protocol

The agent interacts with the bandit over $T$ discrete rounds. At each round
$t = 1, 2, \dots, T$:

1. The agent selects an arm $A_t \in \mathcal{A}$.
2. The environment draws a reward $R_t \sim P_{A_t}$, independently of all past
   rounds.
3. The agent observes **only** $R_t$ — the reward of the chosen arm. It does not
   observe what the other arms would have paid (this is the *bandit feedback* or
   "partial information" setting, as opposed to *full information*).

### Notation summary

| Symbol | Meaning |
|---|---|
| $K$ | number of arms |
| $\mathcal{A} = \{1,\dots,K\}$ | action set |
| $A_t$ | arm chosen at round $t$ |
| $R_t$ | reward received at round $t$ |
| $P_a$ | reward distribution of arm $a$ |
| $\mu_a$ | true mean reward of arm $a$, $\mathbb{E}[R \mid A=a]$ |
| $\mu^*$ | mean of the best arm, $\max_a \mu_a$ |
| $a^*$ | an optimal arm, $\arg\max_a \mu_a$ |
| $Q_t(a)$ | the agent's *estimate* of $\mu_a$ at round $t$ |
| $N_t(a)$ | number of times arm $a$ has been pulled before round $t$ |
| $\Delta_a$ | suboptimality gap, $\mu^* - \mu_a$ |

### Objective

The agent does **not** know the means $\mu_a$. Its goal is to maximise expected
cumulative reward over $T$ rounds:

$$\max \; \mathbb{E}\left[\sum_{t=1}^{T} R_t\right] = \max \; \mathbb{E}\left[\sum_{t=1}^{T} \mu_{A_t}\right].$$

### Optimal action

If the means were known, the optimal strategy is trivial: always pull the best
arm $a^* = \arg\max_{a} \mu_a$, achieving expected reward $T\mu^*$. The entire
difficulty comes from the means being **unknown** and having to be *estimated
from noisy samples while simultaneously acting*.

---

## 2. Exploration vs Exploitation

The agent maintains an estimate $Q_t(a)$ of each arm's mean. At any round it
faces a tension:

- **Exploitation** — pull the arm that currently *looks* best,
  $\arg\max_a Q_t(a)$, to earn reward now.
- **Exploration** — pull a less-certain arm to *improve the estimate* $Q_t(a)$,
  which may reveal a better arm and earn more reward later.

### The role of uncertainty

Pure exploitation is dangerous because $Q_t(a)$ is only an estimate. Early on,
with few samples, $Q_t(a)$ can be far from $\mu_a$ by chance. Formally, if arm
$a$ has been pulled $N_t(a)$ times, the sample mean estimate has standard error

$$\text{SE}(Q_t(a)) = \frac{\sigma_a}{\sqrt{N_t(a)}},$$

where $\sigma_a$ is the standard deviation of $P_a$. The estimate tightens only
as $\sqrt{N_t(a)}$ grows — so an arm pulled few times carries large uncertainty,
and an arm that *looks* slightly worse may actually be better. Exploration is the
act of paying a short-term cost to reduce this uncertainty. Every bandit
algorithm is, at heart, a different rule for **how to spend samples on reducing
uncertainty about the arms that matter.**

---

## 3. The ε-Greedy Policy

ε-greedy is the simplest exploration rule. With a fixed parameter
$\varepsilon \in [0,1]$:

$$
A_t =
\begin{cases}
\text{a uniformly random arm in } \mathcal{A}, & \text{with probability } \varepsilon \quad (\text{explore})\\[4pt]
\arg\max_{a} Q_t(a), & \text{with probability } 1-\varepsilon \quad (\text{exploit})
\end{cases}
$$

### Action selection probabilities

Let $a^{\text{greedy}}_t = \arg\max_a Q_t(a)$ be the current greedy arm (assume a
unique maximizer for clarity). The probability the policy selects an arbitrary
arm $a$ is:

$$
\pi_t(a) =
\begin{cases}
(1-\varepsilon) + \dfrac{\varepsilon}{K}, & a = a^{\text{greedy}}_t \\[8pt]
\dfrac{\varepsilon}{K}, & a \neq a^{\text{greedy}}_t
\end{cases}
$$

The $\varepsilon/K$ term appears for *every* arm because the random branch picks
uniformly over all $K$ arms — including, with probability $\varepsilon/K$, the
greedy one.

### Estimate update: the incremental mean

`bandit-risk` estimates $Q$ as the running sample mean of observed rewards. After
pulling arm $a$ and observing reward $R$, the update is

$$Q_{\text{new}}(a) = Q_{\text{old}}(a) + \frac{1}{N(a)}\big(R - Q_{\text{old}}(a)\big).$$

This is algebraically the sample mean but computed **incrementally**, in $O(1)$
time and memory, without storing past rewards. Derivation: if $Q_n$ is the mean
of the first $n$ rewards, then

$$
Q_{n+1} = \frac{1}{n+1}\sum_{i=1}^{n+1} R_i
= \frac{1}{n+1}\left( R_{n+1} + n Q_n \right)
= Q_n + \frac{1}{n+1}\left( R_{n+1} - Q_n \right).
$$

The term $(R - Q)$ is a *prediction error*; the step size $1/N(a)$ shrinks as the
arm accumulates data, so later samples move the estimate less. (Replacing
$1/N(a)$ with a constant step size $\alpha$ yields an exponential recency-weighted
average — useful for non-stationary bandits, but not used here.)

### Limiting cases

- **$\varepsilon = 0$ (pure greedy).** The agent never explores. It commits to
  whichever arm first looks good. If early noise makes a suboptimal arm's estimate
  exceed the true best arm's, the agent can pull the wrong arm *forever*, never
  collecting the samples that would correct it. Expected regret is then $\Theta(T)$
  — linear and unbounded — in the worst case.
- **$\varepsilon = 1$ (pure random).** The agent ignores $Q$ entirely and pulls
  uniformly at random every round. It learns the means accurately but never
  *uses* that knowledge. Expected per-step regret is the average gap
  $\frac{1}{K}\sum_a \Delta_a$, constant in $t$, so cumulative regret is again
  linear in $T$.

Both extremes give linear regret. The useful regime is $0 < \varepsilon < 1$, and
better still, an $\varepsilon$ that **changes over time** (next section).

---

## 4. ε-Decay

A fixed $\varepsilon$ is suboptimal: early on you want lots of exploration
(estimates are poor), but late on, once the best arm is clear, continued
exploration is pure waste. **Decaying** $\varepsilon$ over time captures this.

### Exponential decay

`bandit-risk` uses multiplicative (exponential) decay with a floor:

$$\varepsilon_{t+1} = \max\big(\varepsilon_{\min},\; \varepsilon_t \cdot d\big), \qquad 0 < d \le 1.$$

Unrolling (ignoring the floor) gives $\varepsilon_t = \varepsilon_0\, d^{\,t}$,
i.e. exponential decay toward $0$. The floor $\varepsilon_{\min}$ keeps a small
nonzero exploration rate so the agent can still recover if it has been unlucky.

### Linear decay

An alternative schedule decreases $\varepsilon$ by a fixed amount each step until
it reaches the floor:

$$\varepsilon_t = \max\big(\varepsilon_{\min},\; \varepsilon_0 - c\,t\big), \qquad c > 0.$$

Exponential decay drops quickly then levels off; linear decay declines at a
constant rate. Exponential is generally preferred when you expect the best arm to
become clear early.

### Why decay often outperforms fixed ε — theoretical intuition

The classic result (Auer, Cesa-Bianchi & Fischer, 2002) is that a schedule
$\varepsilon_t = \min(1, cK / (d^2 t))$ — decaying like $1/t$ — achieves
**logarithmic** expected regret, $O(\log T)$, matching the optimal rate up to
constants. The intuition: the *total* exploration spent is
$\sum_t \varepsilon_t \approx \sum_t \frac{1}{t} \approx \log T$, which grows
without bound (so every arm is sampled infinitely often and the best arm is
eventually identified almost surely) but grows *slowly enough* that the regret
from exploration stays logarithmic rather than linear. A fixed $\varepsilon$
spends $\varepsilon T$ on exploration — linear — and therefore cannot beat linear
regret. Decay is what lets exploration cost grow sublinearly while still
guaranteeing identification of the best arm.

---

## 5. Cumulative Regret

We cannot evaluate an agent by reward alone, because the achievable reward
depends on the (unknown) means. Instead we measure **regret**: how much worse the
agent did than an oracle that always pulls $a^*$.

### Definition

The **instantaneous (pseudo-)regret** at round $t$ is

$$r_t = \mu^* - \mu_{A_t},$$

and the **cumulative regret** over $T$ rounds is

$$\boxed{\,R_T = \sum_{t=1}^{T} \left(\mu^* - \mu_{A_t}\right).\,}$$

### Every symbol

- $R_T$ — total regret after $T$ rounds. Lower is better; $R_T = 0$ iff the best
  arm was pulled every round.
- $T$ — the time horizon (number of rounds).
- $\mu^* = \max_a \mu_a$ — the true mean of the optimal arm.
- $A_t$ — the arm the agent actually chose at round $t$.
- $\mu_{A_t}$ — the true mean of that chosen arm.
- $\mu^* - \mu_{A_t} = \Delta_{A_t} \ge 0$ — the per-round gap.

### Crucial subtlety: true means, not observed rewards

Regret is defined using the **true means** $\mu$, not the noisy realised rewards
$R_t$. A lucky draw from a bad arm still incurs full regret $\Delta_{A_t}$,
because regret measures the *quality of the decision*, not the luck of the
outcome. This is exactly why the experiment code computes regret via
`env.true_mean(arm)` and `env.optimal_mean`, never from the value returned by
`step()`.

### The gap decomposition

Grouping rounds by which arm was pulled gives an equivalent and illuminating form.
Let $N_T(a)$ be the number of pulls of arm $a$ in $T$ rounds and
$\Delta_a = \mu^* - \mu_a$:

$$\mathbb{E}[R_T] = \sum_{a=1}^{K} \mathbb{E}[N_T(a)]\,\Delta_a.$$

Regret accrues **only** from suboptimal arms, weighted by how suboptimal each is.
This explains the curve shapes in the Week 1 plots:

- A **flat** curve ⇒ the agent stopped pulling high-gap arms (it found and
  committed to the best arm).
- A **linearly rising** curve ⇒ suboptimal arms are still pulled at a constant
  rate (the signature of fixed $\varepsilon$, where $\mathbb{E}[N_T(\text{bad})]$
  grows linearly in $T$).
- A curve that **bends from steep to flat** ⇒ learning: heavy early exploration,
  then concentration on the best arm (the signature of decay).

### Why cumulative regret is the primary metric

It is horizon-aware (captures the whole learning trajectory, not just the final
policy), it is comparable across algorithms and problems, and it has a rich
theory: the Lai–Robbins lower bound (1985) proves *no* algorithm can do better
than $\Omega(\log T)$ asymptotically, giving an absolute yardstick against which
ε-greedy, UCB1, and Thompson Sampling can all be measured.

---

## 6. Why `InspectionBanditEnv` Is a Genuine Bandit

The environment models choosing one of 5 inspection strategies for a property
drawn from a portfolio. The reward for strategy $a$ is

$$R_t \sim \mathcal{N}(\mu_a, \sigma_a^2), \quad \text{clipped to } [0,1],$$

with $(\mu_a, \sigma_a)$ **fixed constants** independent of $t$ and of all prior
actions. Mathematically, the joint law of $R_t$ given $A_t = a$ is the same
distribution $P_a$ at every round:

$$P(R_t \mid A_t = a, \; \text{history}_{1:t-1}) = P(R_t \mid A_t = a) = P_a.$$

Because the reward law is conditionally independent of history given the current
action, the problem satisfies the bandit definition of §1. There is no state
variable that the action mutates. ✅

---

## 7. Why `ESGBanditEnv` Is Also a Genuine Bandit

The environment models a single-shot retrofit recommendation for one building in
one budget cycle, across 6 retrofit arms, again with fixed Gaussian rewards
$R_t \sim \mathcal{N}(\mu_a, \sigma_a^2)$ clipped to $[0,1]$.

### Independence assumption

The key modelling assumption is **independence across rounds**: recommending
solar PV in one cycle does not alter the reward distribution of a heat-pump
recommendation in another. Each round represents a *different* building/budget
decision, so there is no carried-over state. Under this assumption the same
conditional-independence identity as §6 holds, and ESG retrofit selection is a
genuine stationary bandit. ✅

### Why introducing state would make it an MDP

Consider the rejected "MEES deadline bonus": let the reward gain a term when
`steps_remaining < 100` and the arm has been pulled fewer than 3 times. Then

$$R_t = f\big(A_t, \underbrace{t}_{\text{time}}, \underbrace{N_t(A_t)}_{\text{history}}\big),$$

so the reward law now depends on the round index and on past actions. The process
becomes **non-stationary** (time-dependent) and **history-dependent** — precisely
a Markov Decision Process, where the "state" encodes time-remaining and per-arm
pull counts, and actions transition that state. The stationary-bandit regret
theory of §§3–5 would no longer apply. This is why the mechanic was removed:
deadline-aware urgency must live in the *agent's policy*, never in the
*environment's reward*, if the bandit guarantees are to remain valid.

---

## 8. Statistical Assumptions

The theory above relies on a precise set of assumptions, all satisfied by the
`bandit-risk` environments:

- **Independent rewards.** $R_t \perp R_s$ for $t \neq s$ given the chosen arms.
  Implemented by drawing each reward from a fresh call to the random generator.
- **Stationary reward distributions.** $P_a$ does not change with $t$. Guaranteed
  by holding $(\mu_a, \sigma_a)$ as fixed constants.
- **Gaussian sampling.** Rewards are $\mathcal{N}(\mu_a, \sigma_a^2)$, then
  clipped to $[0,1]$ to represent a normalised score. (Clipping introduces a
  small bias when $\mu_a$ is near a boundary; with the chosen parameters this is
  negligible, but it is worth noting for rigour.)
- **Random seeds and reproducibility.** Each environment and agent accepts a
  `seed`, fixing the pseudo-random stream so that experiments are exactly
  reproducible — essential for scientific comparison across algorithms and for
  CI determinism.

---

## 9. Computational Complexity

Let $K$ be the number of arms and $T$ the horizon.

### ε-greedy per-round cost

- **Action selection.** The explore branch is $O(1)$ (sample one integer). The
  exploit branch computes $\arg\max_a Q_t(a)$ over $K$ values: $O(K)$.
- **Update.** The incremental-mean update touches a single arm's $Q$ and $N$:
  $O(1)$.

So each round is $O(K)$, dominated by the argmax.

### Totals

| Quantity | Cost |
|---|---|
| Per-round time | $O(K)$ |
| Total time over $T$ rounds | $O(KT)$ |
| Memory | $O(K)$ — two length-$K$ arrays ($Q$ and $N$) |

The memory is $O(K)$ rather than $O(KT)$ precisely because of the incremental
mean: rewards are never stored. This $O(K)$ space, $O(KT)$ time profile is what
makes bandit methods practical at scale.

---

## 10. Connection to Reinforcement Learning

Bandits are the simplest member of the reinforcement-learning family. The
`bandit-risk` curriculum follows the natural ladder of increasing generality:

$$
\textbf{Bandits} \;\rightarrow\; \textbf{Contextual Bandits} \;\rightarrow\; \textbf{Tabular RL} \;\rightarrow\; \textbf{Deep RL} \;\rightarrow\; \textbf{Safe RL}
$$

- **Bandits** — one decision, no state. Learn $\mu_a$. *(This library, Month 1.)*
- **Contextual bandits** — a context $x_t$ is observed before each choice;
  reward is $\mathbb{E}[R \mid a, x]$. Still single-shot (no transitions), but the
  optimal arm depends on context. *(LinUCB, Week 4.)*
- **Tabular RL (MDPs)** — actions cause *state transitions*; reward and next
  state depend on the current state. Requires value functions / Bellman
  equations. *(Where maintenance scheduling correctly belongs.)*
- **Deep RL** — MDPs with state/action spaces too large to tabulate; function
  approximation via neural networks.
- **Safe RL** — RL with explicit safety constraints on the policy, crucial for
  high-stakes physical systems.

### Why this progression

Each step adds exactly one new ingredient: contextual bandits add **context**;
tabular RL adds **state transitions**; deep RL adds **function approximation**;
safe RL adds **constraints**. Mastering the bandit case in isolation means the
exploration–exploitation machinery is fully understood *before* the additional
complexity of state and transitions is layered on. Skipping it tends to produce
practitioners who can run deep-RL libraries but cannot diagnose why exploration
is failing — because the failure is usually a bandit-level phenomenon.

---

## 11. References

**Textbooks**

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
  (2nd ed.). MIT Press. — Chapter 2 covers multi-armed bandits, ε-greedy, and the
  incremental mean update used here.
- Lattimore, T. & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge
  University Press. — The comprehensive modern reference; rigorous regret theory.

**Foundational papers**

- Lai, T. L. & Robbins, H. (1985). "Asymptotically efficient adaptive allocation
  rules." *Advances in Applied Mathematics*, 6(1), 4–22. — The $\Omega(\log T)$
  regret lower bound.
- Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). "Finite-time analysis of the
  multiarmed bandit problem." *Machine Learning*, 47, 235–256. — UCB1 and the
  $O(\log T)$ analysis of ε-greedy with a decaying schedule.
- Thompson, W. R. (1933). "On the likelihood that one unknown probability exceeds
  another in view of the evidence of two samples." *Biometrika*, 25(3/4),
  285–294. — The original posterior-sampling idea behind Thompson Sampling.
- Badanidiyuru, A., Kleinberg, R. & Slivkins, A. (2013). "Bandits with Knapsacks."
  *FOCS*. — Budget/resource-constrained bandits; the literature that a "deadline
  reward bonus" would actually fall under (and why such a mechanic is not novel).

**Surveys**

- Slivkins, A. (2019). "Introduction to Multi-Armed Bandits." *Foundations and
  Trends in Machine Learning*, 12(1–2), 1–286.
- Bubeck, S. & Cesa-Bianchi, N. (2012). "Regret Analysis of Stochastic and
  Nonstochastic Multi-armed Bandit Problems." *Foundations and Trends in Machine
  Learning*, 5(1), 1–122.
