# Exploration vs Exploitation in Insurance

Every insurer runs a version of the explore-versus-exploit problem without
naming it. An underwriting team has several inspection strategies it could
deploy across a portfolio, and only a finite budget of surveyor hours to spend.
Each visit is a bet: stick with the method that has worked so far, or try a
different one that *might* uncover risk more cheaply. The multi-armed bandit is
the cleanest mathematical model of exactly this tension, and the exploration
rate `ε` is the dial that sets how often the team is willing to try something
other than its current favourite.

A very low rate such as `ε = 0.01` behaves like an underwriter who has settled
into one inspection method and almost never deviates. On the surface this looks
disciplined and efficient — and when one strategy is clearly best, it is. But it
carries a hidden cost: if a better strategy exists, the team will almost never
sample it often enough to notice. In our ESG experiment this failure is visible.
The two strongest retrofits, heat-pump and BMS optimisation, sit close together
(true means 0.71 and 0.68), and an agent that barely explores can lock onto the
*second*-best option early and never gather the evidence to correct itself. Too
little exploration is not caution; it is a quiet, compounding loss.

The opposite extreme, `ε = 0.30`, is the team that experiments constantly —
roughly one in three inspections spent on something other than the best known
strategy. This feels open-minded, but it is expensive. A third of the budget is
permanently committed to inferior options long after the evidence has made the
winner obvious, so cumulative regret keeps climbing in a straight line. Constant
experimentation is only rational while you are still genuinely uncertain; past
that point it is waste dressed up as diligence. The honest operating model sits
in between and changes over time: **epsilon decay** — explore aggressively early,
when you have little evidence and every option is plausible, then steadily shift
budget toward the strategy that keeps proving itself. That curve mirrors how a
maturing insurance portfolio actually *should* behave, and it is the single most
transferable lesson from Week 1.

ESG retrofit selection adds a constraint that sharpens all of this: a deadline.
A building owner facing a MEES (Minimum Energy Efficiency Standards) compliance
cut-off cannot afford the low-`ε` trap of quietly committing to a mediocre
retrofit, because there is no second budget cycle in which to recover. When a
deadline is approaching, the rational response is to *deliberately* widen
exploration in the time that remains — to make sure the genuinely uncertain
options have been tested before the window closes, rather than defaulting to the
familiar choice. Note carefully where this belongs: the urgency is a property of
the *decision-maker's policy*, not of the environment's payoffs. The retrofit
either reduces a building's energy intensity or it does not; that physical
reward does not change because a deadline is near. Conflating the two — letting
the deadline inflate the modelled reward — would quietly turn a clean,
well-understood bandit into a non-stationary problem and undermine the very
guarantees that make the result trustworthy. Keeping the urgency in the policy,
and the reward honest, is what lets the same model speak credibly to an
underwriter, an investor, and a reviewer at the same time.

---

## Reuse notes

- **README:** the three-`ε` framing (too little / too much / decay) plus the
  heat-pump-vs-solar result is a ready-made "Why this matters" section.
- **LinkedIn:** lead with the underwriter analogy; the deadline paragraph is the
  bridge from ML to a business audience.
- **Investor storytelling:** the point that *the right amount of experimentation
  depends on how distinguishable your options are* generalises directly to
  product, pricing, and go-to-market bets.
- **Research notes:** the deliberate separation of urgency (policy) from reward
  (environment) is the principled position to defend, and the natural lead-in to
  contextual bandits in Week 4.
