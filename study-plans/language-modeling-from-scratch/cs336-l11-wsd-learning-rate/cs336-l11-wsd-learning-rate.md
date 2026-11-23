# Generate a Warmup-Stable-Decay Schedule

The learning rate is one of the most sensitive hyperparameters in training, and almost no large model uses a fixed one. A warmup-stable-decay (WSD) schedule ramps the rate up, holds it at a peak for most of training, then brings it down quickly at the end. The heart of this problem is getting the boundaries right, so that warmup ends exactly at the peak and decay ends exactly at the final rate.

## Why the learning rate changes during training

At the very start of training the weights are random and the optimizer's running statistics are empty, so large steps can push the model somewhere it never recovers from. A warmup phase starts with a small rate and raises it gradually, giving Adam-style optimizers time to collect reliable estimates of the gradient scale.

Late in training the opposite problem appears. A large rate keeps the weights bouncing around a good region without settling into it. Lowering the rate lets the model descend into the bottom of that region, and the loss usually drops noticeably during this final decay.

## Ramp up, hold, then cool down

Let $T$ be the total number of steps, $w$ the warmup length, $s$ the stable length, and $d = T - w - s$ the decay length. Let $\eta_{\max}$ be the peak rate and $r$ the final ratio, so training ends at $r \eta_{\max}$.

**Warmup** rises linearly. At zero-based warmup index $t$ the rate is

$$
\eta_t = \eta_{\max} \frac{t + 1}{w}
$$

so the first step already has a small positive rate and the last warmup step reaches the peak exactly.

**Stable** holds the rate at $\eta_{\max}$ for $s$ steps. Holding the peak lets the model make fast progress while the loss is still far from its minimum.

**Decay** follows a half cosine from the peak down to $r \eta_{\max}$. At zero-based decay index $j$, progress through the decay is $p_j = j / (d - 1)$, and the rate is

$$
\eta_{w+s+j} = \eta_{\max} \left[ r + (1 - r) \frac{1 + \cos(\pi p_j)}{2} \right]
$$

The cosine term starts at 1 when $p_j = 0$ and ends at 0 when $p_j = 1$, so the first decay step is still at the peak and the last one is exactly $r \eta_{\max}$. The half cosine falls slowly at first, fastest in the middle and slowly again near the end, so the model spends a few steps near both the peak and the final rate instead of jumping between them. With a single decay step the formula $j / (d - 1)$ would divide by zero, and since the schedule must end at the final rate, that lone step takes progress 1.

## Why WSD instead of a plain cosine schedule

The common alternative is a cosine schedule that starts decaying right after warmup and reaches its minimum at the very last step. Its shape depends on the total step count, so the number of training steps must be fixed before training starts, and continuing a finished run means restarting the schedule.

WSD separates the long stable phase from a short decay. Any checkpoint from the stable phase can be turned into a finished model by running a short decay from it, and training can continue from the same stable checkpoint later. This makes WSD convenient for scaling-law studies, where one long run can produce models at many data sizes, and it is used in models such as MiniCPM (Hu et al., 2024).

## A nine-step example

Take $T = 9$, $w = 4$, $s = 1$, a peak of 0.4 and a final ratio of 0.1, so the decay has $d = 4$ steps.

Warmup gives $0.4 \times 1/4 = 0.1$, then 0.2, 0.3 and 0.4. The stable phase gives one step at 0.4. The decay steps have progress 0, 1/3, 2/3 and 1, with cosine terms

$$
\frac{1 + \cos 0}{2} = 1, \qquad \frac{1 + \cos(\pi / 3)}{2} = 0.75, \qquad \frac{1 + \cos(2\pi / 3)}{2} = 0.25, \qquad \frac{1 + \cos \pi}{2} = 0
$$

so the decay rates are $0.4 \times (0.1 + 0.9 \times 1) = 0.4$, $0.4 \times 0.775 = 0.31$, $0.4 \times 0.325 = 0.13$ and $0.4 \times 0.1 = 0.04$. The full schedule is

$$
[0.1, \ 0.2, \ 0.3, \ 0.4, \ 0.4, \ 0.4, \ 0.31, \ 0.13, \ 0.04]
$$

The peak appears three times in a row, because the last warmup step, the stable step and the first decay step all sit at $\eta_{\max}$. The middle two decay steps show the cosine shape, since the drop from 0.31 to 0.13 is larger than the drops on either side.

## Common mistakes to avoid

- **Starting warmup at zero.** The first warmup rate is $\eta_{\max} / w$, so no step trains with a zero learning rate.
- **Dividing decay progress by $d$.** Progress uses $d - 1$, so the last decay step lands exactly on the final rate.
- **Mishandling a one-step decay.** With $d = 1$ the single decay step takes progress 1 and equals $r \eta_{\max}$.
- **Decaying to zero instead of the final ratio.** The cosine shrinks only the part above $r \eta_{\max}$.
- **Using a linear decay.** A straight-line decay gives different values at every interior decay step.
