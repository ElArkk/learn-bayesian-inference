# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "arviz>=0.22",
#     "httpx>=0.28",
#     "marimo[lsp]>=0.23.15,<0.24",
#     "matplotlib>=3.10",
#     "numpy>=2.2",
#     "pandas>=2.2",
#     "pymc>=5.25; sys_platform != 'emscripten'",
#     "scipy>=1.15",
#     "torch>=2.7; sys_platform != 'emscripten'",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import importlib
    import json
    import math
    import os
    import sys
    from dataclasses import dataclass
    from functools import lru_cache

    import httpx
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch
    from scipy.special import expit, logsumexp
    from scipy.stats import beta, multivariate_normal, norm


    return (
        dataclass,
        httpx,
        importlib,
        json,
        logsumexp,
        mo,
        norm,
        np,
        os,
        plt,
        sys,
    )


@app.cell(hide_code=True)
def _(dataclass, logsumexp, np):
    @dataclass(frozen=True)
    class RunResult:
        figure: object | None
        summary: str
        table: object | None = None

    def normal_logpdf(x, mean, sd):
        x = np.asarray(x, dtype=float)
        return -0.5 * np.log(2.0 * np.pi * sd**2) - 0.5 * ((x - mean) / sd) ** 2

    def normal_mean_log_likelihood(mu, data, sigma):
        return np.sum(normal_logpdf(np.asarray(data), mu, sigma), axis=0)

    def normal_mean_log_posterior(mu, data, sigma, prior_mu, prior_sigma):
        return normal_mean_log_likelihood(mu, data, sigma) + normal_logpdf(
            mu, prior_mu, prior_sigma
        )

    def finite_difference_gradient(fn, x, eps=1e-5):
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            step = np.zeros_like(x)
            step[idx] = eps
            grad[idx] = (fn(x + step) - fn(x - step)) / (2.0 * eps)
        return float(grad) if grad.ndim == 0 else grad

    def gradient_ascent_demo(fn, start, learning_rate, steps):
        path = [float(start)]
        for _ in range(int(steps)):
            path.append(path[-1] + learning_rate * finite_difference_gradient(fn, path[-1]))
        return np.asarray(path)

    def metropolis_1d(log_target, start, proposal_sd, draws, rng):
        samples = np.empty(int(draws), dtype=float)
        accepted = np.zeros(int(draws), dtype=bool)
        current = float(start)
        current_lp = float(log_target(current))
        for i in range(int(draws)):
            proposal = current + rng.normal(0.0, proposal_sd)
            proposal_lp = float(log_target(proposal))
            if np.log(rng.uniform()) < proposal_lp - current_lp:
                current, current_lp = proposal, proposal_lp
                accepted[i] = True
            samples[i] = current
        return samples, accepted

    def autocorrelation_curve(x, max_lag=80):
        x = np.asarray(x, dtype=float)
        centered = x - x.mean()
        variance = np.dot(centered, centered)
        if variance == 0:
            return np.ones(int(max_lag) + 1)
        return np.asarray(
            [
                1.0
                if lag == 0
                else np.dot(centered[:-lag], centered[lag:]) / variance
                for lag in range(int(max_lag) + 1)
            ]
        )

    def approximate_ess(x, max_lag=200):
        ac = autocorrelation_curve(x, min(int(max_lag), len(x) - 1))
        positive = ac[1:][ac[1:] > 0]
        tau = max(1.0, 1.0 + 2.0 * positive.sum())
        return len(x) / tau

    def bivariate_logpdf(theta, rho=0.9, scales=(1.0, 1.0)):
        theta = np.asarray(theta, dtype=float)
        covariance = np.array(
            [
                [scales[0] ** 2, rho * scales[0] * scales[1]],
                [rho * scales[0] * scales[1], scales[1] ** 2],
            ]
        )
        precision = np.linalg.inv(covariance)
        return -0.5 * theta @ precision @ theta

    def bivariate_gradient(theta, rho=0.9, scales=(1.0, 1.0)):
        covariance = np.array(
            [
                [scales[0] ** 2, rho * scales[0] * scales[1]],
                [rho * scales[0] * scales[1], scales[1] ** 2],
            ]
        )
        return -np.linalg.solve(covariance, np.asarray(theta, dtype=float))

    def metropolis_nd(log_target, start, proposal_sd, draws, rng):
        current = np.asarray(start, dtype=float).copy()
        samples = np.empty((int(draws), len(current)))
        accepted = np.zeros(int(draws), dtype=bool)
        current_lp = float(log_target(current))
        for i in range(int(draws)):
            proposal = current + rng.normal(0.0, proposal_sd, size=current.shape)
            proposal_lp = float(log_target(proposal))
            if np.log(rng.uniform()) < proposal_lp - current_lp:
                current, current_lp = proposal, proposal_lp
                accepted[i] = True
            samples[i] = current
        return samples, accepted

    def leapfrog(position, momentum, step_size, steps, grad_log_target):
        q = np.asarray(position, dtype=float).copy()
        p = np.asarray(momentum, dtype=float).copy()
        trajectory = [q.copy()]
        p += 0.5 * step_size * grad_log_target(q)
        for step in range(int(steps)):
            q += step_size * p
            trajectory.append(q.copy())
            if step != int(steps) - 1:
                p += step_size * grad_log_target(q)
        p += 0.5 * step_size * grad_log_target(q)
        return q, -p, np.asarray(trajectory)

    def hamiltonian(log_target, position, momentum):
        return -float(log_target(position)) + 0.5 * float(np.dot(momentum, momentum))

    def hmc_transition(log_target, grad_log_target, current, step_size, steps, rng):
        momentum = rng.normal(size=np.asarray(current).shape)
        proposal, proposal_momentum, trajectory = leapfrog(
            current, momentum, step_size, steps, grad_log_target
        )
        old_h = hamiltonian(log_target, current, momentum)
        new_h = hamiltonian(log_target, proposal, proposal_momentum)
        accepted = np.log(rng.uniform()) < old_h - new_h
        return (proposal if accepted else np.asarray(current)), accepted, trajectory, new_h - old_h

    def u_turn(start, position, momentum):
        return float(np.dot(np.asarray(position) - np.asarray(start), momentum)) < 0.0

    def rhat_basic(chains):
        chains = np.asarray(chains, dtype=float)
        _m, n = chains.shape
        chain_means = chains.mean(axis=1)
        between = n * chain_means.var(ddof=1)
        within = chains.var(axis=1, ddof=1).mean()
        variance = ((n - 1) / n) * within + between / n
        return float(np.sqrt(variance / within))

    def gaussian_mixture_em(data, means, weights, sd, iterations):
        data = np.asarray(data, dtype=float)
        means = np.asarray(means, dtype=float).copy()
        weights = np.asarray(weights, dtype=float).copy()
        history = []
        responsibilities = np.zeros((len(data), len(means)))
        for _ in range(int(iterations)):
            log_joint = np.stack(
                [
                    np.log(weights[k] + 1e-12)
                    - 0.5 * ((data - means[k]) / sd) ** 2
                    - np.log(sd * np.sqrt(2.0 * np.pi))
                    for k in range(len(means))
                ],
                axis=1,
            )
            log_norm = logsumexp(log_joint, axis=1, keepdims=True)
            responsibilities = np.exp(log_joint - log_norm)
            counts = responsibilities.sum(axis=0) + 1e-12
            weights = counts / len(data)
            means = (responsibilities * data[:, None]).sum(axis=0) / counts
            history.append(float(log_norm.sum()))
        return means, weights, responsibilities, np.asarray(history)

    def shrinkage_estimates(observed, standard_errors, population_mean, population_sd):
        observed = np.asarray(observed, dtype=float)
        variances = np.asarray(standard_errors, dtype=float) ** 2
        prior_variance = float(population_sd) ** 2
        weight = prior_variance / (prior_variance + variances)
        return weight * observed + (1.0 - weight) * population_mean, weight

    def elbo_mc(log_joint, mean, log_sd, eps):
        sd = np.exp(log_sd)
        samples = mean + sd * eps
        log_q = -0.5 * eps**2 - log_sd - 0.5 * np.log(2.0 * np.pi)
        return float(np.mean(log_joint(samples) - log_q))

    return (
        RunResult,
        approximate_ess,
        autocorrelation_curve,
        bivariate_gradient,
        bivariate_logpdf,
        gaussian_mixture_em,
        gradient_ascent_demo,
        hamiltonian,
        leapfrog,
        metropolis_1d,
        metropolis_nd,
        normal_logpdf,
        normal_mean_log_posterior,
        rhat_basic,
        shrinkage_estimates,
        u_turn,
    )


@app.cell(hide_code=True)
def _(mo):
    COLORS = {
        "prior": "#6D5BD0",
        "likelihood": "#F59E0B",
        "posterior": "#0EA5A8",
        "sample": "#2563EB",
        "reject": "#DC2626",
        "accent": "#EC4899",
        "ink": "#172033",
    }

    def style_axes(ax, title=None, xlabel=None, ylabel=None):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        if title:
            ax.set_title(title, loc="left", fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    def finish_figure(fig):
        fig.patch.set_facecolor("#FCFCFD")
        fig.tight_layout()
        return fig

    mo.Html(
        """
        <style>
          :root { --lab-ink:#172033; --lab-teal:#0EA5A8; --lab-paper:#FCFCFD; }
          .lab-hero {padding:1.2rem 1.35rem;border-radius:18px;
            background:linear-gradient(135deg,#102A43,#0B6E75);color:white;
            box-shadow:0 12px 32px rgba(16,42,67,.16)}
          .lab-loop {font-size:1.05rem;letter-spacing:.03em;margin-top:.7rem}
          .lab-card {padding:1rem 1.15rem;border:1px solid #DCE3EC;border-radius:14px;
            background:#FCFCFD;margin:.5rem 0}
          .lab-tag {font-size:.75rem;text-transform:uppercase;letter-spacing:.09em;
            color:#0B6E75;font-weight:700}
          .takeaway {border-left:5px solid #0EA5A8;padding:.75rem 1rem;
            background:#EAF8F7;border-radius:8px}
          code {font-size:.92em}
        </style>
        """
    )
    return COLORS, finish_figure, style_axes


@app.cell(hide_code=True)
def _():
    LABS = [{'title': 'Lab 0 — How to use this inference course',
      'intuition': 'This notebook is a small system that you can disturb. Make a '
                   'prediction, run the system, inspect the result, change code, and '
                   'explain what changed.',
      'prediction': 'Which part of the loop will most often change your first intuition: '
                    'simulation, inspection, or code? Why?',
      'code': 'def learning_loop():\n    # TODO: return the five stages in order\n    ...',
      'solution': 'def learning_loop():\n'
                  "    return ['predict', 'simulate', 'inspect', 'code', 'explain']",
      'hints': ('The first stage happens before you see output.',
                'There are five lowercase words.',
                'Return a Python list.'),
      'quiz': 'Why must prediction come before simulation?',
      'answer': 'A prior prediction makes surprise visible. Surprise is useful evidence '
                'that your mental model must change.',
      'takeaway': 'Treat each plot as a test of a mental model. Mark a lab complete only '
                  'after you can explain the result without the plot.',
      'math': 'The loop is procedural. It has no new mathematical symbols.'},
     {'title': 'Lab 1 — A probability distribution is a landscape',
      'intuition': 'Density is height. Probability is area under the curve. A wide, low '
                   'region can contain more probability than a narrow, high region.',
      'prediction': 'If the Normal standard deviation grows by 3×, what happens to peak '
                    'height and to total area?',
      'code': 'def normal_log_density(x, mean, sd):\n'
              '    # TODO: use NumPy; do not use scipy.stats\n'
              '    ...',
      'solution': 'def normal_log_density(x, mean, sd):\n'
                  '    return -0.5*np.log(2*np.pi*sd**2) - 0.5*((x-mean)/sd)**2',
      'hints': ('Work in log space.',
                'Use -0.5 log(2πσ²) - 0.5((x-μ)/σ)².',
                'The result must work for an array x.'),
      'quiz': 'Can a point with greater density have less nearby probability mass?',
      'answer': 'Density height measures probability concentration per unit of daily '
                'demand at one location. Probability mass is area under the density over a '
                'stated sales interval, so it depends on both height and interval width.',
      'takeaway': 'A distribution is a landscape whose total area is one. Height and mass '
                  'answer different questions.',
      'math': 'For X ~ Normal(μ, σ), μ moves the center and σ controls spread. The '
              'integral of p(x) over an interval is its probability mass.'},
     {'title': 'Lab 2 — Likelihood reshapes the landscape',
      'intuition': 'The prior is the landscape before data. The likelihood scores each '
                   'possible parameter with the observed data. Their product gives the '
                   'posterior shape.',
      'prediction': 'Add a distant observation. Will the posterior move more when the '
                    'prior is narrow or wide?',
      'code': 'def log_likelihood(mu, x, sigma):\n'
              '    # TODO: sum Normal log densities\n'
              '    ...\n'
              '\n'
              'def log_posterior(mu, x, sigma, prior_mu, prior_sigma):\n'
              '    # TODO: likelihood + log prior\n'
              '    ...',
      'solution': 'def log_likelihood(mu, x, sigma):\n'
                  '    return '
                  'np.sum(-0.5*np.log(2*np.pi*sigma**2)-0.5*((np.asarray(x)-mu)/sigma)**2)\n'
                  '\n'
                  'def log_posterior(mu, x, sigma, prior_mu, prior_sigma):\n'
                  '    return log_likelihood(mu,x,sigma) '
                  '-0.5*np.log(2*np.pi*prior_sigma**2)-0.5*((mu-prior_mu)/prior_sigma)**2',
      'hints': ('Independent observations add in log space.',
                'log posterior = log likelihood + log prior + a constant.',
                'You do not need the normalization constant for parameter comparison.'),
      'quiz': 'Does the likelihood describe uncertainty about data or about μ?',
      'answer': 'With observed data fixed, it is a function of μ. It scores possible '
                'values of μ.',
      'takeaway': 'Bayesian updating multiplies prior mass by data compatibility and then '
                  'normalizes.',
      'math': 'p(μ|x) ∝ p(x|μ)p(μ). Here x is observed, μ is unknown, and σ is known.'},
     {'title': 'Lab 3 — Optimization means finding a point',
      'intuition': 'Optimization follows local slope to one high point. MLE uses only '
                   'likelihood. MAP also uses the prior.',
      'prediction': 'With a large learning rate, will ascent reach the mode faster or can '
                    'it jump across it?',
      'code': 'def gradient_ascent(fn, start, rate, steps):\n'
              '    x = float(start)\n'
              '    path = [x]\n'
              '    for _ in range(steps):\n'
              '        # TODO: finite-difference gradient and update x\n'
              '        ...\n'
              '    return np.asarray(path)',
      'solution': 'def gradient_ascent(fn, start, rate, steps):\n'
                  '    x=float(start); path=[x]; eps=1e-5\n'
                  '    for _ in range(steps):\n'
                  '        grad=(fn(x+eps)-fn(x-eps))/(2*eps)\n'
                  '        x=x+rate*grad; path.append(x)\n'
                  '    return np.asarray(path)',
      'hints': ('Estimate the slope from two nearby function values.',
                'g ≈ [f(x+ε)-f(x-ε)]/(2ε).',
                'For ascent, add rate × gradient.'),
      'quiz': 'What object does MAP return?',
      'answer': 'MAP returns one parameter value: the posterior mode. It does not return '
                'posterior uncertainty.',
      'takeaway': 'MLE and MAP are point optimization methods. Their paths can look like '
                  'neural-network training paths.',
      'math': 'θ_MAP = argmaxθ log p(θ|x). θ is the parameter; the gradient gives the '
              'local ascent direction.'},
     {'title': 'Lab 4 — Why MAP is not Bayesian uncertainty',
      'intuition': 'Two posteriors can have the same mode and very different widths. MAP '
                   'cannot show this difference.',
      'prediction': 'If two posteriors have equal modes but one has 10× the variance, will '
                    'MAP distinguish them?',
      'code': 'def laplace_sd(log_posterior, map_value, eps=1e-3):\n'
              '    # TODO: use local second curvature\n'
              '    ...',
      'solution': 'def laplace_sd(log_posterior, map_value, eps=1e-3):\n'
                  '    f0=log_posterior(map_value)\n'
                  '    '
                  'curvature=(log_posterior(map_value+eps)-2*f0+log_posterior(map_value-eps))/eps**2\n'
                  '    return np.sqrt(-1/curvature)',
      'hints': ('A narrow peak has more negative second curvature.',
                'variance ≈ -1 / second derivative of log posterior.',
                'Return the square root of variance.'),
      'quiz': 'What information does local curvature miss for a multimodal posterior?',
      'answer': 'It misses other modes, long tails, skew, and global geometry.',
      'takeaway': 'MAP says where one peak is. Posterior samples or an approximation can '
                  'also show width and shape.',
      'math': 'A Laplace approximation uses a Normal centered at θ_MAP. Its variance comes '
              'from local log-posterior curvature.'},
     {'title': 'Lab 5 — Random-walk MCMC',
      'intuition': 'Metropolis proposes a nearby point and accepts it by a probability '
                   'ratio. Repeated states are part of the chain, not errors.',
      'prediction': 'If proposal standard deviation grows by 10×, what happens to '
                    'acceptance and movement?',
      'code': 'def metropolis(log_target, start, proposal_sd, draws, rng):\n'
              '    current=float(start); samples=[]; accepted=0\n'
              '    for _ in range(draws):\n'
              '        # TODO: propose, compare log ratio, and save current\n'
              '        ...\n'
              '    return np.asarray(samples), accepted/draws',
      'solution': 'def metropolis(log_target,start,proposal_sd,draws,rng):\n'
                  '    current=float(start); samples=[]; accepted=0\n'
                  '    for _ in range(draws):\n'
                  '        proposal=current+rng.normal(0,proposal_sd)\n'
                  '        if np.log(rng.uniform()) < '
                  'log_target(proposal)-log_target(current):\n'
                  '            current=proposal; accepted+=1\n'
                  '        samples.append(current)\n'
                  '    return np.asarray(samples), accepted/draws',
      'hints': ('The proposal is symmetric.',
                'Accept when log(u) < log p(proposal) - log p(current).',
                'Always append the current state, also after rejection.'),
      'quiz': 'Why can high acceptance be a bad sign?',
      'answer': 'Very small proposals are often accepted but move slowly. Samples then '
                'contain repeated information.',
      'takeaway': 'A useful chain balances acceptance with distance moved. Sample count '
                  'alone does not measure information.',
      'math': 'For a symmetric proposal, α = min(1, p(θ′)/p(θ)). θ is current and θ′ is '
              'proposed.'},
     {'title': 'Lab 6 — Autocorrelation and effective sample size',
      'intuition': 'A chain can have many rows but few independent pieces of information. '
                   'Autocorrelation measures how much the past predicts the future.',
      'prediction': 'Which has more information: 5,000 sticky samples or 1,000 nearly '
                    'independent samples?',
      'code': 'def autocorrelation(x, lag):\n'
              '    x=np.asarray(x)-np.mean(x)\n'
              '    # TODO: normalized lagged dot product\n'
              '    ...',
      'solution': 'def autocorrelation(x, lag):\n'
                  '    x=np.asarray(x)-np.mean(x)\n'
                  '    if lag==0: return 1.0\n'
                  '    return np.dot(x[:-lag],x[lag:])/np.dot(x,x)',
      'hints': ('Center the sequence first.',
                'Compare x[:-lag] with x[lag:].',
                'Normalize by the total centered sum of squares.'),
      'quiz': 'Can ESS be greater than the raw sample count in this simple estimate?',
      'answer': 'Effective sample size should not exceed the number of stored draws in '
                'this intuition estimate. Strong autocorrelation reduces it, and the '
                'notebook clips numerical estimates at the draw count.',
      'takeaway': 'ESS translates a correlated chain into an approximate count of '
                  'independent samples.',
      'math': 'ESS ≈ N/(1+2Σρk). N is draw count and ρk is autocorrelation at lag k.'},
     {'title': 'Lab 7 — Two-dimensional posterior geometry',
      'intuition': 'Correlation turns a round target into a narrow diagonal valley. A '
                   'random walk wastes proposals across the narrow direction.',
      'prediction': 'Which target is harder for an isotropic random walk: correlation 0 or '
                    '0.99?',
      'code': 'def covariance(rho):\n    # TODO: return a 2×2 correlation matrix\n    ...',
      'solution': 'def covariance(rho):\n    return np.array([[1.0,rho],[rho,1.0]])',
      'hints': ('The diagonal contains variances.',
                'Both off-diagonal entries are rho.',
                'The matrix must be symmetric.'),
      'quiz': 'Why does a valid proposal often move only a short distance at high '
              'correlation?',
      'answer': 'The proposal scale must fit the narrow direction, so it also takes small '
                'steps along the long direction.',
      'takeaway': 'Posterior geometry controls sampling difficulty. Dimension makes this '
                  'mismatch more costly.',
      'math': 'ρ is correlation. Values near ±1 make one covariance direction narrow.'},
     {'title': 'Lab 8 — Gradients as a force field',
      'intuition': 'The log-density gradient points uphill. An optimizer can follow it '
                   'directly. A sampler can use the same local information while pursuing '
                   'a different goal.',
      'prediction': 'At point (2, 2) on a centered Gaussian, does the log-density gradient '
                    'point inward or outward?',
      'code': 'def numerical_gradient(fn, point, eps=1e-5):\n'
              '    point=np.asarray(point,dtype=float)\n'
              '    # TODO: one centered difference per coordinate\n'
              '    ...',
      'solution': 'def numerical_gradient(fn,point,eps=1e-5):\n'
                  '    point=np.asarray(point,dtype=float); g=np.zeros_like(point)\n'
                  '    for i in range(len(point)):\n'
                  '        step=np.zeros_like(point); step[i]=eps\n'
                  '        g[i]=(fn(point+step)-fn(point-step))/(2*eps)\n'
                  '    return g',
      'hints': ('Change one coordinate at a time.',
                'Use a centered difference for each coordinate.',
                'Store each partial derivative in an array.'),
      'quiz': 'The gradient points uphill. Why does following it directly lead to one '
              'optimized point, while posterior sampling must also represent regions away '
              'from that point? Explain the different goals without using concepts from '
              'later labs.',
      'answer': 'Gradient ascent uses the local uphill direction to change parameters '
                'toward a mode, so it returns one optimized point. Posterior sampling has '
                'a different goal: represent the full distribution, including '
                'lower-density regions that still contain probability mass. A gradient can '
                'guide a sampler, but the gradient alone does not define its full '
                'transition rule. The next lab introduces the additional HMC mechanism.',
      'takeaway': 'The same gradient can drive different algorithms because each algorithm '
                  'updates state differently.',
      'math': '∇log p(θ) is a vector of local slopes. Each component belongs to one '
              'coordinate of θ.'},
     {'title': 'Lab 9 — Momentum and Hamiltonian Monte Carlo',
      'intuition': 'HMC has two changing vectors. The parameter position θ says where the '
                   'particle is, and the temporary momentum r says how it is moving. The '
                   'posterior gradient changes momentum; the changed momentum then changes '
                   'position.',
      'prediction': 'Set both initial momentum components to zero while the particle '
                    'starts away from the mode. Will it stay still, move to the mode and '
                    'stop, or move through the mode?',
      'code': 'def energy(log_target, position, momentum):\n'
              '    # TODO: potential plus kinetic energy\n'
              '    ...',
      'solution': 'def energy(log_target,position,momentum):\n'
                  '    potential=-float(log_target(position))\n'
                  '    kinetic=0.5*float(np.dot(momentum,momentum))\n'
                  '    return potential+kinetic',
      'hints': ('Potential energy is negative log density.',
                'Kinetic energy is half the squared momentum for unit mass.',
                'Add the two terms.'),
      'quiz': 'Why does zero initial momentum not keep this particle fixed? What second '
              'condition would also have to be true for it to remain at rest?',
      'answer': 'Zero initial momentum describes only the first instant. Because the '
                'particle starts away from the mode, the posterior gradient is nonzero and '
                'changes its momentum. That new momentum changes its position. It would '
                'remain fixed only if momentum were zero and the posterior gradient at its '
                'starting position were also zero.',
      'takeaway': 'HMC alternates two causes: the posterior gradient changes momentum, and '
                  'momentum changes position. A particle can therefore start with zero '
                  'momentum, accelerate from rest, and pass through the mode after '
                  'potential energy becomes kinetic energy.',
      'math': 'For unit mass, dr/dt = ∇ log p(θ) and dθ/dt = r. The first relation says '
              'that the posterior slope changes momentum. The second says that momentum '
              'changes position. Also, U(θ) = -log p(θ), K(r) = rᵀr/2, and H = U + K.'},
     {'title': 'Lab 10 — The leapfrog integrator',
      'intuition': 'A full momentum update at only one end uses a one-sided view of a '
                   'force that changes as position moves. Leapfrog splits that update '
                   'across both ends. This makes the map symmetric and reversible. A final '
                   'accept-or-reject decision then corrects the small energy error that '
                   'remains.',
      'prediction': 'Which update should retrace its starting state more accurately when '
                    'simulated backward: half momentum–full position–half momentum, or '
                    'full momentum–full position? What should larger step sizes do to '
                    'average HMC acceptance?',
      'code': 'def leapfrog_step(q,p,step,grad_logp):\n'
              '    # TODO: half p, full q, half p\n'
              '    ...',
      'solution': 'def leapfrog_step(q,p,step,grad_logp):\n'
                  '    p=p+0.5*step*grad_logp(q)\n'
                  '    q=q+step*p\n'
                  '    p=p+0.5*step*grad_logp(q)\n'
                  '    return q,p',
      'hints': ('Start and end with momentum.',
                'Use half of the step for each momentum update.',
                'The position update uses the new half-step momentum.'),
      'quiz': 'Explain two separate safeguards in HMC: why the half-full-half order makes '
              'leapfrog reversible, and how the Metropolis correction uses ΔH to accept a '
              'proposal or repeat the current position.',
      'answer': 'The same half momentum update appears on both sides of the position '
                'update. When the step sign is reversed, these operations undo themselves '
                'in reverse order, so leapfrog can retrace the state. Finite steps still '
                'change Hamiltonian energy by ΔH. HMC accepts the proposed endpoint with '
                'probability min(1, exp(-ΔH)); otherwise it keeps the old position as the '
                'next chain state. This accept-or-reject rule removes the sampling bias '
                'from the remaining integration error.',
      'takeaway': 'The two half momentum updates are a symmetry device, not an arbitrary '
                  'split. They make the numerical map reversible and keep energy error '
                  'controlled. The Metropolis correction does not repair the path; it uses '
                  'its final energy error to accept the endpoint or keep the previous '
                  'state, which preserves the target distribution.',
      'math': 'Leapfrog applies K(ε/2) → D(ε) → K(ε/2), where K changes momentum and D '
              'changes position. Its inverse is the same sequence with -ε. For ΔH = H_new '
              '- H_old, the HMC acceptance probability is α = min(1, exp(-ΔH)). Draw u '
              'uniformly from 0 to 1; accept when u < α, otherwise repeat the old '
              'position.'},
     {'title': 'Lab 11 — From one HMC trajectory to posterior samples',
      'intuition': 'Leapfrog creates a deterministic proposal after position and momentum '
                   'are fixed. A full HMC transition surrounds that path with a momentum '
                   'draw and an accept-or-repeat decision. This lab tests what those outer '
                   'steps change when the process is repeated.',
      'prediction': 'Imagine two repeated procedures on the same target. Procedure A '
                    'starts once at (q₀, p₀) and then keeps applying deterministic '
                    'leapfrog updates without any new random draw. Procedure B starts each '
                    'transition by drawing fresh momentum, runs a fixed leapfrog '
                    'trajectory, draws a Uniform value for the Metropolis decision, and '
                    'stores either the proposal or the old position. Before seeing the '
                    'experiment, which procedure do you expect can represent the full '
                    'posterior over time? Explain what you think each random draw '
                    'contributes.',
      'code': 'def one_hmc_transition(position, log_target, integrate, rng):\n'
              '    # TODO: sample momentum, integrate, compare H, accept or repeat\n'
              '    ...',
      'solution': 'def one_hmc_transition(position, log_target, integrate, rng):\n'
                  '    old_position = np.asarray(position, dtype=float).copy()\n'
                  '    old_momentum = np.asarray(rng.normal(size=old_position.shape), '
                  'dtype=float)\n'
                  '    proposed_position, proposed_momentum = integrate(\n'
                  '        old_position.copy(), old_momentum.copy()\n'
                  '    )\n'
                  '    proposed_momentum = -np.asarray(proposed_momentum, dtype=float)\n'
                  '    old_h = -float(log_target(old_position)) + 0.5 * float(\n'
                  '        np.dot(old_momentum, old_momentum)\n'
                  '    )\n'
                  '    new_h = -float(log_target(proposed_position)) + 0.5 * float(\n'
                  '        np.dot(proposed_momentum, proposed_momentum)\n'
                  '    )\n'
                  '    log_accept = min(0.0, old_h - new_h)\n'
                  '    if np.log(rng.uniform()) < log_accept:\n'
                  '        return np.asarray(proposed_position, dtype=float), True\n'
                  '    return old_position, False',
      'hints': ('Draw momentum with `rng.normal(size=position.shape)` before calling '
                '`integrate`.',
                'Hamiltonian energy is `-log_target(q) + 0.5 * p·p`. Compare old and '
                'proposed energy.',
                'Use `log_accept = min(0, old_h - new_h)`. Accept when `log(rng.uniform()) '
                '< log_accept`; otherwise return the old position.'),
      'quiz': 'After position and momentum are fixed, a leapfrog trajectory is '
              'deterministic. Name the two random draws in one HMC transition, and explain '
              'exactly what the chain stores after a rejected proposal.',
      'answer': 'HMC first draws fresh momentum, which chooses a new direction and kinetic '
                'energy. After the deterministic trajectory, it draws a Uniform(0,1) value '
                'for the Metropolis decision. An accepted transition stores the proposed '
                'position. A rejected transition stores the old position again, so '
                'repeated positions are required chain states rather than missing data.',
      'takeaway': 'A trajectory is not a chain. Fresh momentum changes the energy orbit, '
                  'the Metropolis decision protects the target distribution, and repeated '
                  'stored positions form posterior samples only after many HMC '
                  'transitions.',
      'math': 'With unit mass, H(q,p) = -log_target(q) + pᵀp/2. Draw p₀ ~ Normal(0,I), '
              'compute the proposed state with L leapfrog steps and a final momentum flip, '
              'then use α=min(1, exp(H_old-H_new)). Store q* after acceptance and q again '
              'after rejection.'},
     {'title': 'Lab 12 — NUTS intuition',
      'intuition': 'A short fixed HMC path wastes possible movement. A long path can turn '
                   'back. NUTS grows a path until it starts to double back.',
      'prediction': 'When displacement and momentum point in opposite directions, is the '
                    'path still moving away from its start?',
      'code': 'def is_uturn(start, current, momentum):\n'
              '    # TODO: use a dot product\n'
              '    ...',
      'solution': 'def is_uturn(start,current,momentum):\n'
                  '    return '
                  'float(np.dot(np.asarray(current)-np.asarray(start),momentum)) < 0',
      'hints': ('Compute displacement from start.',
                'Take displacement · momentum.',
                'A negative value means that the angle is larger than 90°.'),
      'quiz': 'State one waste caused by an HMC path that is too short and one waste '
              'caused by a path that is too long.',
      'answer': 'A path that is too short stops before momentum can carry the particle '
                'far, so it spends gradient work on a small move. A path that is too long '
                'can turn back and repeat parts of the trajectory, so extra leapfrog steps '
                'add cost without reaching a new region. NUTS tries to stop after useful '
                'travel but before substantial retracing.',
      'takeaway': 'NUTS adapts trajectory length. Warmup also adapts step size and often a '
                  'mass matrix.',
      'math': 'A simple U-turn test is (θ-θ₀)·r<0. θ₀ is the start, θ is current, and r is '
              'momentum.'},
     {'title': 'Lab 13 — What convergence means for MCMC',
      'intuition': 'MCMC converges to a stationary sampling regime, not to one point. '
                   'Multiple chains test whether different starts reach the same regime.',
      'prediction': 'If four chains have stable means but occupy different modes, is '
                    'convergence good?',
      'code': 'def basic_rhat(chains):\n'
              '    chains=np.asarray(chains,dtype=float)\n'
              '    # TODO: compare between-chain and within-chain variance\n'
              '    ...',
      'solution': 'def basic_rhat(chains):\n'
                  '    m,n=chains.shape\n'
                  '    means=chains.mean(1); B=n*means.var(ddof=1); '
                  'W=chains.var(1,ddof=1).mean()\n'
                  '    var_hat=(n-1)/n*W+B/n\n'
                  '    return np.sqrt(var_hat/W)',
      'hints': ('Unpack `m, n = chains.shape`. Use axis 1 to summarize the draws inside '
                'each row.',
                'Compute `W` as the mean of the row sample variances. Compute `B` as `n` '
                'times the sample variance of the row means. Both variance operations use '
                '`ddof=1`.',
                'Combine them as `((n - 1) / n) * W + B / n`, then return the square root '
                'of the combined value divided by `W`.'),
      'quiz': 'Can R-hat near 1 prove that sampling is correct?',
      'answer': 'MCMC converges to a stationary sampling regime, not to one parameter '
                'value. In that regime, each chain targets the same posterior '
                'distribution: over time, the chains should visit the same regions with '
                'similar frequencies and produce compatible means, spreads, and other '
                'summaries. Their traces should no longer show a persistent trend or '
                'retain a strong effect from their starting points, although successive '
                'draws can still be autocorrelated. R-hat helps detect disagreement '
                'between chains, but a value near one is not proof that the model or '
                'sampling result is correct.',
      'takeaway': 'Use trace behavior, R-hat, ESS, Monte Carlo error, and divergences '
                  'together.',
      'math': 'W is the average sample variance inside chains. B is n times the sample '
              'variance across chain means. Basic R-hat compares a pooled variance '
              'estimate with W; disagreement between chain centers pushes the ratio above '
              'one.'},
     {'title': 'Lab 14 — Variational inference',
      'intuition': 'VI chooses a tractable distribution q that is close to the target. It '
                   'converts inference into optimization.',
      'prediction': 'For a right-skewed target, will one Gaussian capture both the peak '
                    'and the long tail?',
      'code': 'def mc_elbo(log_joint, mean, log_sd, eps):\n'
              '    # TODO: sample by reparameterization and average log p - log q\n'
              '    ...',
      'solution': 'def mc_elbo(log_joint,mean,log_sd,eps):\n'
                  '    sd=np.exp(log_sd); z=mean+sd*eps\n'
                  '    log_q=-0.5*eps**2-log_sd-0.5*np.log(2*np.pi)\n'
                  '    return np.mean(log_joint(z)-log_q)',
      'hints': ('Use fixed standard-Normal noise eps.',
                'Set z=mean+exp(log_sd)*eps.',
                'ELBO is the mean of log joint minus log q.'),
      'quiz': 'What object does VI return?',
      'answer': 'It returns parameters of an approximate distribution, not exact posterior '
                'samples.',
      'takeaway': 'VI is fast distribution fitting. Its approximation family decides which '
                  'shapes it can represent.',
      'math': 'ELBO=E_q[log p(D,θ)-log q(θ)]. θ is latent state and q is the chosen '
              'approximation.'},
     {'title': 'Lab 15 — How VI fails',
      'intuition': 'A mean-field Gaussian cannot represent skew, multiple modes, or curved '
                   'correlation. Optimization cannot repair a family that lacks the needed '
                   'shape.',
      'prediction': 'Which target will a diagonal Gaussian represent worst: round, '
                    'correlated, or banana-shaped?',
      'code': 'def diagonal_gaussian_sample(mean, log_sd, eps):\n'
              '    # TODO: reparameterized samples\n'
              '    ...',
      'solution': 'def diagonal_gaussian_sample(mean,log_sd,eps):\n'
                  '    return np.asarray(mean)+np.exp(log_sd)*np.asarray(eps)',
      'hints': ('Use one scale per coordinate.',
                'Exponentiate log_sd to keep scale positive.',
                'Multiply eps elementwise.'),
      'quiz': 'Can a full-covariance Gaussian represent a banana exactly?',
      'answer': 'A full-covariance Gaussian can represent linear correlation, but it '
                'cannot represent a curved dependence or several separated modes.',
      'takeaway': 'Always inspect approximation family against target geometry. A good '
                  'ELBO does not make the family more expressive.',
      'math': 'Mean-field means q(θ)=∏qᵢ(θᵢ). This removes posterior dependence between '
              'coordinates.'},
     {'title': 'Lab 16 — Expectation maximization',
      'intuition': 'If assignments were known, means would be easy to fit. If means were '
                   'known, assignments would be easy to estimate. EM alternates these '
                   'tasks.',
      'prediction': 'If both component means start together, can symmetric EM separate '
                    'them without a disturbance?',
      'code': 'def m_step(x, responsibilities):\n'
              '    # TODO: update component weights and means\n'
              '    ...',
      'solution': 'def m_step(x,responsibilities):\n'
                  '    counts=responsibilities.sum(0)\n'
                  '    weights=counts/len(x)\n'
                  '    means=(responsibilities*np.asarray(x)[:,None]).sum(0)/counts\n'
                  '    return weights,means',
      'hints': ('Soft counts are column sums.',
                'Weights are soft counts divided by sample count.',
                'Use responsibilities as weights for each mean.'),
      'quiz': 'Does EM preserve full uncertainty about parameters?',
      'answer': 'Standard EM returns point estimates. Responsibilities express soft '
                'assignments during optimization.',
      'takeaway': 'EM is coordinate optimization over latent assignment expectations and '
                  'model parameters.',
      'math': 'The E-step computes responsibilities rᵢk. The M-step maximizes expected '
              'complete-data log likelihood.'},
     {'title': 'Lab 17 — EM versus VI versus MCMC',
      'intuition': 'These methods learn different objects. A point, an approximate '
                   'distribution, and dependent posterior samples are not interchangeable '
                   'outputs.',
      'prediction': 'For a multimodal posterior where calibrated uncertainty matters, '
                    'which method is the safest first choice?',
      'code': 'def output_object(method):\n'
              '    # TODO: map MAP, EM, VI, and MCMC to their output object\n'
              '    ...',
      'solution': 'def output_object(method):\n'
                  "    return {'MAP':'point','EM':'point + "
                  "responsibilities','VI':'approximate distribution','MCMC':'posterior "
                  "samples'}[method]",
      'hints': ('MAP returns one optimum.',
                'VI returns q parameters.',
                'MCMC returns correlated draws.'),
      'quiz': 'Is NUTS a target distribution or a sampling algorithm?',
      'answer': 'It is an adaptive HMC sampling algorithm. The posterior remains the '
                'target.',
      'takeaway': 'Choose a method by the object you need, the geometry, and the failure '
                  'cost.',
      'math': 'MAP and EM optimize points; VI optimizes a distribution; MCMC constructs a '
              'chain whose stationary distribution is the posterior.'},
     {'title': 'Lab 18 — Hierarchical models and partial pooling',
      'intuition': 'Groups share population information. Low-data groups move more toward '
                   'the population mean than high-data groups.',
      'prediction': 'As population variance approaches zero, what happens to the estimate '
                    'for the 1/1 group?',
      'code': 'def shrink(raw, se, population_mean, population_sd):\n'
              '    # TODO: Normal-Normal shrinkage mean\n'
              '    ...',
      'solution': 'def shrink(raw,se,population_mean,population_sd):\n'
                  '    prior_var=population_sd**2; data_var=se**2\n'
                  '    weight=prior_var/(prior_var+data_var)\n'
                  '    return weight*raw+(1-weight)*population_mean',
      'hints': ('Combine a data estimate and population mean.',
                'The data weight is τ²/(τ²+SE²).',
                'Small population scale gives a small data weight.'),
      'quiz': 'Which group shrinks most: 70/100, 7/10, or 1/1?',
      'answer': 'The 1/1 group has the least information, so it shrinks most.',
      'takeaway': 'Partial pooling is adaptive regularization. Data-rich groups move less; '
                  'data-poor groups borrow more.',
      'math': 'θⱼ~Normal(μ,τ). μ is the population mean and τ is between-group scale.'},
     {'title': 'Lab 19 — Hierarchical coupon model',
      'intuition': 'Sensitivity and false-fire rates vary by rule but share population '
                   'distributions. Posterior samples can propagate this uncertainty into '
                   'transaction odds.',
      'prediction': 'Which ratio sⱼ/fⱼ will have the widest uncertainty: the rule with 20 '
                    'positive investigations or the rule with 2?',
      'code': 'def classify_variables():\n'
              '    # TODO: return observed, latent, and deterministic names\n'
              '    ...',
      'solution': 'def classify_variables():\n'
                  '    return '
                  "{'observed':['k_pos','k_neg'],'latent':['mu_s','sigma_s','eta','mu_f','sigma_f','xi'],'deterministic':['s','f','s_over_f']} ",
      'hints': ('Counts k are observed.',
                'Population terms and logits are latent.',
                'Rates and their ratio are deterministic transforms.'),
      'quiz': 'Why propagate samples into posterior odds instead of using mean rates?',
      'answer': 'The nonlinear ratio and odds update can be skewed. Sample propagation '
                'keeps uncertainty and dependence.',
      'takeaway': 'A hierarchical posterior gives rule-level rates, shrinkage, and '
                  'uncertainty for downstream decisions.',
      'math': 'sⱼ=logit⁻¹(ηⱼ) and fⱼ=logit⁻¹(ξⱼ). Binomial counts connect these rates to '
              'observed investigations.'},
     {'title': "Lab 20 — Neal's funnel",
      'intuition': 'When a scale becomes small, lower-level parameters must fit through a '
                   'narrow neck. This valid model creates hard sampling geometry.',
      'prediction': 'Where will a centered sampler have more difficulty: the wide mouth or '
                    'the narrow neck?',
      'code': 'def funnel_sample(v, z):\n'
              '    # TODO: centered x with scale exp(v/2)\n'
              '    ...',
      'solution': 'def funnel_sample(v,z):\n    return np.exp(v/2)*np.asarray(z)',
      'hints': ('Variance is exp(v).',
                'Standard deviation is exp(v/2).',
                'Multiply a standard-Normal z by that scale.'),
      'quiz': 'Does a divergence mean that the statistical model is invalid?',
      'answer': 'A divergence does not by itself prove that the statistical model is '
                'invalid. It often means that numerical integration cannot follow the '
                'posterior geometry with the current parameterization and settings.',
      'takeaway': 'Statistical validity and computational geometry are different '
                  'questions.',
      'math': 'v controls log variance. x|v~Normal(0,exp(v/2)); negative v creates the '
              'narrow neck.'},
     {'title': 'Lab 21 — Non-centered parameterization',
      'intuition': 'A non-centered model samples a standard variable z and constructs the '
                   'group parameter. This can straighten a funnel.',
      'prediction': 'When group data are weak, which form often samples better: centered '
                    'or non-centered?',
      'code': 'def noncenter(mu, sigma, z):\n    # TODO: deterministic transform\n    ...',
      'solution': 'def noncenter(mu,sigma,z):\n    return mu+sigma*np.asarray(z)',
      'hints': ('z has standard-Normal scale.',
                'Scale z by sigma.',
                'Then add the population mean.'),
      'quiz': 'Is the centered form always worse?',
      'answer': 'Non-centering is not always better. Strong group data can favor centered '
                'coordinates, so the useful parameterization depends on information '
                'strength and posterior geometry.',
      'takeaway': 'A reparameterization can keep the same model while giving the sampler '
                  'much easier coordinates.',
      'math': 'Centered: ηⱼ~Normal(μ,σ). Non-centered: zⱼ~Normal(0,1), ηⱼ=μ+σzⱼ.'},
     {'title': 'Lab 22 — Neural networks through the same lens',
      'intuition': 'Neural-network training usually finds one weight vector. Likelihood '
                   'gives the data loss, and a Gaussian prior gives L2 regularization '
                   'under MAP.',
      'prediction': 'If L2 strength increases, what happens to fitted weight magnitude?',
      'code': 'def map_loss(mse, weights, l2):\n'
              '    # TODO: data loss plus Gaussian-prior penalty\n'
              '    ...',
      'solution': 'def map_loss(mse,weights,l2):\n'
                  '    return float(mse)+0.5*l2*float(np.sum(np.asarray(weights)**2))',
      'hints': ('A zero-mean Gaussian prior penalizes squared weights.',
                'Use half × l2 × sum(w²).',
                'Add the penalty to data loss.'),
      'quiz': 'What uncertainty does ordinary SGD training preserve over weights?',
      'answer': 'It preserves none by itself. It returns a point, even if minibatch noise '
                'makes the path stochastic.',
      'takeaway': 'Cross-entropy is negative log likelihood; L2 can be a prior; SGD is '
                  'point optimization; VI fits an approximate weight posterior.',
      'math': 'w*=argmin L(w) is point training. Bayesian learning targets p(w|D), a '
              'distribution over weights.'},
     {'title': 'Lab 23 — Final synthesis challenge',
      'intuition': 'Method choice starts with the object you need. Then consider posterior '
                   'geometry, latent structure, compute budget, and failure cost.',
      'prediction': 'A latent mixture must serve calibrated rare-event decisions. Which '
                    'method would you test first, and what diagnostic matters?',
      'code': 'def choose_method(needs_uncertainty, latent_mixture, posterior_hard):\n'
              '    # TODO: return MAP, EM, VI, or NUTS\n'
              '    ...',
      'solution': 'def choose_method(needs_uncertainty,latent_mixture,posterior_hard):\n'
                  "    if needs_uncertainty and posterior_hard: return 'NUTS'\n"
                  "    if latent_mixture and not needs_uncertainty: return 'EM'\n"
                  "    if needs_uncertainty: return 'VI'\n"
                  "    return 'MAP'",
      'hints': ('First decide whether a point is enough.',
                'EM is natural for a point fit with latent assignments.',
                'Use NUTS when uncertainty matters and geometry is manageable at this '
                'scale.'),
      'quiz': 'What five questions should you answer before you choose an inference '
              'method?',
      'answer': 'Name the learned object, target, uncertainty behavior, gradient role, and '
                'likely failure mode.',
      'takeaway': 'You now have one map: model → target → algorithm → output → '
                  'diagnostics. Use geometry to predict behavior.',
      'math': 'A probabilistic model defines p(D,θ). Inference can optimize a point, '
              'optimize q(θ), or sample p(θ|D).'}]

    CONTROL_SPECS = {0: (),
     1: (('mean', 'slider', -3.0, 3.0, 0.1, 0.0),
         ('sd', 'slider', 0.2, 3.0, 0.1, 1.0),
         ('interval', 'slider', 0.2, 4.0, 0.1, 1.0)),
     2: (('prior mean', 'slider', 30.0, 90.0, 1.0, 55.0),
         ('prior width', 'slider', 2.0, 30.0, 1.0, 15.0),
         ('observations', 'slider', 1, 20, 1, 5),
         ('new sales day', 'slider', 30.0, 110.0, 1.0, 90.0)),
     3: (('start', 'slider', -5.0, 5.0, 0.2, -4.0),
         ('learning rate', 'slider', 0.01, 0.8, 0.01, 0.15),
         ('steps', 'slider', 1, 40, 1, 12)),
     4: (('wide sd', 'slider', 0.5, 4.0, 0.1, 2.5),),
     5: (('proposal sd', 'slider', 0.05, 8.0, 0.05, 1.0),
         ('draws', 'slider', 100, 5000, 100, 1500)),
     6: (('small proposal', 'slider', 0.02, 1.0, 0.02, 0.15),
         ('large proposal', 'slider', 0.5, 8.0, 0.1, 2.0)),
     7: (('correlation', 'slider', -0.98, 0.98, 0.02, 0.9),
         ('proposal sd', 'slider', 0.03, 1.5, 0.03, 0.35)),
     8: (('correlation', 'slider', -0.95, 0.95, 0.05, 0.8),),
     9: (('momentum x', 'slider', -3.0, 3.0, 0.1, 2.0),
         ('momentum y', 'slider', -3.0, 3.0, 0.1, 0.5),
         ('steps', 'slider', 1, 80, 1, 35)),
     10: (('step size', 'slider', 0.01, 1.2, 0.01, 0.2), ('steps', 'slider', 1, 80, 1, 25)),
     11: (('transitions', 'slider', 50, 800, 25, 300),
          ('leapfrog steps', 'slider', 1, 30, 1, 12)),
     12: (('step size', 'slider', 0.02, 0.7, 0.01, 0.18),
          ('max steps', 'slider', 5, 120, 1, 60),
          ('engine', 'dropdown', ('manual geometry', 'PyMC NUTS'), 'manual geometry')),
     13: (('broken separation', 'slider', 0.0, 6.0, 0.2, 0.0),),
     14: (('q mean', 'slider', -2.0, 5.0, 0.1, 0.5),
          ('q sd', 'slider', 0.1, 4.0, 0.1, 1.0)),
     15: (('target', 'dropdown', ('skewed', 'multimodal', 'banana'), 'banana'),),
     16: (('initial left mean', 'slider', -5.0, 1.0, 0.1, -1.0),
          ('initial right mean', 'slider', -1.0, 5.0, 0.1, 1.0),
          ('iterations', 'slider', 1, 25, 1, 8)),
     17: (('uncertainty required', 'dropdown', ('yes', 'no'), 'yes'),),
     18: (('population sd', 'slider', 0.01, 0.5, 0.01, 0.12),),
     19: (('prior odds', 'slider', 0.01, 1.0, 0.01, 0.1),
          ('rule', 'slider', 1, 3, 1, 3),
          ('engine',
           'dropdown',
           ('fast conjugate view', 'PyMC NUTS'),
           'fast conjugate view')),
     20: (('neck depth', 'slider', -9.0, -1.0, 0.2, -6.0),
          ('engine', 'dropdown', ('geometry view', 'PyMC NUTS'), 'geometry view')),
     21: (('population sd', 'slider', 0.05, 3.0, 0.05, 0.4),
          ('engine', 'dropdown', ('coordinate view', 'PyMC compare'), 'coordinate view')),
     22: (('L2 strength', 'slider', 0.0, 3.0, 0.05, 0.2),
          ('training steps', 'slider', 10, 300, 10, 120)),
     23: (('scenario',
           'dropdown',
           ('fast point forecast',
            'latent mixture',
            'calibrated small model',
            'large Bayesian model'),
           'calibrated small model'),)}
    return CONTROL_SPECS, LABS


@app.cell(hide_code=True)
def journey_guides():
    LAB_GUIDES = [{'act': 'Prologue · Build the decision system',
      'route': 'question → prediction → evidence → explanation',
      'story': 'You are building an inference engine for three automated coupon rules. '
               'Before you trust any result, you need a repeatable way to test your own '
               'intuition. This lab gives you that loop.',
      'mission': 'Encode the learning loop',
      'exercise': '**Why this code exists.** The five stages are the operating procedure '
                  'for every later lab.\n'
                  '\n'
                  '**Your task.** Complete `learning_loop()`.\n'
                  '\n'
                  '**Inputs.** None.\n'
                  '\n'
                  "**Return.** This exact list of strings: `['predict', 'simulate', "
                  "'inspect', 'code', 'explain']`.\n"
                  '\n'
                  '**Suggested steps.** Create one list. Put the five stage names in the '
                  'shown order. Return the list.\n'
                  '\n'
                  '**Checkpoint.** The test calls the function and compares the complete '
                  'list, including order and spelling.',
      'prediction': 'Before you see a result, which stage do you expect will most often '
                    'change your first idea: simulation, visual inspection, or '
                    'implementation? State one reason.',
      'quiz': 'Why must the prediction be recorded before the simulation result appears? '
              'Explain what would be lost if you predicted afterward.',
      'chapter_opening': 'You have taken over a small neighborhood bakery. Every evening '
                         "you must choose tomorrow's production, but demand, customer "
                         'behavior, and promotion quality are uncertain. During this '
                         'course you will build the inference system that supports those '
                         'decisions.',
      'bridge': 'You do not need advanced probability before you start. You need to state '
                'what you expect, inspect evidence, and revise your explanation.',
      'scenario': 'The bakery loses money in two ways: unsold bread becomes waste, and '
                  'missing stock turns customers away. A useful model must therefore '
                  'report uncertainty, not only a forecast.',
      'goals': ['Use the five-stage learning loop.',
                'Separate a model, an inference method, and a decision.',
                'Keep a written record of predictions and explanations.'],
      'model': 'The recurring loop is **predict → simulate → inspect → code → explain**. A '
               'prediction records your current model. A simulation produces evidence. '
               'Inspection finds the mismatch. Code exposes the mechanism. Explanation '
               'checks whether the new model transfers beyond one plot.',
      'experiment': 'The first experiment is a map of the course workflow. It has no '
                    'stochastic model yet. Read each stage as a question that you will '
                    'answer in every later lab.',
      'exercise_intro': 'The function is small on purpose. It verifies that you know the '
                        'order in which evidence enters the learning process.',
      'next': 'Next, the bakery faces its first unknown quantity: the number of loaves '
              'customers will want tomorrow.',
      'prediction_answer': 'There is no universal winning stage. Simulation often exposes '
                           'a wrong expectation, visual inspection shows where it failed, '
                           'and implementation reveals assumptions that prose hid. The '
                           'important step is to record the prediction first, so the '
                           'mismatch is visible.'},
     {'act': 'Chapter 1 · One uncertain demand number',
      'route': 'density → mass → likelihood → posterior',
      'story': 'The coupon engine will later compare many possible rule qualities. First, '
               'you need to read a probability curve correctly. A high point is not the '
               'same thing as a large amount of probability.',
      'mission': 'Implement the Normal log density',
      'exercise': '**Why this code exists.** Likelihoods, priors, posteriors, MAP, MCMC, '
                  'HMC, and VI all need a function that scores a possible value. This is '
                  'the first scoring function.\n'
                  '\n'
                  '**Your task.** Complete `normal_log_density(x, mean, sd)` with NumPy '
                  'only. Do not use `scipy.stats`.\n'
                  '\n'
                  '**Inputs.** `x` is one number or a NumPy array of values to score. '
                  '`mean` is the center of the Normal distribution. `sd` is its positive '
                  'standard deviation.\n'
                  '\n'
                  '**Return.** The natural logarithm of the Normal density at every value '
                  'in `x`. Keep the same scalar or array behavior that NumPy arithmetic '
                  'gives you.\n'
                  '\n'
                  '**Suggested steps.** Compute the constant term `-0.5 * log(2π sd²)`. '
                  'Compute the squared standardized distance `(x - mean)² / sd²`. Subtract '
                  'half of that distance.\n'
                  '\n'
                  '**Example.** For `x=[0, 1]`, `mean=0`, and `sd=1`, the result is '
                  'approximately `[-0.9189, -1.4189]`. The value farther from the mean has '
                  'a lower log-density.\n'
                  '\n'
                  '**Checkpoint.** The test compares both values with the known '
                  'standard-Normal result.',
      'prediction': 'The total area under a Normal density is always 1. If demand SD '
                    'becomes three times larger while the shaded interval width stays '
                    'fixed, what happens to peak height and shaded probability mass? '
                    'Explain both effects.',
      'quiz': 'In this demand plot, what does density height measure, and what does the '
              'shaded probability mass measure? Explain why a high point on the curve is '
              'not itself a high probability.',
      'chapter_opening': "Chapter 1 starts with one unknown quantity: tomorrow's bread "
                         'demand. You will learn to read probability distributions, update '
                         'average demand from data, optimize one estimate, and see what a '
                         'point estimate hides.',
      'bridge': 'The course loop is now in place. We can use it on a probability model '
                'whose complete state fits in one plot.',
      'scenario': 'Let X be the number of sourdough loaves customers request tomorrow. The '
                  'mean mu controls the center of demand. The standard deviation sigma '
                  'controls ordinary day-to-day variation.',
      'goals': ['Distinguish density height from probability mass.',
                'Read the effects of mean and standard deviation.',
                'Connect interval probability to a stock decision.'],
      'model': '$$X \\sim \\operatorname{Normal}(\\mu,\\sigma)$$\n'
               '\n'
               '$$P(a \\le X \\le b)=\\int_a^b p(x)\\,dx$$\n'
               '\n'
               "X is tomorrow's demand. The parameter mu moves the center. A larger sigma "
               'spreads the same total probability over a wider range, so the peak becomes '
               'lower. The integral is area, and that area is probability mass.',
      'experiment': 'The curve shows possible sales totals. The shaded region is the '
                    'chance that demand stays within the selected distance of the mean. '
                    'Change demand SD first and keep the interval fixed. Then change the '
                    'interval and keep demand SD fixed.',
      'exercise_intro': 'This function becomes the basic score used by the likelihood, '
                        'posterior, optimizer, and samplers in later labs.',
      'next': 'A demand distribution is useful only after its parameters have meaning. '
              'Next, daily sales records update the unknown average demand mu.',
      'prediction_answer': 'When demand SD becomes three times larger, the Normal curve '
                           'spreads the same total area over a range that is three times '
                           'wider. Its peak becomes one third as high. The total area '
                           'remains 1, but a fixed-width interval around the mean contains '
                           'less probability mass.'},
     {'act': 'Chapter 1 · Let sales records update demand',
      'route': 'density → mass → likelihood → posterior',
      'story': 'You can now score one value under one distribution. Next, observations '
               'reshape the landscape of plausible parameter values. The same Normal score '
               'becomes a likelihood when `mu` is the value that can change.',
      'mission': 'Implement the log likelihood and log posterior',
      'exercise': '**Why this code exists.** The optimizer and samplers in later labs need '
                  'one function that scores a candidate average demand against all '
                  'observed days. Reuse the `normal_log_density` function that you '
                  'implemented in Lab 1; do not use `scipy.stats`.\n'
                  '\n'
                  '**Your task.** Implement `log_likelihood(mu, x, sigma)` by evaluating '
                  '`normal_log_density(x, mu, sigma)` and summing the observation scores. '
                  'Then implement `log_posterior(mu, x, sigma, prior_mu, prior_sigma)` as '
                  'that likelihood plus `normal_log_density(mu, prior_mu, prior_sigma)`.\n'
                  '\n'
                  '**Arguments.** `mu` is one candidate average demand. `x` is a NumPy '
                  'array of observed sales. `sigma` is known day-to-day noise. `prior_mu` '
                  'and `prior_sigma` define the Normal prior for the candidate `mu`.\n'
                  '\n'
                  '**Return.** Each function must return one finite scalar log score.\n'
                  '\n'
                  '**Checkpoint.** The test checks the likelihood and prior terms '
                  'separately, so scoring the observations twice will fail. Do not score '
                  'the observations a second time.',
      'prediction': 'Set **new sales day** to 90 loaves, which is far above the current '
                    'estimate. Will this one day move the posterior more when **prior '
                    'width** is small or large? Explain what prior width says about the '
                    "bakery owner's confidence before current sales data.",
      'quiz': 'With the sales records fixed and candidate values of mu changing, state the '
              'role of the prior, likelihood, and posterior.',
      'chapter_opening': '',
      'bridge': 'Lab 1 treated mu and sigma as known controls. The bakery does not '
                'actually know average demand, so mu now becomes the unknown quantity on '
                'the horizontal axis.',
      'scenario': 'You have a prior estimate of average daily demand from the previous '
                  'owner. New daily sales records arrive one at a time. We assume the '
                  'day-to-day scale sigma is known for now so that you can see the update '
                  'clearly.',
      'goals': ['Read likelihood as a function of a parameter.',
                'Combine prior and likelihood in log space.',
                'Predict how data quantity and prior width control the posterior.'],
      'model': '$$x_i \\mid \\mu \\sim \\operatorname{Normal}(\\mu,\\sigma), \\qquad\n'
               '\\mu \\sim \\operatorname{Normal}(\\mu_0,\\tau)$$\n'
               '\n'
               '$$p(\\mu\\mid x)\\propto p(\\mu)\\prod_{i=1}^{n}p(x_i\\mid\\mu)$$\n'
               '\n'
               'The value x_i is observed demand on day i. The values mu_0 and tau are the '
               'prior center and width. Each candidate mu receives one likelihood score '
               'from the fixed data. More observations usually make the likelihood '
               'narrower.',
      'experiment': 'The left panel builds the current posterior from the prior and the '
                    'selected number of existing sales days. The right panel adds one '
                    'highlighted sales day and overlays the posterior before and after it. '
                    'Keep the existing-day count and the new sales value fixed. First use '
                    'a narrow prior width, then a wide prior width, and compare the '
                    'reported posterior shift. You can also change the new day itself to '
                    'see why an ordinary day has less effect than an unusually high day.',
      'exercise_intro': 'You will add independent log scores instead of multiplying many '
                        'small densities. This is the same numerical pattern used in '
                        'production probabilistic models.',
      'next': 'The posterior now contains a complete range of plausible average demand '
              'values. Next, the ordering system asks for one operating number.',
      'prediction_answer': 'The unexpected 90-loaf day moves the posterior more when the '
                           'prior is wide. A wide prior places substantial weight on many '
                           'possible demand means, so it expresses less confidence in its '
                           'center. A narrow prior concentrates belief near its center and '
                           'resists one unusual day. The prior stays fixed during the '
                           'update; the posterior is the object that moves.'},
     {'act': 'Chapter 1 · Turn a distribution into one operating value',
      'route': 'posterior landscape → slope → one best point',
      'story': 'The posterior is a complete landscape, but some systems need one operating '
               'value. You will now climb the same landscape and watch how the step size '
               'controls the path.',
      'mission': 'Implement finite-difference gradient ascent',
      'exercise': '**Why this code exists.** This is the small version of the update loop '
                  'used in many machine-learning optimizers.\n'
                  '\n'
                  '**Your task.** Complete `gradient_ascent(fn, start, rate, steps)`.\n'
                  '\n'
                  '**Inputs.** `fn(x)` returns the scalar objective to maximize. `start` '
                  'is the first position. `rate` is the learning rate. `steps` is the '
                  'number of updates.\n'
                  '\n'
                  '**Return.** A NumPy array with the start and every later position. Its '
                  'length must be `steps + 1`.\n'
                  '\n'
                  '**Suggested steps.** At each position, estimate the derivative with a '
                  'centered finite difference, such as `(fn(x+eps)-fn(x-eps))/(2*eps)`. '
                  'Update `x = x + rate * gradient`. Append the new value.\n'
                  '\n'
                  '**Checkpoint.** On the objective `-(x-2)²`, the path must move from '
                  '`-2` to within `0.05` of `2` after 80 updates.',
      'prediction': 'If the learning rate becomes much larger, will the path approach the '
                    'mode smoothly, oscillate across it, or behave like the smaller rate? '
                    'Explain the roles of local slope and step size.',
      'quiz': 'What score does MLE maximize? What extra information enters the MAP score? '
              'State what both methods return.',
      'chapter_opening': '',
      'bridge': 'The posterior from Lab 2 preserves uncertainty. A production plan can '
                'still require one number, so we must be precise about which point we '
                'choose and how we search for it.',
      'scenario': 'The overnight planning job needs one estimate of average demand. MLE '
                  'uses only the current sales records. MAP also uses the prior experience '
                  'encoded in Lab 2.',
      'goals': ['Separate MLE from MAP.',
                'Separate an optimization objective from an optimizer.',
                'Diagnose learning rates from the optimization path.'],
      'model': '$$\\hat\\mu_{MLE}=\\arg\\max_\\mu \\log p(x\\mid\\mu)$$\n'
               '\n'
               '$$\\hat\\mu_{MAP}=\\arg\\max_\\mu [\\log p(x\\mid\\mu)+\\log p(\\mu)]$$\n'
               '\n'
               '$$\\mu_{t+1}=\\mu_t+\\eta\\,\\frac{d}{d\\mu}\\log p(\\mu\\mid x)$$\n'
               '\n'
               'The value eta is the learning rate. A small value gives slow progress. A '
               'very large value can cross the mode repeatedly or diverge.',
      'experiment': 'Every marker is one optimizer state on the same demand posterior. '
                    'Change the start first. Then hold the start fixed and compare a '
                    'small, useful, and excessive learning rate.',
      'exercise_intro': 'You will estimate the derivative numerically once before using '
                        'analytic gradients or automatic differentiation. This makes the '
                        "optimizer's information requirement explicit.",
      'next': 'The planner now has one estimate. Before using it for stock, you must ask '
              'what information disappeared when the posterior became one point.',
      'prediction_answer': 'A larger learning rate multiplies every local slope by a '
                           'larger step. A moderate increase can reach the mode faster, '
                           'but a large rate crosses the mode, then crosses back, and can '
                           'oscillate or diverge. The gradient gives direction and local '
                           'steepness; the rate controls travel distance.'},
     {'act': 'Chapter 1 · Recover uncertainty hidden by a point',
      'route': 'one best point → local curvature → uncertainty width',
      'story': 'A point estimate can hide risk. Two rules can have the same MAP value '
               'while one has much less evidence. Local curvature gives a first, limited '
               'estimate of the missing width.',
      'mission': 'Estimate Laplace width from local curvature',
      'exercise': '**Why this code exists.** The Laplace approximation turns local '
                  'curvature at the MAP into a local Gaussian uncertainty estimate.\n'
                  '\n'
                  '**Your task.** Complete `laplace_sd(log_posterior, map_value, eps)`.\n'
                  '\n'
                  '**Inputs.** `log_posterior(x)` returns a scalar log score. `map_value` '
                  'is the mode. `eps` is a small finite-difference distance.\n'
                  '\n'
                  '**Return.** One positive approximate posterior standard deviation.\n'
                  '\n'
                  '**Suggested steps.** Estimate the second derivative with '
                  '`(f(x+eps)-2f(x)+f(x-eps))/eps²`. At a mode this curvature is negative. '
                  'Convert it with `sqrt(-1 / curvature)`.\n'
                  '\n'
                  '**Checkpoint.** For a Normal log-posterior with true standard deviation '
                  '`2`, the estimate must be close to `2`.',
      'prediction': 'If two posteriors have the same MAP but one has ten times the '
                    'variance, what does MAP report for each? Which stock-decision '
                    'information is missing?',
      'quiz': 'Why can two branches with the same MAP demand require different safety '
              'stock? Connect posterior width to stockout or waste probability.',
      'chapter_opening': '',
      'bridge': 'Lab 3 found a mode. A mode does not tell the bakery how likely a stockout '
                'is or how far demand can move from the estimate.',
      'scenario': 'Two bakery branches both have MAP demand of 60 loaves. One has stable '
                  'weekday traffic. The other has large event-driven changes. The same '
                  'best estimate supports different production risks.',
      'goals': ['See why equal MAP values can imply different decisions.',
                'Read credible width as uncertainty.',
                'Understand the local scope of a Laplace approximation.'],
      'model': 'Near a mode, a second-order approximation gives\n'
               '\n'
               '$$\\log p(\\mu\\mid x)\\approx C-\\frac{(\\mu-\\hat\\mu_{MAP})^2}{2s^2}$$\n'
               '\n'
               'with\n'
               '\n'
               '$$s^2\\approx-\\left[\\frac{d^2}{d\\mu^2}\\log p(\\mu\\mid '
               'x)\\right]^{-1}_{\\mu=\\hat\\mu_{MAP}}.$$\n'
               '\n'
               'More negative curvature means a narrower local approximation. This local '
               'fit can miss skew, tails, and other modes.',
      'experiment': "Both curves keep the same mode. Change only the wide posterior's "
                    'standard deviation. Compare the 95 percent intervals and imagine '
                    'ordering the MAP number of loaves in both cases.',
      'exercise_intro': 'The code converts local curvature into a standard deviation. It '
                        'is useful, but the plot also shows why local information cannot '
                        'describe every posterior.',
      'next': 'The need for global uncertainty leads to sampling. Chapter 2 makes a chain '
              'spend time across the full demand posterior.',
      'prediction_answer': 'MAP reports the same maximizing value for both posteriors. It '
                           'does not report that one posterior is much wider. The missing '
                           'information includes credible intervals and the tail '
                           'probabilities that determine stockout and waste risk.'},
     {'act': 'Chapter 2 · Sample the demand posterior',
      'route': 'point estimate → random walk → posterior samples',
      'story': 'Local width is not enough when a posterior is skewed or has several '
               'regions. You now build a chain that explores the full target and keeps '
               'repeated states when proposals fail.',
      'mission': 'Implement Metropolis-Hastings from scratch',
      'exercise': '**Why this code exists.** This sampler is the smallest complete example '
                  'of posterior sampling by accept-or-reject moves.\n'
                  '\n'
                  '**Your task.** Complete `metropolis(log_target, start, proposal_sd, '
                  'draws, rng)`.\n'
                  '\n'
                  '**Inputs.** `log_target(x)` is an unnormalized log-density. `start` is '
                  'the first state. `proposal_sd` controls random-walk step size. `draws` '
                  'is the number of stored states. `rng` is a NumPy random generator.\n'
                  '\n'
                  '**Return.** `(samples, acceptance_rate)`. `samples` must have `draws` '
                  'entries, including repeats after rejection.\n'
                  '\n'
                  '**Suggested steps.** Propose `current + Normal(0, proposal_sd)`. '
                  'Compute `log_alpha = proposed_logp - current_logp`. Accept when `log(U) '
                  '< min(0, log_alpha)`. Always append the current state after the '
                  'decision.\n'
                  '\n'
                  '**Checkpoint.** On a standard Normal target, 5,000 draws must have a '
                  'mean near zero and a plausible acceptance rate.',
      'prediction': 'If proposal standard deviation becomes ten times larger, what happens '
                    'to acceptance rate? What happens to the distance of a move when it is '
                    'accepted? Give both answers.',
      'quiz': 'Why must a rejected proposal appear as a repeated demand value in the saved '
              'chain?',
      'chapter_opening': 'Chapter 2 replaces one optimized demand estimate with '
                         'representative draws from the posterior. You will first build a '
                         'random-walk sampler, then learn why geometry and dependence '
                         'control its quality.',
      'bridge': 'The Laplace approximation summarized local width around one mode. We now '
                'want a mechanism that can represent the full target without assuming that '
                'it is Gaussian.',
      'scenario': 'The state of the chain is one possible value of average daily demand. A '
                  'proposal asks whether a nearby value should become the next state.',
      'goals': ['Implement a Metropolis transition.',
                'Interpret accepted and rejected moves.',
                'Tune proposal scale by movement and acceptance together.'],
      'model': 'For a symmetric proposal,\n'
               '\n'
               "$$\\alpha=\\min\\left(1,\\frac{p(\\mu'\\mid x)}{p(\\mu\\mid "
               'x)}\\right).$$\n'
               '\n'
               'Draw u uniformly between 0 and 1. Accept when log u is smaller than the '
               'proposed minus current log target. The rule accepts all uphill moves and '
               'some downhill moves.',
      'experiment': 'The left plot compares posterior shape and chain occupancy. The trace '
                    'shows order in time. Consecutive-state scatter shows how far the '
                    'chain moves. Try a very small proposal, then a very large one.',
      'exercise_intro': 'This short function contains the complete random-walk mechanism. '
                        'Do not optimize the target; preserve repeated states and '
                        'probabilistic downhill moves.',
      'next': 'A histogram can look reasonable even when the chain moves badly. Next, you '
              'measure how much independent information the rows contain.',
      'prediction_answer': 'A proposal scale that is ten times larger usually lowers '
                           'acceptance because more proposals land in low-density regions. '
                           'Proposed distances become larger. Accepted moves can also '
                           'travel farther, but the accepted set is selective, and very '
                           'large proposals make acceptance rare.'},
     {'act': 'Chapter 2 · Measure dependence in the chain',
      'route': 'posterior samples → dependence → effective information',
      'story': 'A long chain can still contain little information when each state '
               'resembles the previous state. You need a way to measure this stickiness '
               'before you trust sample averages.',
      'mission': 'Implement autocorrelation at one lag',
      'exercise': '**Why this code exists.** Autocorrelation is the basic signal behind '
                  'effective sample size.\n'
                  '\n'
                  '**Your task.** Complete `autocorrelation(x, lag)`.\n'
                  '\n'
                  '**Inputs.** `x` is a one-dimensional sequence. `lag` is a nonnegative '
                  'integer shift.\n'
                  '\n'
                  '**Return.** One normalized correlation value. Lag zero must return `1` '
                  'for a nonconstant sequence.\n'
                  '\n'
                  '**Suggested steps.** Center `x`. Use the dot product between the '
                  'overlapping parts `x[:-lag]` and `x[lag:]`. Divide by the full centered '
                  'sum of squares. Handle lag zero without empty slices.\n'
                  '\n'
                  '**Checkpoint.** An increasing sequence must have autocorrelation `1` at '
                  'lag zero and a positive value above `0.4` at lag one.',
      'prediction': 'Which can contain more information: 5,000 sticky draws or 1,000 '
                    'nearly independent draws? Name the diagnostic needed to decide.',
      'quiz': 'What dependence does ESS correct for, and why can 5,000 saved draws provide '
              'far fewer than 5,000 independent pieces of information?',
      'chapter_opening': '',
      'bridge': 'Lab 5 showed acceptance and movement. Neither raw draw count nor '
                'acceptance rate alone tells you how much information the chain contains.',
      'scenario': 'The bakery stores 5,000 posterior draws. If each draw is almost '
                  'determined by the previous one, the file is large but the estimate of '
                  'stockout risk can still be noisy.',
      'goals': ['Read trace and autocorrelation together.',
                'Implement lag autocorrelation.',
                'Use ESS as an information count.'],
      'model': 'For centered draws y_t = x_t - mean(x),\n'
               '\n'
               '$$\\rho_k\\approx\\frac{\\sum_t y_t y_{t+k}}{\\sum_t y_t^2}.$$\n'
               '\n'
               'A rough effective sample size is\n'
               '\n'
               '$$ESS\\approx\\frac{N}{1+2\\sum_{k\\ge1}\\rho_k}.$$\n'
               '\n'
               'Positive lag correlation increases the denominator and reduces effective '
               'information.',
      'experiment': 'Both chains contain 5,000 draws from the same demand target. Compare '
                    'traces, acceptance, autocorrelation decay, and ESS. Look for a '
                    'proposal that accepts often but still moves slowly.',
      'exercise_intro': 'This simple calculation makes the repeated-information penalty '
                        'visible before a library reports ESS for you.',
      'next': 'One dimension hides an important problem. Next, baseline demand and price '
              'response create a narrow two-parameter posterior.',
      'prediction_answer': 'One thousand nearly independent draws can contain more '
                           'information than 5,000 sticky draws. The raw row count cannot '
                           'decide this. Inspect autocorrelation and effective sample '
                           'size, which discount repeated information in nearby states.'},
     {'act': 'Chapter 2 · See two-parameter posterior geometry',
      'route': 'one dimension → correlated geometry → narrow valleys',
      'story': 'Real posteriors couple parameters. For example, sensitivity and prevalence '
               'can trade off. A two-dimensional correlated Normal lets you see why a '
               'random walk wastes moves across a narrow valley.',
      'mission': 'Build the covariance matrix',
      'exercise': '**Why this code exists.** This matrix controls the shape of the '
                  'two-dimensional target used by Metropolis, gradients, and HMC.\n'
                  '\n'
                  '**Your task.** Complete `covariance(rho)`.\n'
                  '\n'
                  '**Inputs.** `rho` is a correlation value between `-1` and `1`.\n'
                  '\n'
                  '**Return.** The NumPy-compatible 2×2 matrix `[[1, rho], [rho, 1]]`.\n'
                  '\n'
                  '**Suggested steps.** Put unit variances on the diagonal and `rho` in '
                  'both off-diagonal positions.\n'
                  '\n'
                  '**Checkpoint.** With `rho=0.7`, the matrix must match the expected '
                  'values and have positive eigenvalues.',
      'prediction': 'Which target is harder for an isotropic random walk: rho = 0 or rho = '
                    '0.99? Identify the direction in which proposed parameter combinations '
                    'leave the plausible valley.',
      'quiz': 'Why can one isotropic proposal scale not move efficiently both along and '
              'across a narrow correlated demand valley?',
      'chapter_opening': '',
      'bridge': 'The one-dimensional demand posterior had only one direction. A real '
                'demand model has parameters that can compensate for each other.',
      'scenario': 'Suppose the bakery estimates centered baseline demand beta_0 and '
                  'centered price response beta_price. A higher baseline combined with a '
                  'stronger negative price effect can explain similar sales, so '
                  'uncertainty forms a diagonal valley.',
      'goals': ['Read contours as equal-density sets.',
                'Connect correlation to parameter trade-offs.',
                'Predict random-walk failure from geometry.'],
      'model': 'We use a standardized bivariate Normal target:\n'
               '\n'
               '$$\\theta=(\\beta_0,\\beta_{price}),\\qquad\n'
               '\\Sigma=\\begin{bmatrix}1&\\rho\\\\\\rho&1\\end{bmatrix}.$$\n'
               '\n'
               'The value rho controls orientation and narrowness. The coordinates are '
               'centered at their posterior estimates, so zero is the mode.',
      'experiment': 'The left plot shows the first 400 random-walk states. The right plot '
                    'shows accumulated draws. Increase absolute correlation while keeping '
                    'the isotropic proposal scale fixed.',
      'exercise_intro': 'The matrix is small, but it rotates and stretches the posterior '
                        'geometry used throughout the HMC chapter.',
      'next': 'A random walk cannot see the direction of higher posterior density. Next, '
              'you draw that local direction at every point.',
      'prediction_answer': 'The target with rho = 0.99 is harder for an isotropic random '
                           'walk. Its probability lies in a long, narrow diagonal valley. '
                           'Proposals that move perpendicular to that diagonal quickly '
                           'cross a narrow wall and leave the plausible region.'},
     {'act': 'Chapter 2 · Turn local slope into a field',
      'route': 'geometry → local slope → algorithm input',
      'story': 'The narrow valley shows that blind proposals waste effort. A gradient '
               'gives local directional information. You will compute it without automatic '
               'differentiation, then separate that information from the goal of the '
               'algorithm that uses it.',
      'mission': 'Compute a vector gradient by finite differences',
      'exercise': '**Why this code exists.** The same vector gradient will later drive '
                  'optimization, leapfrog motion, HMC, VI, and neural-network training.\n'
                  '\n'
                  '**Your task.** Complete `numerical_gradient(fn, point, eps)`.\n'
                  '\n'
                  '**Inputs.** `fn(point)` returns one scalar. `point` is a '
                  'one-dimensional NumPy vector. `eps` is the finite-difference step.\n'
                  '\n'
                  '**Return.** A gradient vector with one derivative for each coordinate.\n'
                  '\n'
                  '**Suggested steps.** For each coordinate, create a zero step vector. '
                  'Set one entry to `eps`. Evaluate the centered difference and store it '
                  'in the matching gradient entry.\n'
                  '\n'
                  '**Checkpoint.** For `-sum((x-1)²)` at `[0,2]`, the gradient must be '
                  'close to `[2,-2]`.',
      'prediction': 'At a point away from the centered Gaussian mode, does the log-density '
                    'gradient point generally toward or away from the center? Explain why '
                    'this direction increases the score.',
      'quiz': 'The gradient points uphill. Why does following it directly lead to one '
              'optimized point, while posterior sampling must also represent regions away '
              'from that point? Explain the different goals without using concepts from '
              'later labs.',
      'chapter_opening': '',
      'bridge': 'The correlated contours show where density is high, but the random walk '
                'does not use their slope. Gradients provide this local geometric '
                'information.',
      'scenario': 'At any proposed baseline-and-price pair, the bakery model can evaluate '
                  'how the log posterior changes if either parameter moves slightly.',
      'goals': ['Interpret every gradient component.',
                'Compute a numerical gradient.',
                'Separate a gradient from an algorithm that uses it.'],
      'model': '$$\\nabla\\log p(\\theta)=\n'
               '\\left(\\frac{\\partial\\log p}{\\partial\\beta_0},\n'
               '\\frac{\\partial\\log p}{\\partial\\beta_{price}}\\right).$$\n'
               '\n'
               'For coordinate j, a centered finite difference is\n'
               '\n'
               '$$\\frac{f(\\theta+\\epsilon e_j)-f(\\theta-\\epsilon '
               'e_j)}{2\\epsilon}.$$\n'
               '\n'
               'The gradient points toward the fastest local increase in log posterior.',
      'experiment': 'The contour map is the same target as Lab 7. Arrows are normalized so '
                    'that you can inspect direction without long arrows hiding short ones. '
                    'Follow several arrows toward the mode by eye.',
      'exercise_intro': 'The implementation changes one coordinate at a time. Torch '
                        'autograd later computes the same object for a neural network.',
      'next': 'Chapter 3 keeps the same price-demand landscape but adds momentum, which '
              'turns local slope into long posterior travel.',
      'prediction_answer': 'Away from the mode of a centered Gaussian, the log-density '
                           'gradient points generally toward the center. Moving that way '
                           'reduces the distance measured by the posterior geometry, so '
                           'the log score increases. The arrows become shorter near the '
                           'mode because the slope approaches zero.'},
     {'act': 'Chapter 3 · Build motion from two state updates',
      'route': 'posterior gradient → changed momentum → changed position → energy exchange',
      'story': 'Lab 8 gave every point on the posterior landscape a gradient arrow. This '
               'lab turns that static field into motion. HMC does not move position '
               'directly along the gradient. First the gradient changes a temporary '
               'momentum vector. Then that momentum changes the parameter position. '
               'Keeping these two updates separate explains both motion from rest and '
               'motion through the mode.',
      'mission': 'Compute Hamiltonian energy',
      'exercise': '**Why this code exists.** The experiment tracks potential energy from '
                  'position and kinetic energy from momentum. HMC uses their sum to check '
                  'whether its simulated motion is a plausible proposal.\n'
                  '\n'
                  '**Your task.** Complete `energy(log_target, position, momentum)`.\n'
                  '\n'
                  '**Inputs.** `log_target(position)` returns one target log-density. '
                  '`position` is the parameter vector θ. `momentum` is the temporary '
                  'vector r with the same shape.\n'
                  '\n'
                  '**Return.** One scalar `U + K`, where `U = -log_target(position)` and '
                  '`K = 0.5 * momentum·momentum`.\n'
                  '\n'
                  '**Suggested steps.** Compute potential energy. Compute the momentum dot '
                  'product. Multiply that dot product by one half. Add the two energies.\n'
                  '\n'
                  '**Checkpoint.** For the supplied position and momentum, total energy '
                  'must equal `2.5`.',
      'prediction': 'Set **momentum x** and **momentum y** to zero (`0`). The particle '
                    'starts away from the mode. Will it stay still, move toward the mode '
                    'and stop, or move toward and pass through? State which of the two HMC '
                    'updates creates its first movement.',
      'quiz': 'Why does zero initial momentum not keep this particle fixed? What second '
              'condition would also have to be true for it to remain at rest?',
      'chapter_opening': 'Chapter 3 builds HMC from a two-state physical simulation to a '
                         'production NUTS fit. The target stays the same correlated '
                         'baseline-and-price posterior, so you can compare the path with '
                         'the random walk and gradient field from the prior labs.',
      'bridge': 'Lab 8 showed the posterior gradient as a force field, but a field alone '
                'does not yet define a trajectory. This lab adds the temporary state that '
                'the field changes.',
      'scenario': 'The two coordinates θ₁ and θ₂ are the standardized bakery '
                  'baseline-demand and price-response parameters. HMC treats this '
                  'parameter pair as a particle position. The particle is only a '
                  'computational device; it is not a physical claim about sales.',
      'goals': ['Read the two components of one momentum vector.',
                'Follow the gradient-to-momentum-to-position update sequence.',
                'Explain motion from rest and motion through the mode by energy exchange.'],
      'model': 'The position is θ = (θ₁, θ₂). The temporary momentum is r = (r₁, r₂). The '
               'controls named **momentum x** and **momentum y** set only the initial '
               'values r₁ and r₂. A positive component points along the positive plot '
               'axis, a negative component points in the opposite direction, and a larger '
               'absolute value supplies more initial kinetic energy.\n'
               '\n'
               'For unit mass, the continuous rules are\n'
               '\n'
               '$$\\frac{dr}{dt}=\\nabla \\log p(\\theta),\\qquad '
               '\\frac{d\\theta}{dt}=r.$$\n'
               '\n'
               'Read the first rule as: posterior slope changes momentum. Read the second '
               'as: momentum changes position. Thus r = 0 is not permanent when the '
               'gradient is nonzero.',
      'experiment': 'First set both momentum controls to zero and keep 35 steps. In the '
                    'left plot, compare the teal HMC path with the dashed gray path that '
                    'has the posterior force removed. In the middle plot, inspect r₁, r₂, '
                    'and the momentum length. In the right plot, inspect how potential '
                    'energy becomes kinetic energy while their total stays almost '
                    'constant.',
      'exercise_intro': 'The path gives the motion. Your energy function gives the '
                        'bookkeeping that tells HMC whether the numerical path stayed '
                        'close to the intended dynamics.',
      'next': 'This lab used many small two-state updates as a complete motion. The next '
              'lab isolates one leapfrog step and shows why its half-full-half order '
              'controls energy error.',
      'prediction_answer': 'With both initial momentum components at zero, the HMC '
                           'particle still moves toward and then through the mode. It '
                           'starts away from the mode, so the posterior gradient changes '
                           'zero momentum into nonzero momentum before position changes. '
                           'At the mode the gradient is zero for an instant, but the '
                           'accumulated momentum is not, so the particle does not stop '
                           'there. The force-free gray reference stays fixed and isolates '
                           'this cause.'},
     {'act': 'Chapter 3 · Make approximate motion reversible and exact in distribution',
      'route': 'one-sided error → symmetric leapfrog → energy error → accept or repeat',
      'story': 'Lab 9 described two continuous changes: the posterior gradient changes '
               'momentum, and momentum changes position. A computer must approximate both '
               'changes with finite steps. The order now matters. Leapfrog places the '
               'position update between two matching half momentum updates. This lets the '
               'numerical path run backward through the same operations. The remaining '
               'energy error becomes an explicit accept-or-reject probability instead of '
               'an invisible sampling bias.',
      'mission': 'Implement one reversible leapfrog step',
      'exercise': '**Why this code exists.** Repeating one symmetric step creates the '
                  'numerical trajectory used by HMC. Its order makes the proposal '
                  'reversible, which the final accept-or-reject correction requires.\n'
                  '\n'
                  '**Your task.** Complete `leapfrog_step(q, p, step, grad_logp)`.\n'
                  '\n'
                  '**Inputs.** `q` is the code name for position θ. `p` is the code name '
                  'for momentum r. `step` is ε, the signed step size. `grad_logp(q)` '
                  'returns the posterior gradient at the current position.\n'
                  '\n'
                  '**Return.** The updated pair `(q, p)` with the same shapes as the '
                  'inputs.\n'
                  '\n'
                  '**Required order.** First add half of `step * grad_logp(q)` to '
                  'momentum. Next move position by one full `step * p` using that '
                  'half-updated momentum. Finally add another half momentum update, now '
                  'using the gradient at the new position.\n'
                  '\n'
                  '**Checks.** A small step must have small energy error. Applying your '
                  'function once with +ε and once with -ε must also return to the original '
                  'state, within floating-point error.',
      'prediction': 'Keep the number of steps fixed. The teal method uses **half momentum '
                    '→ full position → half momentum**. The orange method uses one full '
                    'momentum update before position. Which method should return closer to '
                    'its start when the clock runs backward? As **step size** grows, what '
                    'should happen to average HMC acceptance across many momentum draws?',
      'quiz': 'Explain two separate safeguards in HMC: why the half-full-half order makes '
              'leapfrog reversible, and how the Metropolis correction uses ΔH to accept a '
              'proposal or repeat the current position.',
      'chapter_opening': '',
      'bridge': 'Lab 9 used small updates to create motion and showed energy changing '
                'form. It did not yet explain which discrete update order can safely '
                'become an HMC proposal.',
      'scenario': 'The target remains the correlated baseline-demand and price-response '
                  'posterior. Its narrow direction changes force quickly, so a finite '
                  'update can easily use a stale force value and finish at the wrong '
                  'energy.',
      'goals': ['Explain why symmetric half momentum updates make leapfrog reversible.',
                'Compute the acceptance probability from signed Hamiltonian error.',
                'Separate correction of sampling bias from repair of a poor numerical '
                'path.'],
      'model': 'The code names are `q` for position θ and `p` for momentum r. One leapfrog '
               'step is\n'
               '\n'
               '$$p_{1/2}=p_0+\\frac{\\epsilon}{2}\\nabla\\log \\pi(q_0),$$\n'
               '$$q_1=q_0+\\epsilon p_{1/2},$$\n'
               '$$p_1=p_{1/2}+\\frac{\\epsilon}{2}\\nabla\\log \\pi(q_1).$$\n'
               '\n'
               'The first half uses the force at the old position. The second half uses '
               'the force at the new position. Because the sequence is symmetric, changing '
               'ε to -ε undoes the step.\n'
               '\n'
               'After several steps, define $\\Delta '
               'H=H(q_{new},p_{new})-H(q_{old},p_{old})$. Leapfrog is approximate, so ΔH '
               'is usually not exactly zero. The Metropolis correction is\n'
               '\n'
               '$$\\alpha=\\min(1,e^{-\\Delta H}),\\qquad '
               'u\\sim\\operatorname{Uniform}(0,1).$$\n'
               '\n'
               'Accept the proposed position when $u<\\alpha$. Otherwise, the next chain '
               'state repeats the old position.',
      'experiment': 'The left panel overlays symmetric leapfrog and a one-sided full-kick '
                    "method on the same posterior. The middle panel tracks each method's "
                    'signed energy error through the path. The right panel estimates '
                    'average leapfrog acceptance across fixed fresh momentum draws. The '
                    'result table also runs each selected path backward. Compare return '
                    'error before you compare distance traveled.',
      'exercise_intro': 'Your function implements the symmetric map. The local test now '
                        'checks both small energy error and the ability to undo one step '
                        'with a negative step size.',
      'next': 'You now have a reversible fixed-length trajectory and a valid endpoint '
              'correction. Next, fresh momentum and repeated accept-or-repeat transitions '
              'turn this deterministic engine into a posterior chain.',
      'prediction_answer': 'Symmetric leapfrog returns much closer to its starting '
                           'position when the clock runs backward. Its half-full-half '
                           'sequence reads the force at both ends and is its own inverse '
                           'when the step sign changes. The one-sided sequence does not '
                           'have this symmetry. As step size grows, leapfrog usually '
                           'creates larger energy errors, so average Metropolis acceptance '
                           'falls. One selected path can still have negative ΔH and '
                           'acceptance 1; the average trend, not every individual path, is '
                           'the useful tuning signal.'},
     {'act': 'Chapter 3 · Turn deterministic motion into a Markov chain',
      'route': 'fresh momentum → deterministic trajectory → accept or repeat → stored '
               'chain state',
      'story': 'Labs 9 and 10 built the deterministic engine inside HMC. That engine can '
               'produce a long, reversible path after one position and momentum are fixed. '
               'The unresolved question is what must surround that path before repeated '
               'endpoints can represent a posterior. This lab compares one continuously '
               'integrated path with complete transitions that redraw momentum, apply the '
               'Metropolis decision, and store one position each time.',
      'mission': 'Implement one complete HMC transition',
      'exercise': '**Why this code exists.** Lab 10 supplied a reversible trajectory '
                  'function. A sampler still needs to draw momentum, judge the proposed '
                  'endpoint, and return the position that must be stored in the chain.\n'
                  '\n'
                  '**Your task.** Complete `one_hmc_transition(position, log_target, '
                  'integrate, rng)`.\n'
                  '\n'
                  '**Inputs.** `position` is the current parameter vector q. '
                  '`log_target(q)` returns the target log density. `integrate(q, p)` runs '
                  'the fixed leapfrog trajectory and returns its endpoint `(proposed_q, '
                  'proposed_p)` before the final momentum flip. `rng` is a NumPy-like '
                  'random generator with `normal` and `uniform` methods.\n'
                  '\n'
                  '**Return.** `(next_position, accepted)`. `accepted` is a Python '
                  'boolean. On rejection, `next_position` must equal the original '
                  'position.\n'
                  '\n'
                  '**Translate the Hamiltonian into Python.** Lab 9 defined total energy '
                  'as potential plus kinetic energy. For the unit mass matrix used in '
                  'these small labs,\n'
                  '\n'
                  '$$H(q,p)=U(q)+K(p)=-\\log p(q)+\\frac12 p^T p.$$\n'
                  '\n'
                  'Here `log_target(q)` returns $\\log p(q)$. The minus sign converts a '
                  'high log density into low potential energy: likely positions sit low in '
                  'the energy landscape. `np.dot(p, p)` computes $p^T p=\\sum_i p_i^2$, '
                  'the squared length of the momentum vector. Multiplying it by one half '
                  'gives kinetic energy for unit mass. `float(...)` converts a NumPy '
                  'scalar into an ordinary Python scalar, which makes the later comparison '
                  "unambiguous. Therefore the current joint state's energy is\n"
                  '\n'
                  '```python\n'
                  'old_h = -float(log_target(old_position)) + 0.5 * float(\n'
                  '    np.dot(old_momentum, old_momentum)\n'
                  ')\n'
                  '```\n'
                  '\n'
                  'Compute `new_h` with the same formula and the proposed position and '
                  'momentum. The final momentum sign flip does not change kinetic energy '
                  'because $(-p)^T(-p)=p^Tp$. HMC compares `old_h` and `new_h`; it does '
                  'not treat either value alone as an acceptance probability.\n'
                  '\n'
                  '**Required sequence.** Copy the old position. Draw fresh '
                  'standard-Normal momentum with the same shape. Call `integrate`. Flip '
                  'the sign of the proposed momentum so that the proposal uses the '
                  'reversible map from Lab 10. Compute old and proposed Hamiltonians. '
                  'Accept when `log(rng.uniform()) < min(0, old_h - new_h)`; otherwise '
                  'return the old position.\n'
                  '\n'
                  '**Checks.** The test verifies that fresh momentum reaches the '
                  'integrator, an equal-energy proposal is accepted, and a very '
                  'high-energy proposal is rejected with the old position repeated.',
      'prediction': 'Imagine two repeated procedures on the same target. Procedure A '
                    'starts once at (q₀, p₀) and then keeps applying deterministic '
                    'leapfrog updates without any new random draw. Procedure B starts each '
                    'transition by drawing fresh momentum, runs a fixed leapfrog '
                    'trajectory, draws a Uniform value for the Metropolis decision, and '
                    'stores either the proposal or the old position. Before seeing the '
                    'experiment, which procedure do you expect can represent the full '
                    'posterior over time? Explain what you think each random draw '
                    'contributes.',
      'quiz': 'After position and momentum are fixed, a leapfrog trajectory is '
              'deterministic. Name the two random draws in one HMC transition, and explain '
              'exactly what the chain stores after a rejected proposal.',
      'chapter_opening': '',
      'bridge': 'Lab 10 produced a valid proposed path and explained the final energy '
                'correction. It did not yet show how many such proposals become a set of '
                'posterior draws.',
      'scenario': 'The bakery needs many plausible pairs of baseline demand and price '
                  'response, not one mechanical orbit through parameter space. Each stored '
                  'pair will later support uncertainty calculations and predictions.',
      'goals': ['Separate a leapfrog step, trajectory, HMC transition, and Markov chain.',
                'Identify the two sources of randomness in an HMC transition.',
                'Implement fresh momentum, Hamiltonian acceptance, and rejection as a '
                'repeated state.'],
      'model': 'Start one transition at the current position $q_t$ and draw fresh '
               'auxiliary momentum $p_0\\sim\\mathcal N(0,I)$. For the unit mass used '
               'here, the joint Hamiltonian is\n'
               '\n'
               '$$H(q,p)=-\\operatorname{log\\_target}(q)+\\frac12p^Tp.$$\n'
               '\n'
               'The first term is potential energy and the second is kinetic energy. Run '
               '$L$ leapfrog steps to produce $(q^*,p^*)$, then flip the proposed momentum '
               'sign. Compute old and proposed Hamiltonians with the same formula. Draw '
               '$u\\sim\\operatorname{Uniform}(0,1)$ and accept when '
               '$u<\\min(1,\\exp[H(q_t,p_0)-H(q^*,p^*)])$. Otherwise store $q_t$ again.',
      'experiment': 'After you lock your prediction and run the experiment, the first '
                    'panel will expand one transition into its start, path, proposal, and '
                    'decision. The second will show positions stored by fixed-length HMC '
                    'with fresh momentum. The third will show one continuous no-refresh '
                    'path. Its time colors and Hamiltonian-error inset will let you test '
                    'whether visible area coverage is sufficient evidence of posterior '
                    'sampling.',
      'exercise_intro': 'The integrator argument contains the trajectory mechanism from '
                        "Lab 10. Your function adds the sampler's outer layer. Before you "
                        'write it, translate the Hamiltonian equation below into the two '
                        'scalar values `old_h` and `new_h`.',
      'next': 'This lab uses a fixed number L of leapfrog steps for every transition. The '
              'complete sampler now works, but L is still a manual choice. NUTS is the '
              'next lab because it chooses when a trajectory has traveled far enough and '
              'starts to double back.',
      'prediction_answer': 'Broad-looking coverage in the orange position plot is not '
                           'sufficient evidence of posterior sampling. The plot hides '
                           'momentum. Even at one fixed total Hamiltonian, potential '
                           'energy and kinetic energy can trade places, so the position q '
                           'can cross many density contours. Also, leapfrog is a discrete '
                           'approximation. Its phase error and small Hamiltonian error can '
                           'make a repeated curve shift and appear to fill a shape. Stable '
                           'leapfrog usually keeps this error near a nearby shadow energy '
                           'instead of creating valid new posterior draws. The path still '
                           'comes from one initial (q,p) state and has no random '
                           'transition across joint energy shells. Proper HMC draws fresh '
                           'Normal momentum for each transition and uses a Uniform '
                           'Metropolis decision, so it can preserve and represent the full '
                           'target distribution.'},
     {'act': 'Chapter 3 · Adapt trajectory length with NUTS',
      'route': 'fixed trajectory → U-turn test → adaptive NUTS length',
      'story': 'A good step size does not tell you how many steps to take. NUTS watches '
               'the geometry of the path and stops expansion when travel starts to '
               'reverse.',
      'mission': 'Implement the U-turn criterion',
      'exercise': '**Why this code exists.** This dot-product check is the core geometric '
                  'idea behind NUTS trajectory stopping.\n'
                  '\n'
                  '**Your task.** Complete `is_uturn(start, current, momentum)`.\n'
                  '\n'
                  '**Inputs.** `start` is the initial position. `current` is the current '
                  'position. `momentum` is the current direction of motion.\n'
                  '\n'
                  '**Return.** A Python boolean. Return `True` when the displacement and '
                  'momentum point in opposing directions.\n'
                  '\n'
                  '**Suggested steps.** Compute `displacement = current - start`. Compute '
                  'its dot product with momentum. A negative result means a U-turn.\n'
                  '\n'
                  '**Checkpoint.** At current point `[2,0]`, momentum `[-1,0]` must report '
                  'a U-turn; momentum `[1,0]` must not.',
      'prediction': 'Displacement points right while current momentum points left. Is the '
                    'particle still increasing its distance from the start? What sign do '
                    'you expect for their dot product?',
      'quiz': 'State one waste caused by an HMC path that is too short and one waste '
              'caused by a path that is too long.',
      'chapter_opening': '',
      'bridge': 'Lab 11 assembled fixed-length HMC transitions into a posterior chain. The '
                'remaining manual choice is L, the number of leapfrog steps. A value that '
                'is too short wastes travel, while one that is too long can double back.',
      'scenario': 'The price-demand posterior can be long and curved. A useful stop rule '
                  'must compare current motion with displacement from the trajectory '
                  'start.',
      'goals': ['Interpret the U-turn dot product.',
                'Implement a minimal detector.',
                'Inspect warmup, R-hat, ESS, and divergences in PyMC.'],
      'model': 'A simple check is\n'
               '\n'
               '$$d=\\theta-\\theta_0,\\qquad d^T r<0.$$\n'
               '\n'
               'If the dot product is negative, displacement and momentum form an angle '
               'larger than 90 degrees. The particle has started to move back toward its '
               'start. Production NUTS checks a tree, not only one endpoint.',
      'experiment': 'Use manual geometry first. The plot draws displacement and current '
                    'momentum at the detected stop. Then select PyMC NUTS to see how a '
                    'production library packages trajectory building and diagnostics.',
      'exercise_intro': 'The exercise implements the geometric core only. It does not '
                        'attempt tree building, detailed balance checks, or warmup '
                        'adaptation.',
      'next': 'An adaptive algorithm can still fail. Next, you use multiple chains to '
              'decide whether posterior draws are trustworthy.',
      'prediction_answer': 'The particle is no longer increasing its distance from the '
                           'start. A right-pointing displacement and left-pointing '
                           'momentum form an obtuse angle, so their dot product is '
                           'negative. This is the geometric signal that the trajectory has '
                           'started a U-turn.'},
     {'act': 'Chapter 3 · Diagnose the sampling regime',
      'route': 'samples → multiple chains → trust diagnostics',
      'story': 'A sampler can run without an exception and still give a wrong summary. You '
               'now compare independent starts to test whether they reached the same '
               'sampling regime.',
      'mission': 'Implement a basic R-hat estimate',
      'exercise': '**Why this code exists.** R-hat asks whether several chains show one '
                  'common sampling regime. It compares movement inside chains with '
                  'disagreement between their centers.\n'
                  '\n'
                  '**Your task.** Complete `basic_rhat(chains)` for an array shaped '
                  '`(number_of_chains, draws_per_chain)`.\n'
                  '\n'
                  '**Input layout.** Each row is one chain. Each column is one retained '
                  'draw. If `chains.shape` is `(m, n)`, then `m` is the number of chains '
                  'and `n` is the number of draws per chain. The input contains at least '
                  'two equal-length chains.\n'
                  '\n'
                  '**Step 1 — row summaries.** Compute one mean and one sample variance '
                  'for each row. Use `axis=1` because draws run across the columns. Use '
                  '`ddof=1` for both sample-variance calculations in this exercise.\n'
                  '\n'
                  '**Step 2 — within-chain variance.** Average the `m` row variances.\n'
                  '\n'
                  '$$\n'
                  'W=\\operatorname{mean}(s_1^2,\\ldots,s_m^2)\n'
                  '$$\n'
                  '\n'
                  '`W` answers: how much does a typical chain vary around its own mean?\n'
                  '\n'
                  '**Step 3 — between-chain variance.** Take the sample variance of the '
                  '`m` chain means and multiply it by `n`.\n'
                  '\n'
                  '$$\n'
                  'B=n\\operatorname{Var}(\\bar{x}_1,\\ldots,\\bar{x}_m)\n'
                  '$$\n'
                  '\n'
                  '`B` answers: how strongly do the chain centers disagree, on the same '
                  'variance scale as `W`?\n'
                  '\n'
                  '**Step 4 — combine and compare.** Form the pooled variance estimate, '
                  'then return the square root of its ratio to `W`.\n'
                  '\n'
                  '$$\n'
                  '\\widehat{V}=\\frac{n-1}{n}W+\\frac{B}{n}\n'
                  '$$\n'
                  '\n'
                  '$$\n'
                  '\\widehat{R}=\\sqrt{\\frac{\\widehat{V}}{W}}\n'
                  '$$\n'
                  '\n'
                  '**Small check before coding.** For `chains = np.array([[0, 2], [2, '
                  '4]])`, you should get `W = 2`, `B = 4`, and R-hat approximately '
                  '`1.225`.\n'
                  '\n'
                  '**Return.** One scalar R-hat estimate. Four independent standard-Normal '
                  'chains with many draws should give a value near `1`.\n'
                  '\n'
                  '**Checkpoint.** The local test checks a well-mixed case, a '
                  'separated-chain case, and the small hand-calculated example above.',
      'prediction': 'Four chains have stable individual means, but each stays in a '
                    'different posterior mode. Is convergence good or bad? Which visible '
                    'comparison reveals the failure?',
      'quiz': 'MCMC does not converge to a point. What does it converge to, and what '
              'behavior should several chains share in that regime?',
      'chapter_opening': '',
      'bridge': 'NUTS removes one tuning choice, but it cannot certify that all important '
                'regions were explored. Diagnostics compare evidence from time, chains, '
                'and geometry.',
      'scenario': 'Weekday and event-day demand can create separated posterior regions. A '
                  'chain can look stable because it stays inside one region, not because '
                  'it explored the full target.',
      'goals': ['Distinguish stationary movement from point convergence.',
                'Read R-hat, ESS, and cumulative means.',
                'Diagnose a deliberately broken example.'],
      'model': 'R-hat compares pooled variance with within-chain variance. When chains '
               'occupy different regions, between-chain variation raises R-hat. $MCSE '
               '\\approx posterior\\ SD/\\sqrt{ESS}$ converts sampling information into '
               'uncertainty in a reported posterior summary.',
      'experiment': 'Increase broken separation. Each trace can look locally stable while '
                    'chain histograms and cumulative means disagree. Read all four panels '
                    'before you look at the R-hat summary.',
      'exercise_intro': 'The implementation is a teaching version. Production R-hat uses '
                        'rank normalization and split chains, but the '
                        'between-versus-within logic starts here.',
      'next': 'Reliable sampling can be expensive. Chapter 4 asks what changes when '
              'inference becomes optimization over an approximate distribution.',
      'prediction_answer': 'Convergence is bad even if each chain has a stable mean. Each '
                           'chain is stable in a different mode, so none represents the '
                           'full posterior. Compare traces and chain histograms across '
                           'chains; between-chain disagreement also makes R-hat large.'},
     {'act': 'Chapter 4 · Approximate and latent inference',
      'route': 'posterior sampling → fit q → optimize ELBO',
      'story': 'NUTS can be accurate but expensive. For faster repeated decisions, you can '
               'fit a simple distribution to the target. This turns inference back into '
               'optimization, but the learned object is now a distribution.',
      'mission': 'Estimate the ELBO by Monte Carlo',
      'exercise': '**Why this code exists.** The ELBO is the objective used to fit '
                  'variational parameters.\n'
                  '\n'
                  '**Your task.** Complete `mc_elbo(log_joint, mean, log_sd, eps)` for a '
                  'one-dimensional Gaussian `q`.\n'
                  '\n'
                  '**Inputs.** `log_joint(z)` returns the target log score. `mean` and '
                  '`log_sd` parameterize `q`. `eps` is an array of fixed standard-Normal '
                  'noise draws.\n'
                  '\n'
                  '**Return.** The sample mean of `log_joint(z) - log_q(z)`.\n'
                  '\n'
                  '**Suggested steps.** Reparameterize with `sd=exp(log_sd)` and '
                  '`z=mean+sd*eps`. Compute the Normal log-density of each `z` under `q`. '
                  'Average target log score minus `log_q`.\n'
                  '\n'
                  '**Checkpoint.** The supplied standard-Normal example must return a '
                  'finite scalar.',
      'prediction': "Can one Gaussian match a strongly skewed target's peak, long tail, "
                    'and asymmetry at the same time? Name the feature it must compromise.',
      'quiz': 'What object does VI optimize, and what object does MAP optimize? State the '
              'difference without using only the word distribution.',
      'chapter_opening': 'Chapter 4 studies two ways to avoid full posterior sampling. VI '
                         'optimizes a tractable distribution. EM alternates around hidden '
                         'customer assignments. Both gain speed by changing the learned '
                         'object or representation.',
      'bridge': 'NUTS aims at the full posterior. Repeated bakery planning can need a '
                'faster approximation, especially when the model grows.',
      'scenario': 'A delivery-capacity effect makes a demand parameter posterior skewed. '
                  'You will try to represent it with one adjustable Gaussian distribution '
                  'q.',
      'goals': ['Interpret the ELBO.',
                'Adjust an approximation manually before optimizing it.',
                'Compare VI shape with target shape.'],
      'model': '$$\\operatorname{ELBO}(\\phi)=\n'
               '\\mathbb E_{q_\\phi(\\theta)}\n'
               '[\\log p(D,\\theta)-\\log q_\\phi(\\theta)].$$\n'
               '\n'
               'The value phi contains the approximation mean and log scale. The '
               'joint-model term rewards plausible samples. The contribution from negative '
               'log q rewards entropy and prevents a zero-width point solution.',
      'experiment': 'First move and scale your Gaussian q manually. Then compare it with '
                    'the optimized q and the skewed target. Inspect both the peak and the '
                    'tail.',
      'exercise_intro': 'Fixed standard-Normal noise makes the sample a differentiable '
                        'function of variational parameters. This is the '
                        'reparameterization trick used by autodiff VI.',
      'next': 'A good optimizer can still return a poor approximation. Next, you separate '
              'optimization failure from family failure.',
      'prediction_answer': "One Gaussian cannot match a strongly skewed target's peak, "
                           'long tail, and asymmetry at the same time. It must compromise. '
                           'A fit near the peak often becomes too narrow and misses tail '
                           'mass; a wider fit covers more tail but gives a poorer central '
                           'shape.'},
     {'act': 'Chapter 4 · Expose approximation-family limits',
      'route': 'approximation family → geometric mismatch → failure',
      'story': 'Optimization cannot create a shape that the approximation family does not '
               'contain. You will make the mean-field assumption explicit before you '
               'inspect its failures.',
      'mission': 'Implement reparameterized diagonal-Gaussian sampling',
      'exercise': '**Why this code exists.** This operation is the sampling path through '
                  'which automatic differentiation trains a mean-field variational '
                  'approximation.\n'
                  '\n'
                  '**Your task.** Complete `diagonal_gaussian_sample(mean, log_sd, eps)`.\n'
                  '\n'
                  '**Inputs.** `mean` and `log_sd` have one entry per dimension. `eps` has '
                  'shape `(number_of_samples, dimensions)` and contains standard-Normal '
                  'noise.\n'
                  '\n'
                  '**Return.** Samples with the same shape as `eps`, using `mean + '
                  'exp(log_sd) * eps` with NumPy broadcasting.\n'
                  '\n'
                  '**Checkpoint.** With zero noise, five two-dimensional samples must all '
                  'equal the supplied mean `[1,2]`.',
      'prediction': 'Rank round, correlated, and banana-shaped targets from easiest to '
                    'hardest for a diagonal Gaussian. Explain the missing structure in the '
                    'hardest case.',
      'quiz': 'If a stable VI fit underestimates uncertainty, why can the approximation '
              'family be the cause even when optimization succeeded?',
      'chapter_opening': '',
      'bridge': 'Lab 14 showed a Gaussian compromise on one skewed target. We now compare '
                'several geometries that require different kinds of expressiveness.',
      'scenario': 'Bakery demand can have a long event tail, separate weekday and weekend '
                  'regimes, or curved dependence between price and baseline demand.',
      'goals': ['Identify skew, modes, and dependence.',
                'Compare diagonal and full-covariance Gaussian limits.',
                'Avoid blaming all VI error on the optimizer.'],
      'model': 'Mean-field VI assumes\n'
               '\n'
               '$$q(\\theta)=\\prod_j q_j(\\theta_j).$$\n'
               '\n'
               'This rules out dependence. A full-covariance Gaussian can rotate and '
               'stretch one ellipse, but it still cannot represent a curved banana or '
               'several separated modes.',
      'experiment': 'Switch among skewed, multimodal, and banana targets. For each target, '
                    'name the geometric property that the shown Gaussian cannot express.',
      'exercise_intro': 'This vectorized operation is the sample path through which '
                        'gradients reach every mean and log-scale parameter.',
      'next': 'VI approximates continuous uncertainty. The next bakery problem hides '
              'discrete customer groups and leads to EM.',
      'prediction_answer': 'For a diagonal Gaussian, a round target is easiest, a '
                           'correlated elliptical target is harder, and a banana target is '
                           'hardest. The diagonal family cannot represent correlation. '
                           'Even a full Gaussian can rotate one ellipse but cannot bend it '
                           'into curved dependence.'},
     {'act': 'Chapter 4 · Alternate around hidden customer groups',
      'route': 'unknown labels → soft assignments → parameter updates',
      'story': 'Some inference problems hide discrete assignments instead of continuous '
               'parameters. In a mixture, estimating labels and estimating component means '
               'make each other easier. EM alternates these two simpler tasks.',
      'mission': 'Implement the Gaussian-mixture M-step',
      'exercise': '**Why this code exists.** The M-step turns soft E-step assignments into '
                  'updated mixture parameters.\n'
                  '\n'
                  '**Your task.** Complete `m_step(x, responsibilities)`.\n'
                  '\n'
                  '**Inputs.** `x` is a one-dimensional array of observations. '
                  '`responsibilities` has shape `(observations, components)`; each row '
                  'contains soft component probabilities.\n'
                  '\n'
                  '**Return.** `(weights, means)`. Each weight is its component '
                  'responsibility mass divided by the number of observations. Each mean is '
                  'the responsibility-weighted data mean.\n'
                  '\n'
                  '**Suggested steps.** Sum responsibilities down the observation axis to '
                  'get `N_k`. Divide by sample count for weights. Use `sum(r_ik * x_i) / '
                  'N_k` for each mean.\n'
                  '\n'
                  '**Checkpoint.** Hard assignments for points `[-2,-1]` and `[1,2]` must '
                  'give weights `[0.5,0.5]` and means `[-1.5,1.5]`.',
      'prediction': 'Both component means start at exactly the same time with equal '
                    'weights. Can deterministic EM break the symmetry without a '
                    'disturbance? Explain why or why not.',
      'quiz': 'What quantity changes in the E-step? Which two parameter types change in '
              'this M-step?',
      'chapter_opening': '',
      'bridge': 'Not every latent variable is a continuous parameter. Customer visit times '
                'can come from hidden morning and afternoon groups.',
      'scenario': 'The data record each visit time as hours from noon, but they do not '
                  'label the visit as morning or afternoon. If labels were known, group '
                  'means would be easy to fit. If means were known, labels would be easy '
                  'to estimate.',
      'goals': ['Interpret soft responsibilities.',
                'Implement the Gaussian-mixture M-step.',
                'Verify non-decreasing observed-data log likelihood.'],
      'model': '$$p(x_i)=\\sum_{k=1}^{K}\\pi_k\\,\n'
               '\\mathcal N(x_i\\mid\\mu_k,\\sigma_k).$$\n'
               '\n'
               'The E-step computes responsibilities $r_{ik}=P(z_i=k\\mid x_i)$. The '
               'M-step uses $N_k=\\sum_i r_{ik}$, $\\pi_k=N_k/N$, and $\\mu_k=\\sum_i '
               'r_{ik}x_i/N_k$. Each responsibility is the current probability that visit '
               'i belongs to component k under the present mixture parameters, not a '
               'permanent observed label.',
      'experiment': 'Point color mixes the two responsibilities. Component means are large '
                    'markers. The second panel verifies that each EM iteration does not '
                    'decrease observed-data log likelihood.',
      'exercise_intro': 'Responsibilities act like fractional labels. The M-step is '
                        'ordinary weighted estimation once those soft labels are '
                        'available.',
      'next': 'You now have point optimization, distribution optimization, latent '
              'alternation, and sampling. Next, compare them by the object required by the '
              'decision.',
      'prediction_answer': 'Deterministic EM cannot break exact symmetry by itself. '
                           'Identical component parameters give identical responsibilities '
                           'in the E-step, so the M-step returns identical updates. A '
                           'small initial perturbation, unequal initialization, or another '
                           'source of asymmetry is required.'},
     {'act': 'Chapter 4 · Select by output and failure mode',
      'route': 'MAP / EM / VI / MCMC → different outputs',
      'story': 'You now have several algorithms, but they do not return interchangeable '
               'results. Before you choose by speed, identify the object that the '
               'downstream decision needs.',
      'mission': 'Map methods to output objects',
      'exercise': '**Why this code exists.** Method selection starts with the object you '
                  'need, not with an algorithm name.\n'
                  '\n'
                  '**Your task.** Complete `output_object(method)`.\n'
                  '\n'
                  "**Inputs.** `method` is one of `'MAP'`, `'EM'`, `'VI'`, or `'MCMC'`.\n"
                  '\n'
                  "**Return.** Use these exact descriptions: MAP → `'point'`; EM → `'point "
                  "+ responsibilities'`; VI → `'approximate distribution'`; MCMC → "
                  "`'posterior samples'`.\n"
                  '\n'
                  '**Suggested steps.** Use a dictionary from method name to output '
                  'description and return the selected value.\n'
                  '\n'
                  '**Checkpoint.** The test checks the exact MAP string and checks that '
                  "the MCMC description contains `'samples'`.",
      'prediction': 'A multimodal bakery posterior must support calibrated decisions. '
                    'Which method is the safest first reference among MAP, EM, VI, and '
                    'MCMC or NUTS? What output object do you need?',
      'quiz': 'Give one precise difference between MAP and VI, then one precise difference '
              'between VI and MCMC.',
      'chapter_opening': '',
      'bridge': 'The previous algorithms can all produce a fitted result, but those '
                'results do not carry the same information.',
      'scenario': 'The bakery has several tasks: one production number, customer segment '
                  'assignments, fast approximate uncertainty, and calibrated uncertainty '
                  'for an expensive decision.',
      'goals': ['Map each method to its learned object.',
                'Compare uncertainty behavior.',
                'Name one important failure mode per method.'],
      'model': 'Use this decision order: **required output → latent structure → posterior '
               'geometry → compute budget → diagnostics**. Speed is meaningful only after '
               'the output object is sufficient for the bakery decision.',
      'experiment': 'Set whether uncertainty is required. Read the actual output object '
                    'and main risk in every table row. Do not interpret the recommendation '
                    'as a universal ranking.',
      'exercise_intro': 'The mapping forces method selection to begin with the result that '
                        'downstream code will consume.',
      'next': 'Chapter 5 adds several bakery branches. Their unequal data sizes create a '
              'reason to share information with a hierarchy.',
      'prediction_answer': 'For a multimodal posterior and calibrated decisions, MCMC or '
                           'NUTS is the safest first reference among these choices. The '
                           'required output is posterior draws that preserve uncertainty '
                           'and mode weight. You must still verify that chains move '
                           'between modes; NUTS cannot guarantee that.'},
     {'act': 'Chapter 5 · Build hierarchical bakery models',
      'route': 'separate groups → shared population → partial pooling',
      'story': 'Return to the coupon rules. Some rules have many investigations and others '
               'have almost none. A hierarchy lets weak groups borrow information without '
               'forcing every group to be equal.',
      'mission': 'Compute a precision-weighted shrinkage mean',
      'exercise': '**Why this code exists.** This closed-form update shows partial pooling '
                  'before PyMC packages the same idea in a larger model.\n'
                  '\n'
                  '**Your task.** Complete `shrink(raw, se, population_mean, '
                  'population_sd)`.\n'
                  '\n'
                  '**Inputs.** `raw` is a group estimate. `se` is its standard error. '
                  '`population_mean` is the shared center. `population_sd` controls '
                  'allowed group variation.\n'
                  '\n'
                  '**Return.** The precision-weighted posterior mean. Use observation '
                  'precision `1/se²` and population precision `1/population_sd²`.\n'
                  '\n'
                  '**Suggested steps.** Multiply each mean by its precision. Add the '
                  'weighted means. Divide by total precision.\n'
                  '\n'
                  '**Checkpoint.** A weak group under a tight population must move closer '
                  'to `0.6` than a strong group under a broad population.',
      'prediction': 'As population standard deviation approaches zero, where does the 1/1 '
                    'branch estimate move? Name the exact shared reference point.',
      'quiz': 'Two branches have the same raw rate, but one has a much larger standard '
              'error. Which branch shrinks more, and why?',
      'chapter_opening': 'Chapter 5 moves from one bakery to related branches and coupon '
                         'rules. Hierarchies improve estimates by sharing information, but '
                         'they also create funnel geometry that affects HMC.',
      'bridge': 'Separate branch models waste information about their similarity. One '
                'pooled rate ignores real branch differences. Partial pooling gives a '
                'controlled middle position.',
      'scenario': 'Branches A, B, and C observe coupon purchases of 70/100, 7/10, and 1/1. '
                  'The raw rates are 0.70, 0.70, and 1.00, but the evidence behind those '
                  'rates is very different.',
      'goals': ['Compare no pooling, complete pooling, and partial pooling.',
                'Read shrinkage as precision weighting.',
                'Predict which branch moves most.'],
      'model': 'Using a Normal approximation,\n'
               '\n'
               '$$\\hat p_j\\mid p_j\\sim\\mathcal N(p_j,se_j),\\qquad\n'
               'p_j\\sim\\mathcal N(\\mu_{pop},\\tau).$$\n'
               '\n'
               'The posterior mean weights the branch estimate by $1/se_j^2$ and the '
               'population mean by $1/\\tau^2$. A smaller tau means branches are assumed '
               'to be more similar.',
      'experiment': 'Arrows start at raw branch rates and end at partially pooled '
                    'estimates. Reduce population standard deviation and compare the '
                    'established branch with the new 1/1 branch.',
      'exercise_intro': 'This closed-form calculation exposes partial pooling before PyMC '
                        'represents the same idea inside a larger posterior.',
      'next': 'The next lab applies the same sharing idea to three coupon-targeting rules '
              'and propagates uncertainty into purchase odds.',
      'prediction_answer': 'As population standard deviation approaches zero, the 1/1 '
                           'branch moves toward the shared population mean. In that limit, '
                           'the hierarchy allows almost no real branch-to-branch '
                           'variation, so the low-data branch is governed mainly by the '
                           'common estimate.'},
     {'act': 'Chapter 5 · Propagate rule uncertainty into a decision',
      'route': 'counts → latent rates → posterior odds',
      'story': 'You can now model the real three-rule system. Each rule has a sensitivity '
               'and a false-fire rate, but all rules share population distributions. The '
               'final decision uses their ratio, not only a point estimate.',
      'mission': 'Classify variables in the hierarchical model',
      'exercise': '**Why this code exists.** Correct variable roles make the PyMC model, '
                  'diagnostics, and downstream calculations easier to reason about.\n'
                  '\n'
                  '**Your task.** Complete `classify_variables()`.\n'
                  '\n'
                  '**Inputs.** None. Use the variable names from the model shown in this '
                  'lab.\n'
                  '\n'
                  "**Return.** A dictionary with three keys. `'observed'` must include "
                  "`k_pos` and `k_neg`. `'latent'` must include `mu_s`, `sigma_s`, `eta`, "
                  "`mu_f`, `sigma_f`, and `xi`. `'deterministic'` must include `s`, `f`, "
                  'and `s_over_f`.\n'
                  '\n'
                  '**Checkpoint.** The test checks that key observed, latent, and '
                  'deterministic names are in the correct lists.',
      'prediction': 'Which rule has wider uncertainty in s_j/f_j: one reviewed on 20 '
                    'buyers or one reviewed on only 2 buyers? Name the missing evidence '
                    'that causes the width.',
      'quiz': 'Give one named example from this model of an observed count, a latent logit '
              'parameter, a deterministic probability, and a derived likelihood ratio.',
      'chapter_opening': '',
      'bridge': 'Lab 18 pooled one rate per branch. A targeting rule has two relevant '
                'rates, and the downstream decision depends on their ratio.',
      'scenario': 'For each coupon rule, reviewed buyers show how often the rule reaches a '
                  'buyer. Reviewed nonbuyers show how often it wastes an offer. Rule '
                  'firing is evidence about purchase intent; it is not proof of causal '
                  'coupon impact.',
      'goals': ['Classify observed, latent, deterministic, and derived variables.',
                'Fit two linked hierarchies.',
                "Propagate posterior samples through Bayes' odds rule."],
      'model': '$$\\mu_s\\sim\\mathcal N(0,2),\\quad\n'
               '\\sigma_s\\sim\\operatorname{HalfNormal}(1),\\quad\n'
               '\\eta_j\\sim\\mathcal N(\\mu_s,\\sigma_s),\\quad\n'
               's_j=\\operatorname{logit}^{-1}(\\eta_j)$$\n'
               '\n'
               '$$k^{buyer}_j\\sim\\operatorname{Binomial}(n^{buyer}_j,s_j)$$\n'
               '\n'
               'A parallel hierarchy defines $f_j$, the offer rate among nonbuyers. When '
               'rule j fires,\n'
               '\n'
               '$$O(purchase\\mid fire_j)=O(purchase)\\times\\frac{s_j}{f_j}.$$',
      'experiment': 'Select one coupon rule and prior purchase odds. The first two panels '
                    'show uncertainty in buyer coverage and nonbuyer offer rate. The third '
                    'carries every paired posterior draw into purchase odds.',
      'exercise_intro': 'Correct variable roles prevent common errors when you read a PyMC '
                        'model, diagnose chains, and propagate posterior draws.',
      'next': 'The hierarchy is statistically useful, but its centered coordinates can be '
              'difficult for HMC. Next, you see the funnel directly.',
      'prediction_answer': 'The rule reviewed on only two buyers has wider uncertainty in '
                           's_j/f_j. Two buyer observations provide little information '
                           'about buyer coverage s_j. Uncertainty in the nonbuyer rate f_j '
                           'also enters the ratio, so weak evidence in either part '
                           'broadens the final quantity.'},
     {'act': "Chapter 5 · Diagnose the hierarchy's geometry",
      'route': 'valid hierarchy → funnel geometry → sampling difficulty',
      'story': 'The coupon hierarchy is statistically useful, but its coordinates can '
               'create a narrow funnel. You will generate the centered transform that '
               'makes the neck visible.',
      'mission': 'Generate a centered funnel value',
      'exercise': '**Why this code exists.** The transform shows how a small top-level '
                  'scale squeezes every lower-level parameter into a narrow region.\n'
                  '\n'
                  '**Your task.** Complete `funnel_sample(v, z)`.\n'
                  '\n'
                  '**Inputs.** `v` is the log-variance control. `z` is a scalar or NumPy '
                  'array of standard-Normal values.\n'
                  '\n'
                  '**Return.** `exp(v / 2) * z`.\n'
                  '\n'
                  '**Suggested steps.** Convert log variance to standard deviation with '
                  '`exp(v/2)`. Multiply every entry of `z` by that scale.\n'
                  '\n'
                  '**Checkpoint.** At `v=-8`, the result must equal `exp(-4) * z`.',
      'prediction': 'Which region is harder for a centered sampler, the wide mouth or '
                    'narrow neck? What happens to the allowed width of lower-level '
                    'parameters there?',
      'quiz': 'How can a probability model be statistically valid but computationally '
              'difficult for HMC? Name the funnel feature that creates the problem.',
      'chapter_opening': '',
      'bridge': 'The population scales in Lab 19 can become small. Centered group effects '
                'then occupy a narrow region that widens rapidly as the scale grows.',
      'scenario': 'A small population scale means the branches or coupon rules are almost '
                  'identical. A large scale permits large differences. Centered '
                  'coordinates couple each group effect directly to that scale.',
      'goals': ["Recognize Neal's funnel.",
                'Connect local scale to leapfrog step size.',
                'Interpret divergences as geometric warnings.'],
      'model': '$$v\\sim\\mathcal N(0,3),\\qquad\n'
               'x\\mid v\\sim\\mathcal N(0,e^{v/2}).$$\n'
               '\n'
               'The value v controls log variance. At negative v, allowed x values form a '
               'narrow neck. At positive v, they form a wide mouth. One global HMC step '
               'size must handle both.',
      'experiment': 'Move the neck-depth control and read the conditional standard '
                    'deviation. In local PyMC mode, inspect where divergent transitions '
                    'appear on the funnel.',
      'exercise_intro': 'The transform makes the changing conditional scale explicit. It '
                        'is the minimal mechanism behind the funnel shape.',
      'next': 'You do not need to remove the hierarchy. The next lab changes coordinates '
              'while keeping the same probability model.',
      'prediction_answer': 'The narrow neck is harder for a centered sampler. When the '
                           'top-level scale is small, the allowed width of every '
                           'lower-level parameter becomes tiny. A step size that works in '
                           'the wide mouth can jump across the neck and create large '
                           'integration error.'},
     {'act': 'Chapter 5 · Reparameterize the same hierarchy',
      'route': 'centered funnel → standard coordinates → non-centered model',
      'story': 'You do not need to remove the hierarchy. You can express the same group '
               'parameter through a standard variable and a deterministic transform. This '
               'often straightens weak-data geometry.',
      'mission': 'Implement the non-centered transform',
      'exercise': '**Why this code exists.** This is the coordinate change used in '
                  'non-centered hierarchical models.\n'
                  '\n'
                  '**Your task.** Complete `noncenter(mu, sigma, z)`.\n'
                  '\n'
                  '**Inputs.** `mu` is the population center. `sigma` is the positive '
                  'population scale. `z` is a scalar or NumPy array in standard-Normal '
                  'coordinates.\n'
                  '\n'
                  '**Return.** `mu + sigma * z` with NumPy broadcasting.\n'
                  '\n'
                  '**Checkpoint.** With `mu=2`, `sigma=0.5`, and `z=[-1,1]`, the result '
                  'must be `[1.5,2.5]`.',
      'prediction': 'When group data are weak, which form often samples better, centered '
                    'or non-centered? Describe the geometric change rather than only '
                    'naming the form.',
      'quiz': 'Does non-centering change the probability model or only its coordinates? '
              'Explain how the distribution of eta_j stays the same.',
      'chapter_opening': '',
      'bridge': 'The funnel problem comes from coordinates, not from the idea of partial '
                'pooling itself. A non-centered form separates standard-scale variation '
                'from the population transform.',
      'scenario': 'When branch data are weak, population parameters determine most of each '
                  'branch effect. Standardized latent variables can then give HMC a more '
                  'uniform scale.',
      'goals': ['Derive the non-centered transform.',
                'Compare centered and non-centered geometry.',
                'Use diagnostics to choose parameterization.'],
      'model': 'Centered:\n'
               '\n'
               '$$\\eta_j\\sim\\mathcal N(\\mu,\\sigma).$$\n'
               '\n'
               'Non-centered:\n'
               '\n'
               '$$z_j\\sim\\mathcal N(0,1),\\qquad\n'
               '\\eta_j=\\mu+\\sigma z_j.$$\n'
               '\n'
               'The second form produces the same conditional Normal distribution for '
               'eta_j. It changes sampler coordinates, not the scientific model.',
      'experiment': 'The left panel uses sigma and eta and narrows as sigma approaches '
                    'zero. The right uses sigma and z and keeps z near unit scale. Run the '
                    'PyMC comparison locally to inspect divergences and ESS.',
      'exercise_intro': 'This one line is a major practical tool in hierarchical Bayesian '
                        'models. Broadcasting lets it transform many group variables at '
                        'once.',
      'next': 'The final chapter connects this inference machinery to familiar '
              'neural-network training and then asks you to choose a method for a new '
              'bakery problem.',
      'prediction_answer': 'With weak group data, the non-centered form often samples '
                           'better. Standard variables z stay on a stable unit scale, and '
                           'the model constructs group values with mu + sigma*z. This '
                           'removes much of the funnel-shaped dependence between group '
                           'effects and the population scale.'},
     {'act': 'Chapter 6 · Connect the course to neural networks',
      'route': 'log likelihood + prior → loss + regularization',
      'story': 'The same map now reaches familiar ML practice. Standard network training '
               'finds one weight vector. A likelihood gives the data loss, and a Gaussian '
               'prior gives the L2 part of a MAP objective.',
      'mission': 'Build a MAP-style training loss',
      'exercise': '**Why this code exists.** This function makes the Bayesian '
                  'interpretation of L2 regularization explicit.\n'
                  '\n'
                  '**Your task.** Complete `map_loss(mse, weights, l2)`.\n'
                  '\n'
                  '**Inputs.** `mse` is the data-fit loss. `weights` is a NumPy array of '
                  'model weights. `l2` is the regularization strength.\n'
                  '\n'
                  '**Return.** `mse + 0.5 * l2 * sum(weights²)`.\n'
                  '\n'
                  '**Suggested steps.** Compute the squared-weight sum. Scale it by half '
                  'the L2 strength. Add it to the data loss.\n'
                  '\n'
                  '**Checkpoint.** With `l2=0`, the result must equal the data loss. A '
                  'positive L2 value must increase the loss for nonzero weights.',
      'prediction': 'As L2 strength increases, do fitted weight magnitudes usually become '
                    'larger or smaller? What prior assumption produces this penalty?',
      'quiz': 'Complete the mappings: cross-entropy to which probabilistic score, L2 to '
              'which prior view, and SGD or Adam to which kind of estimate?',
      'chapter_opening': 'Chapter 6 returns to standard ML. A tiny network predicts hourly '
                         'demand. Likelihood, priors, gradients, optimization, VI, and '
                         'posterior uncertainty now map onto familiar training terms.',
      'bridge': 'The earlier models had a few parameters. A neural network has many '
                'weights, but the relationship between model score, prior penalty, '
                'gradient, and learned object is unchanged.',
      'scenario': 'The bakery wants a nonlinear demand curve across the day. A small '
                  'network maps time of day to expected standardized sales per interval.',
      'goals': ['Map loss terms to likelihood and prior.',
                'Interpret L2 as MAP regularization.',
                'Distinguish one trained network from a posterior over weights.'],
      'model': 'Ordinary training solves\n'
               '\n'
               '$$w^*=\\arg\\min_w[-\\log p(D\\mid w)-\\log p(w)].$$\n'
               '\n'
               'For a zero-mean Gaussian prior, negative log p(w) contributes '
               '$\\lambda\\lVert w\\rVert^2/2$. Backpropagation computes gradients. SGD or '
               'Adam searches for one w. Bayesian learning instead targets p(w|D) or an '
               'approximation q(w).',
      'experiment': 'Increase L2 strength and watch both the fitted demand curve and '
                    'weight norm. The loss panel shows point optimization, not posterior '
                    'sampling.',
      'exercise_intro': 'The function separates data fit from prior penalty. This is the '
                        'probabilistic meaning of a common neural-network training '
                        'objective.',
      'next': 'One final bakery problem now requires you to select the learned object, '
              'inference method, and diagnostic plan yourself.',
      'prediction_answer': 'As L2 strength increases, fitted weight magnitudes usually '
                           'become smaller. L2 is the negative log contribution of a '
                           'zero-centered Gaussian prior on weights. Stronger '
                           'regularization means a narrower prior and a stronger '
                           'preference for weights near zero.'},
     {'act': 'Finale · Design inference before choosing software',
      'route': 'decision need → geometry → method → diagnostics',
      'story': 'A new sales day arrives. You must choose an inference method before you '
               'know the answer. Your choice must follow the required output, hidden '
               'structure, posterior geometry, and cost of a wrong decision.',
      'mission': 'Encode a small method-selection policy',
      'exercise': '**Why this code exists.** The function forces you to state a decision '
                  'policy instead of choosing an algorithm by habit.\n'
                  '\n'
                  '**Your task.** Complete `choose_method(needs_uncertainty, '
                  'latent_mixture, posterior_hard)`.\n'
                  '\n'
                  '**Inputs.** Each input is a boolean. `needs_uncertainty` says that a '
                  'distribution is required. `latent_mixture` says the problem has hidden '
                  'mixture assignments. `posterior_hard` says the posterior geometry is '
                  'difficult.\n'
                  '\n'
                  "**Return.** Use this policy: uncertainty plus hard geometry → `'NUTS'`; "
                  "latent mixture without uncertainty → `'EM'`; other uncertainty → "
                  "`'VI'`; otherwise → `'MAP'`.\n"
                  '\n'
                  '**Checkpoint.** The test checks the NUTS case and the EM case exactly.\n'
                  '\n'
                  '**After the code passes.** Explain one case where this small policy is '
                  'too simple and name the diagnostic you would use before deployment.',
      'prediction': 'For a latent customer mixture with a multimodal posterior and a need '
                    'for calibrated rare-event decisions, which method would you test '
                    'first? Name one diagnostic that could make you reject the result.',
      'quiz': 'For your selected scenario, state the learned object, target or objective, '
              'uncertainty status, role of gradients, and most important likely failure '
              'mode.',
      'chapter_opening': '',
      'bridge': 'You have optimized points, sampled posteriors, fitted variational '
                'distributions, alternated over hidden labels, and repaired hierarchical '
                'geometry.',
      'scenario': 'The bakery is adding a planning feature. Depending on the selected '
                  'scenario, it can need one fast forecast, customer segments, calibrated '
                  'coupon uncertainty, or a large approximate Bayesian model.',
      'goals': ['State the learned object before the method.',
                'Predict geometry and failure mode.',
                'Choose diagnostics that can reject a bad result.'],
      'model': 'Use the full map:\n'
               '\n'
               '$$decision\\rightarrow probabilistic\\ model\\rightarrow target\n'
               '\\rightarrow algorithm\\rightarrow output\\rightarrow diagnostics.$$\n'
               '\n'
               'MAP and EM return point-like fits. VI returns an approximate distribution. '
               'MCMC and NUTS return dependent posterior draws when sampling succeeds.',
      'experiment': 'Select each bakery scenario. Before you run it, write the five-part '
                    'answer yourself. Then compare your answer with the generated expert '
                    'table.',
      'exercise_intro': 'The policy is intentionally incomplete. Its purpose is to make '
                        'assumptions explicit so that you can identify where a real '
                        'decision needs more information.',
      'next': 'You have completed the course when you can predict how an inference method '
              'will behave on a new geometry and explain why its output is or is not '
              'sufficient for the decision.',
      'prediction_answer': 'Start with MCMC or NUTS because the decision needs calibrated '
                           'uncertainty and the target can be multimodal. Reject the '
                           'result if chains occupy different modes, R-hat remains high, '
                           'ESS is too small, or important divergences remain. Mode '
                           'occupancy is especially important here.'}]
    return (LAB_GUIDES,)


@app.cell(hide_code=True)
def _(
    COLORS,
    RunResult,
    approximate_ess,
    autocorrelation_curve,
    bivariate_gradient,
    bivariate_logpdf,
    finish_figure,
    gradient_ascent_demo,
    hamiltonian,
    metropolis_1d,
    metropolis_nd,
    norm,
    normal_logpdf,
    normal_mean_log_posterior,
    np,
    plt,
    style_axes,
):
    def run_experiment_early(index, values, seed):
        rng = np.random.default_rng(seed)

        if index == 0:
            return RunResult(
                None,
                "The laboratory keeps each system small. Change one control at a time, and write an explanation before you mark the lab complete.",
                [
                    {"stage": "predict", "question": "What do you expect?"},
                    {"stage": "simulate", "question": "What did the system do?"},
                    {"stage": "inspect", "question": "Where is the mismatch?"},
                    {"stage": "code", "question": "Can you build the mechanism?"},
                    {"stage": "explain", "question": "Can you state the causal story?"},
                ],
            )

        if index == 1:
            mean, sd, interval = values["mean"], values["sd"], values["interval"]
            x = np.linspace(mean - 5 * sd, mean + 5 * sd, 600)
            density = np.exp(normal_logpdf(x, mean, sd))
            lo, hi = mean - interval / 2, mean + interval / 2
            mass = norm.cdf(hi, mean, sd) - norm.cdf(lo, mean, sd)
            fig, ax = plt.subplots(figsize=(8, 3.8))
            ax.plot(x, density, color=COLORS["posterior"], lw=3)
            mask = (x >= lo) & (x <= hi)
            ax.fill_between(x[mask], density[mask], color=COLORS["posterior"], alpha=0.3)
            ax.axvline(mean, color=COLORS["accent"], ls="--", label="mean = mode")
            style_axes(ax, "Density height and interval mass", "x", "density")
            ax.legend(frameon=False)
            return RunResult(finish_figure(fig), f"Peak height = {density.max():.3f}. Shaded probability mass = {mass:.3f}. Total area stays 1.")

        if index == 2:
            prior_mu = values["prior mean"]
            prior_sd = values["prior width"]
            n = int(values["observations"])
            new_day = float(values["new sales day"])
            full_data = np.array(
                [48, 52, 57, 61, 68, 72, 65, 76, 69, 80, 74, 62, 78, 84, 71, 67, 75, 81, 77, 64],
                dtype=float,
            )
            existing_data = full_data[:n]
            updated_data = np.append(existing_data, new_day)
            sigma = 8.0
            grid = np.linspace(20, 115, 760)

            log_like_before = np.array(
                [np.sum(normal_logpdf(existing_data, mu, sigma)) for mu in grid]
            )
            log_like_after = np.array(
                [np.sum(normal_logpdf(updated_data, mu, sigma)) for mu in grid]
            )
            likelihood_before = np.exp(log_like_before - log_like_before.max())
            prior = np.exp(normal_logpdf(grid, prior_mu, prior_sd))
            log_prior = normal_logpdf(grid, prior_mu, prior_sd)
            log_post_before = log_like_before + log_prior
            log_post_after = log_like_after + log_prior
            posterior_before = np.exp(
                log_post_before - logsumexp_trap(log_post_before, grid)
            )
            posterior_after = np.exp(
                log_post_after - logsumexp_trap(log_post_after, grid)
            )

            before_mean = np.trapezoid(grid * posterior_before, grid)
            after_mean = np.trapezoid(grid * posterior_after, grid)
            after_sd = np.sqrt(
                np.trapezoid((grid - after_mean) ** 2 * posterior_after, grid)
            )
            shift = after_mean - before_mean

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
            axes[0].plot(
                grid,
                prior / prior.max(),
                color=COLORS["prior"],
                lw=2,
                label="prior, scaled",
            )
            axes[0].plot(
                grid,
                likelihood_before,
                color=COLORS["likelihood"],
                lw=2,
                label="existing-data likelihood, scaled",
            )
            axes[0].plot(
                grid,
                posterior_before / posterior_before.max(),
                color=COLORS["sample"],
                lw=3,
                label="posterior before new day",
            )
            axes[0].scatter(
                existing_data,
                np.full_like(existing_data, -0.04),
                marker="|",
                s=90,
                color=COLORS["ink"],
                label="existing sales days",
            )
            style_axes(
                axes[0],
                f"Build belief from {n} existing sales day(s)",
                "possible average daily loaves, μ",
                "relative height",
            )
            axes[0].legend(frameon=False, fontsize=8, loc="upper left")

            axes[1].plot(
                grid,
                posterior_before / posterior_before.max(),
                color=COLORS["sample"],
                lw=2,
                ls="--",
                label=f"before: mean {before_mean:.1f}",
            )
            axes[1].plot(
                grid,
                posterior_after / posterior_after.max(),
                color=COLORS["posterior"],
                lw=3,
                label=f"after: mean {after_mean:.1f}",
            )
            axes[1].axvline(
                new_day,
                color=COLORS["accent"],
                lw=2,
                ls=":",
                label=f"new day: {new_day:.0f} loaves",
            )
            axes[1].scatter(
                [new_day],
                [-0.04],
                marker="*",
                s=130,
                color=COLORS["accent"],
                zorder=5,
            )
            style_axes(
                axes[1],
                "Measure the effect of one new sales day",
                "possible average daily loaves, μ",
                "relative height",
            )
            axes[1].legend(frameon=False, fontsize=8, loc="upper left")
            return RunResult(
                finish_figure(fig),
                f"Before the new day, posterior mean = {before_mean:.1f} loaves. "
                f"After observing {new_day:.0f} loaves, it is {after_mean:.1f}: "
                f"a shift of {shift:+.1f} loaves. Posterior SD is {after_sd:.1f}. "
                f"Keep the data fixed and compare this shift at small and large prior width.",
            )

        if index == 3:
            data = np.array([0.2, 0.8, 1.1, 1.7, 2.0])
            logp = lambda mu: normal_mean_log_posterior(mu, data, 1.0, 0.0, 2.0)
            path = gradient_ascent_demo(logp, values["start"], values["learning rate"], int(values["steps"]))
            grid = np.linspace(-5, 5, 600)
            density = np.exp(np.array([logp(x) for x in grid]) - max(logp(x) for x in grid))
            path_y = np.exp(np.array([logp(x) for x in path]) - max(logp(x) for x in grid))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(grid, density, color=COLORS["posterior"], lw=3)
            ax.plot(path, path_y, "o-", color=COLORS["accent"], ms=4, label="ascent path")
            style_axes(ax, "Gradient ascent on the log posterior", "μ", "relative posterior")
            ax.legend(frameon=False)
            status = "stable" if np.all(np.isfinite(path)) and abs(path[-1]) < 20 else "unstable"
            return RunResult(finish_figure(fig), f"Final point = {path[-1]:.3f}; path is {status}. The output is one point, not a distribution.")

        if index == 4:
            wide_sd = values["wide sd"]
            grid = np.linspace(-8, 8, 700)
            narrow = np.exp(normal_logpdf(grid, 0.0, 0.35))
            wide = np.exp(normal_logpdf(grid, 0.0, wide_sd))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(grid, narrow, color=COLORS["accent"], lw=2.5, label="narrow posterior")
            ax.plot(grid, wide, color=COLORS["posterior"], lw=2.5, label="wide posterior")
            ax.axvline(0, color=COLORS["ink"], ls="--", label="same MAP")
            ax.axvspan(-0.69, 0.69, color=COLORS["accent"], alpha=0.08)
            ax.axvspan(-1.96 * wide_sd, 1.96 * wide_sd, color=COLORS["posterior"], alpha=0.08)
            style_axes(ax, "Equal mode, unequal uncertainty", "θ", "density")
            ax.legend(frameon=False)
            return RunResult(finish_figure(fig), f"Both MAP values are 0. Their approximate 95% widths are 1.37 and {3.92*wide_sd:.2f}.")

        if index == 5:
            proposal_sd = values["proposal sd"]
            draws = int(values["draws"])
            logp = lambda theta: -0.5 * theta**2
            samples, accepted = metropolis_1d(logp, -4.0, proposal_sd, draws, rng)
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            grid = np.linspace(-5, 5, 500)
            axes[0].plot(grid, np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi), color=COLORS["posterior"], lw=2)
            axes[0].hist(samples, bins=35, density=True, alpha=0.45, color=COLORS["sample"])
            axes[1].plot(samples[: min(500, draws)], lw=0.8, color=COLORS["sample"])
            move = np.abs(np.diff(samples))
            axes[2].scatter(samples[:-1][:500], samples[1:][:500], s=8, alpha=0.35, color=COLORS["sample"])
            for ax, title in zip(axes, ("Target and samples", "Trace", "Consecutive states")):
                style_axes(ax, title)
            return RunResult(finish_figure(fig), f"Acceptance = {accepted.mean():.1%}; mean move = {move.mean():.3f}; sample mean = {samples.mean():.3f}.")

        if index == 6:
            chains = []
            labels = []
            for proposal in (values["small proposal"], values["large proposal"]):
                chain, accepted = metropolis_1d(lambda x: -0.5 * x**2, -3.0, proposal, 5000, rng)
                chains.append((chain, accepted.mean()))
                labels.append(f"proposal {proposal:.2f}")
            fig, axes = plt.subplots(2, 2, figsize=(11, 6))
            rows = []
            for row, ((chain, acceptance), label) in enumerate(zip(chains, labels)):
                axes[row, 0].plot(chain[:800], lw=0.7, color=COLORS["sample"])
                ac = autocorrelation_curve(chain, 80)
                axes[row, 1].stem(range(len(ac)), ac, linefmt=COLORS["posterior"], markerfmt=" ", basefmt=" ")
                style_axes(axes[row, 0], f"{label}: trace")
                style_axes(axes[row, 1], f"{label}: autocorrelation", "lag", "correlation")
                rows.append({"proposal": label, "acceptance": f"{acceptance:.1%}", "ESS (rough)": f"{approximate_ess(chain):.0f}"})
            return RunResult(finish_figure(fig), "A long trace can still contain little new information when lag correlation decays slowly.", rows)

        if index == 7:
            rho = values["correlation"]
            proposal_sd = values["proposal sd"]
            logp = lambda theta: bivariate_logpdf(theta, rho)
            samples, accepted = metropolis_nd(logp, np.array([-2.5, 2.5]), proposal_sd, 3000, rng)
            grid = np.linspace(-3.5, 3.5, 180)
            xx, yy = np.meshgrid(grid, grid)
            zz = np.exp(np.vectorize(lambda x, y: logp(np.array([x, y])))(xx, yy))
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
            for ax in axes:
                ax.contour(xx, yy, zz, levels=8, colors=COLORS["posterior"], alpha=0.7)
            axes[0].plot(samples[:400, 0], samples[:400, 1], lw=0.7, color=COLORS["sample"])
            axes[1].scatter(samples[500:, 0], samples[500:, 1], s=5, alpha=0.2, color=COLORS["sample"])
            style_axes(axes[0], "First 400 random-walk states", "θ₁", "θ₂")
            style_axes(axes[1], "Retained sample cloud", "θ₁", "θ₂")
            return RunResult(finish_figure(fig), f"Acceptance = {accepted.mean():.1%}. Correlation {rho:.2f} makes the useful direction different from the proposal axes.")

        if index == 8:
            rho = values["correlation"]
            grid = np.linspace(-3, 3, 100)
            xx, yy = np.meshgrid(grid, grid)
            zz = np.exp(np.vectorize(lambda x, y: bivariate_logpdf(np.array([x, y]), rho))(xx, yy))
            points = np.linspace(-2.5, 2.5, 11)
            px, py = np.meshgrid(points, points)
            grads = np.asarray([bivariate_gradient(np.array([x, y]), rho) for x, y in zip(px.ravel(), py.ravel())])
            lengths = np.linalg.norm(grads, axis=1, keepdims=True)
            unit = grads / np.maximum(lengths, 1e-9)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.contour(xx, yy, zz, levels=9, colors=COLORS["posterior"], alpha=0.65)
            ax.quiver(px, py, unit[:, 0].reshape(px.shape), unit[:, 1].reshape(py.shape), color=COLORS["accent"], alpha=0.8)
            style_axes(ax, "∇ log p(θ) as a force field", "θ₁", "θ₂")
            return RunResult(finish_figure(fig), "Each arrow points toward a higher log density. Its raw length is slope magnitude; arrows are normalized here so direction is clear.")

        if index == 9:
            initial_momentum = np.array([values["momentum x"], values["momentum y"]], dtype=float)
            steps = int(values["steps"])
            rho = 0.85
            step_size = 0.12
            start = np.array([-2.0, -1.6])
            logp = lambda q: bivariate_logpdf(q, rho)
            grad = lambda q: bivariate_gradient(q, rho)

            position = start.copy()
            momentum = initial_momentum.copy()
            positions = [position.copy()]
            momenta = [momentum.copy()]
            for _ in range(steps):
                momentum = momentum + 0.5 * step_size * grad(position)
                position = position + step_size * momentum
                momentum = momentum + 0.5 * step_size * grad(position)
                positions.append(position.copy())
                momenta.append(momentum.copy())
            positions = np.asarray(positions)
            momenta = np.asarray(momenta)

            time = np.arange(steps + 1) * step_size
            force_free = start + time[:, None] * initial_momentum
            potential = np.array([-logp(q) for q in positions])
            kinetic = 0.5 * np.sum(momenta**2, axis=1)
            total = potential + kinetic
            initial_force = grad(start)
            first_half_momentum = initial_momentum + 0.5 * step_size * initial_force

            grid = np.linspace(-4, 4, 150)
            xx, yy = np.meshgrid(grid, grid)
            zz = np.exp(np.vectorize(lambda x, y: logp(np.array([x, y])))(xx, yy))
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

            axes[0].contour(xx, yy, zz, levels=9, colors=COLORS["posterior"], alpha=0.65)
            axes[0].plot(
                force_free[:, 0],
                force_free[:, 1],
                "--",
                color="#94A3B8",
                lw=2,
                label="same r₀, posterior force off",
            )
            axes[0].plot(
                positions[:, 0],
                positions[:, 1],
                "o-",
                ms=2,
                lw=1.5,
                color=COLORS["sample"],
                label="HMC: posterior force on",
            )
            axes[0].scatter(*start, s=55, color=COLORS["accent"], zorder=4, label="start θ₀")
            axes[0].scatter(0, 0, s=90, marker="*", color=COLORS["ink"], zorder=4, label="mode")
            force_direction = initial_force / max(np.linalg.norm(initial_force), 1e-12)
            axes[0].quiver(
                start[0],
                start[1],
                force_direction[0],
                force_direction[1],
                color="#DC2626",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.008,
                label="initial force direction",
            )
            if np.linalg.norm(initial_momentum) > 1e-12:
                axes[0].quiver(
                    start[0],
                    start[1],
                    initial_momentum[0],
                    initial_momentum[1],
                    color=COLORS["accent"],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.008,
                    label="initial momentum r₀",
                )
            else:
                axes[0].text(
                    start[0] + 0.1,
                    start[1] - 0.35,
                    "initial r₀ = (0, 0)",
                    color=COLORS["accent"],
                    fontsize=9,
                )
            style_axes(axes[0], "What creates the path?", "θ₁", "θ₂")
            axes[0].legend(frameon=False, fontsize=8, loc="upper left")

            axes[1].axhline(0, color="#CBD5E1", lw=1)
            axes[1].plot(momenta[:, 0], color=COLORS["accent"], lw=1.8, label="r₁: horizontal")
            axes[1].plot(momenta[:, 1], color=COLORS["sample"], lw=1.8, label="r₂: vertical")
            axes[1].plot(
                np.linalg.norm(momenta, axis=1),
                color=COLORS["ink"],
                lw=2,
                ls="--",
                label="||r||: momentum length",
            )
            style_axes(axes[1], "The gradient changes momentum", "step", "momentum")
            axes[1].legend(frameon=False, fontsize=8)

            axes[2].plot(potential, color=COLORS["posterior"], lw=1.8, label="U: potential")
            axes[2].plot(kinetic, color=COLORS["sample"], lw=1.8, label="K: kinetic")
            axes[2].plot(total, color=COLORS["ink"], lw=2, ls="--", label="H = U + K")
            style_axes(axes[2], "Energy changes form", "step", "energy")
            axes[2].legend(frameon=False, fontsize=8)

            progress = (positions - start) @ (-start) / float(start @ start)
            crossed = np.flatnonzero(progress > 1.0)
            crossing_text = (
                f"It passed the mode's center line after step {int(crossed[0])}."
                if len(crossed)
                else "It did not yet pass the mode's center line in the selected number of steps."
            )
            if np.linalg.norm(initial_momentum) < 1e-12:
                cause_text = (
                    "Initial momentum was zero. The nonzero posterior gradient changed it to "
                    f"approximately ({first_half_momentum[0]:.2f}, {first_half_momentum[1]:.2f}) "
                    "in the first half-update, so the next position update moved the particle. "
                    "The force-free gray reference stayed at the start."
                )
            else:
                cause_text = (
                    f"Initial momentum was ({initial_momentum[0]:.2f}, {initial_momentum[1]:.2f}). "
                    "The posterior gradient then changed both components during the trajectory."
                )
            return RunResult(
                finish_figure(fig),
                f"{cause_text} {crossing_text} Total-energy drift was "
                f"{float(np.max(np.abs(total - total[0]))):.3g}.",
            )

        if index == 10:
            step_size = float(values["step size"])
            steps = int(values["steps"])
            start = np.array([-1.5, -1.0])
            initial_momentum = np.array([1.2, 0.4])
            rho = 0.8
            logp = lambda q: bivariate_logpdf(q, rho)
            grad = lambda q: bivariate_gradient(q, rho)

            def integrate(method, q0, p0, epsilon, count):
                q = np.asarray(q0, dtype=float).copy()
                p = np.asarray(p0, dtype=float).copy()
                positions = [q.copy()]
                momenta = [p.copy()]
                with np.errstate(over="ignore", invalid="ignore"):
                    for _ in range(count):
                        if method == "leapfrog":
                            p = p + 0.5 * epsilon * grad(q)
                            q = q + epsilon * p
                            p = p + 0.5 * epsilon * grad(q)
                        else:
                            p = p + epsilon * grad(q)
                            q = q + epsilon * p
                        positions.append(q.copy())
                        momenta.append(p.copy())
                        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(p))):
                            break
                return np.asarray(positions), np.asarray(momenta)

            def energy_series(positions, momenta):
                with np.errstate(over="ignore", invalid="ignore"):
                    return np.array(
                        [hamiltonian(logp, q, p) for q, p in zip(positions, momenta)]
                    )

            def acceptance_from_delta(delta_h):
                if not np.isfinite(delta_h):
                    return 0.0
                if delta_h <= 0:
                    return 1.0
                return float(np.exp(-min(float(delta_h), 745.0)))

            leap_q, leap_p = integrate(
                "leapfrog", start, initial_momentum, step_size, steps
            )
            side_q, side_p = integrate(
                "one-sided", start, initial_momentum, step_size, steps
            )
            leap_h = energy_series(leap_q, leap_p)
            side_h = energy_series(side_q, side_p)
            old_h = hamiltonian(logp, start, initial_momentum)
            leap_delta = float(leap_h[-1] - old_h) if np.isfinite(leap_h[-1]) else np.inf
            side_delta = float(side_h[-1] - old_h) if np.isfinite(side_h[-1]) else np.inf
            selected_alpha = acceptance_from_delta(leap_delta)

            if np.all(np.isfinite(leap_q[-1])) and np.all(np.isfinite(leap_p[-1])):
                leap_back_q, leap_back_p = integrate(
                    "leapfrog", leap_q[-1], leap_p[-1], -step_size, steps
                )
                leap_return_error = float(
                    np.linalg.norm(
                        np.concatenate(
                            [leap_back_q[-1] - start, leap_back_p[-1] - initial_momentum]
                        )
                    )
                )
            else:
                leap_return_error = np.inf

            if np.all(np.isfinite(side_q[-1])) and np.all(np.isfinite(side_p[-1])):
                side_back_q, side_back_p = integrate(
                    "one-sided", side_q[-1], side_p[-1], -step_size, steps
                )
                side_return_error = float(
                    np.linalg.norm(
                        np.concatenate(
                            [side_back_q[-1] - start, side_back_p[-1] - initial_momentum]
                        )
                    )
                )
            else:
                side_return_error = np.inf

            step_grid = np.linspace(0.02, 1.2, 48)
            momentum_draws = np.random.default_rng(seed).normal(size=(48, 2))
            average_acceptance = []
            for epsilon in step_grid:
                probabilities = []
                for sampled_momentum in momentum_draws:
                    grid_q, grid_p = integrate(
                        "leapfrog", start, sampled_momentum, epsilon, steps
                    )
                    if np.all(np.isfinite(grid_q[-1])) and np.all(np.isfinite(grid_p[-1])):
                        new_h = hamiltonian(logp, grid_q[-1], grid_p[-1])
                        grid_old_h = hamiltonian(logp, start, sampled_momentum)
                        probabilities.append(acceptance_from_delta(new_h - grid_old_h))
                    else:
                        probabilities.append(0.0)
                average_acceptance.append(float(np.mean(probabilities)))
            average_acceptance = np.asarray(average_acceptance)

            grid = np.linspace(-4, 4, 150)
            xx, yy = np.meshgrid(grid, grid)
            zz = np.exp(np.vectorize(lambda x, y: logp(np.array([x, y])))(xx, yy))
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

            axes[0].contour(xx, yy, zz, levels=9, colors=COLORS["posterior"], alpha=0.65)
            axes[0].plot(
                side_q[:, 0],
                side_q[:, 1],
                "o-",
                ms=2,
                lw=1.2,
                color=COLORS["likelihood"],
                label="one-sided: full p, full q",
            )
            axes[0].plot(
                leap_q[:, 0],
                leap_q[:, 1],
                "o-",
                ms=2,
                lw=1.5,
                color=COLORS["sample"],
                label="leapfrog: ½ p, q, ½ p",
            )
            axes[0].scatter(*start, s=60, color=COLORS["accent"], zorder=5, label="start")
            axes[0].set_xlim(-4, 4)
            axes[0].set_ylim(-4, 4)
            style_axes(axes[0], "Same force, different update order", "θ₁", "θ₂")
            axes[0].legend(frameon=False, fontsize=8)

            leap_error = np.clip(leap_h - old_h, -1e6, 1e6)
            side_error = np.clip(side_h - old_h, -1e6, 1e6)
            axes[1].axhline(0, color="#CBD5E1", lw=1)
            axes[1].plot(
                side_error,
                color=COLORS["likelihood"],
                lw=1.5,
                label="one-sided H - H₀",
            )
            axes[1].plot(
                leap_error,
                color=COLORS["sample"],
                lw=1.8,
                label="leapfrog H - H₀",
            )
            axes[1].set_yscale("symlog", linthresh=1e-3)
            style_axes(axes[1], "Numerical energy error", "trajectory step", "signed ΔH")
            axes[1].legend(frameon=False, fontsize=8)

            axes[2].plot(
                step_grid,
                average_acceptance,
                color=COLORS["accent"],
                lw=2.2,
                label="mean over 48 momentum draws",
            )
            axes[2].axvline(step_size, color=COLORS["ink"], ls="--", label="selected ε")
            axes[2].scatter(
                [step_size],
                [selected_alpha],
                s=55,
                color=COLORS["sample"],
                zorder=5,
                label="selected path α",
            )
            axes[2].set_ylim(-0.03, 1.05)
            style_axes(
                axes[2],
                "Metropolis acceptance after leapfrog",
                "step size ε",
                "acceptance probability α",
            )
            axes[2].legend(frameon=False, fontsize=8)

            def show_number(value):
                if not np.isfinite(value):
                    return "overflow"
                return f"{value:.3g}"

            comparison = [
                {
                    "method": "symmetric leapfrog",
                    "update order": "½ momentum → position → ½ momentum",
                    "final ΔH": show_number(leap_delta),
                    "backward return error": show_number(leap_return_error),
                    "standard HMC correction": f"valid; α = {selected_alpha:.3f}",
                },
                {
                    "method": "one-sided comparison",
                    "update order": "full momentum → position",
                    "final ΔH": show_number(side_delta),
                    "backward return error": show_number(side_return_error),
                    "standard HMC correction": "not valid: map is not reversible",
                },
            ]
            return RunResult(
                finish_figure(fig),
                "The two half momentum updates place the position move inside a symmetric sequence. "
                f"At ε={step_size:.2f}, leapfrog's backward return error is "
                f"{show_number(leap_return_error)}, versus {show_number(side_return_error)} for the "
                "one-sided update. For the selected leapfrog path, "
                f"ΔH={show_number(leap_delta)}, so α=min(1, exp(-ΔH))={selected_alpha:.3f}. "
                "HMC draws u uniformly between 0 and 1: if u<α it accepts the proposed endpoint; "
                "otherwise the chain repeats its old position. This corrects remaining leapfrog "
                "energy error in the sample distribution. It does not make the non-reversible "
                "one-sided comparison into valid HMC.",
                comparison,
            )

        if index == 11:
            transitions = int(values["transitions"])
            leapfrog_steps = int(values["leapfrog steps"])
            step_size = 0.14
            rho = 0.8
            start = np.array([-2.2, -1.7])
            logp = lambda q: bivariate_logpdf(q, rho)
            grad = lambda q: bivariate_gradient(q, rho)
            rng = np.random.default_rng(seed)

            def integrate_path(q0, p0):
                q = np.asarray(q0, dtype=float).copy()
                p = np.asarray(p0, dtype=float).copy()
                path = [q.copy()]
                for _ in range(leapfrog_steps):
                    p = p + 0.5 * step_size * grad(q)
                    q = q + step_size * p
                    p = p + 0.5 * step_size * grad(q)
                    path.append(q.copy())
                return q, p, np.asarray(path)

            def transition(q0, p0, uniform_draw):
                proposed_q, proposed_p, path = integrate_path(q0, p0)
                proposed_p = -proposed_p
                old_h = hamiltonian(logp, q0, p0)
                new_h = hamiltonian(logp, proposed_q, proposed_p)
                log_accept = min(0.0, old_h - new_h)
                accepted = bool(np.log(uniform_draw) < log_accept)
                next_q = proposed_q if accepted else q0.copy()
                return next_q, accepted, path, proposed_q, log_accept

            chain = [start.copy()]
            accepted_flags = []
            first_transition = None
            for transition_index in range(transitions):
                current = chain[-1]
                fresh_momentum = rng.normal(size=2)
                outcome = transition(current, fresh_momentum, rng.uniform())
                next_q, accepted, path, proposed_q, log_accept = outcome
                if transition_index == 0:
                    first_transition = {
                        "start": current.copy(),
                        "path": path,
                        "proposal": proposed_q.copy(),
                        "accepted": accepted,
                        "log_accept": log_accept,
                    }
                chain.append(next_q.copy())
                accepted_flags.append(accepted)
            chain = np.asarray(chain)
            accepted_flags = np.asarray(accepted_flags, dtype=bool)

            orbit_q = start.copy()
            orbit_p = np.array([1.2, 0.4])
            orbit = [orbit_q.copy()]
            orbit_momenta = [orbit_p.copy()]
            for _ in range(transitions):
                orbit_q, orbit_p, _ = integrate_path(orbit_q, orbit_p)
                orbit.append(orbit_q.copy())
                orbit_momenta.append(orbit_p.copy())
            orbit = np.asarray(orbit)
            orbit_momenta = np.asarray(orbit_momenta)
            orbit_energies = np.asarray(
                [
                    hamiltonian(logp, q, p)
                    for q, p in zip(orbit, orbit_momenta, strict=True)
                ]
            )
            orbit_delta_h = orbit_energies - orbit_energies[0]

            grid = np.linspace(-4, 4, 160)
            xx, yy = np.meshgrid(grid, grid)
            zz = np.exp(np.vectorize(lambda x, y: logp(np.array([x, y])))(xx, yy))
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

            first_path = first_transition["path"]
            axes[0].contour(xx, yy, zz, levels=9, colors=COLORS["posterior"], alpha=0.6)
            axes[0].plot(
                first_path[:, 0],
                first_path[:, 1],
                "o-",
                ms=3,
                color=COLORS["sample"],
                label="deterministic leapfrog path",
            )
            axes[0].scatter(
                *first_transition["start"],
                s=65,
                color=COLORS["accent"],
                label="stored start q₀",
                zorder=5,
            )
            proposal_color = COLORS["sample"] if first_transition["accepted"] else "#DC2626"
            decision_label = "proposal accepted" if first_transition["accepted"] else "proposal rejected"
            axes[0].scatter(
                *first_transition["proposal"],
                s=75,
                marker="*",
                color=proposal_color,
                label=decision_label,
                zorder=5,
            )
            style_axes(axes[0], "One trajectory proposes one endpoint", "θ₁", "θ₂")
            axes[0].legend(frameon=False, fontsize=8)

            burn = min(max(10, transitions // 10), max(10, transitions - 1))
            stored = chain[burn:]
            axes[1].contour(xx, yy, zz, levels=9, colors=COLORS["posterior"], alpha=0.55)
            axes[1].scatter(
                stored[:, 0],
                stored[:, 1],
                s=13,
                alpha=0.5,
                color=COLORS["sample"],
                label="stored positions after warmup view",
            )
            rejected_positions = chain[1:][~accepted_flags]
            if len(rejected_positions):
                axes[1].scatter(
                    rejected_positions[:, 0],
                    rejected_positions[:, 1],
                    s=28,
                    marker="x",
                    color="#DC2626",
                    label="rejection: old q stored again",
                )
            style_axes(axes[1], "Fresh momentum: fixed-length HMC chain", "θ₁", "θ₂")
            axes[1].legend(frameon=False, fontsize=8)

            axes[2].contour(xx, yy, zz, levels=9, colors=COLORS["posterior"], alpha=0.55)
            axes[2].plot(
                orbit[:, 0],
                orbit[:, 1],
                lw=0.8,
                alpha=0.35,
                color=COLORS["likelihood"],
            )
            time_points = axes[2].scatter(
                orbit[:, 0],
                orbit[:, 1],
                c=np.arange(len(orbit)),
                cmap="plasma",
                s=11,
                alpha=0.78,
                label="one path; color advances with time",
            )
            fig.colorbar(time_points, ax=axes[2], fraction=0.046, pad=0.03, label="time")
            axes[2].text(
                0.03,
                0.97,
                "Broad-looking q projection\n≠ posterior exploration",
                transform=axes[2].transAxes,
                va="top",
                fontsize=8,
                color=COLORS["ink"],
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "#CBD5E1"},
            )
            energy_inset = axes[2].inset_axes([0.57, 0.06, 0.38, 0.27])
            energy_inset.plot(orbit_delta_h, color=COLORS["likelihood"], lw=1)
            energy_inset.axhline(0.0, color="#64748B", lw=0.8, ls="--")
            energy_inset.set_title("numerical ΔH", fontsize=8)
            energy_inset.set_xlabel("time", fontsize=7)
            energy_inset.tick_params(labelsize=6)
            style_axes(axes[2], "No refresh: apparent exploration trap", "θ₁", "θ₂")
            axes[2].legend(frameon=False, fontsize=8, loc="lower left")

            acceptance_rate = float(np.mean(accepted_flags))
            stored_mean = np.mean(stored, axis=0)
            object_table = [
                {
                    "object": "leapfrog step",
                    "returns": "one new (q, p)",
                    "random after inputs are fixed?": "no",
                    "stored as a posterior draw?": "no",
                },
                {
                    "object": "trajectory",
                    "returns": "one proposed endpoint after L steps",
                    "random after q and p are fixed?": "no",
                    "stored as a posterior draw?": "no",
                },
                {
                    "object": "HMC transition",
                    "returns": "accepted proposal or repeated old q",
                    "random after q and p are fixed?": "Uniform accept draw",
                    "stored as a posterior draw?": "one position",
                },
                {
                    "object": "HMC chain",
                    "returns": "ordered positions from many transitions",
                    "random after q and p are fixed?": "fresh p and u each transition",
                    "stored as a posterior draw?": "all stored q values",
                },
            ]
            return RunResult(
                finish_figure(fig),
                f"The first path is deterministic after its start and momentum are fixed. Across "
                f"{transitions} proper transitions, fresh momentum and the Uniform accept draw produced "
                f"an acceptance rate of {acceptance_rate:.1%}; the displayed post-warmup mean is "
                f"({stored_mean[0]:.2f}, {stored_mean[1]:.2f}). A rejection stores the old position "
                "again. The orange no-refresh path can look broad in the q-only plot because momentum "
                "is hidden and finite leapfrog steps add phase and Hamiltonian error. Its exact H changed "
                f"by only {np.ptp(orbit_energies):.3g} across the displayed path, so this apparent filling "
                "does not show random movement across joint energy shells. Call it one approximate energy "
                "orbit, not one posterior mode. The stored positions from complete HMC transitions, not "
                "the intermediate or no-refresh path points, are the posterior samples.",
                object_table,
            )

        raise ValueError(f"Early experiment does not support lab {index}")

    def logsumexp_trap(log_values, grid):
        peak = np.max(log_values)
        return peak + np.log(np.trapezoid(np.exp(log_values - peak), grid))

    return (run_experiment_early,)


@app.cell(hide_code=True)
def _(
    COLORS,
    RunResult,
    approximate_ess,
    bivariate_gradient,
    bivariate_logpdf,
    finish_figure,
    gaussian_mixture_em,
    importlib,
    leapfrog,
    metropolis_1d,
    np,
    plt,
    rhat_basic,
    shrinkage_estimates,
    style_axes,
    sys,
    u_turn,
):
    def _import_optional(name):
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            raise RuntimeError(
                f"The optional package '{name}' is not installed in this "
                "environment. Run 'uv sync' in the project directory, then "
                "restart the notebook."
            ) from exc


    def run_experiment_late(index, values, seed):
        rng = np.random.default_rng(seed)
        is_wasm = sys.platform == "emscripten"

        if index == 12:
            if values["engine"] == "PyMC NUTS":
                if is_wasm:
                    return RunResult(None, "PyMC is not available in this browser runtime. Select manual geometry here, or run this lab locally for live NUTS diagnostics.")
                pm = _import_optional("pymc")
                with pm.Model() as model:
                    mu = pm.Normal("mu", 0.0, 2.0)
                    pm.Normal("obs", mu, 1.0, observed=np.array([0.2, 0.8, 1.1, 1.7, 2.0]))
                    idata = pm.sample(draws=350, tune=350, chains=2, cores=1, random_seed=seed, progressbar=False)
                chains = np.asarray(idata.posterior["mu"])
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
                for chain in chains:
                    axes[0].plot(chain, lw=0.7, alpha=0.8)
                    axes[1].hist(chain, bins=28, alpha=0.35, density=True)
                style_axes(axes[0], "PyMC NUTS chains", "draw", "μ")
                style_axes(axes[1], "Posterior draws", "μ", "density")
                divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
                return RunResult(finish_figure(fig), f"PyMC ran warmup plus 700 retained draws. Divergences = {divergences}; basic R-hat = {rhat_basic(chains):.3f}.")
            rho = 0.82
            start = np.array([-2.3, -1.8])
            momentum = np.array([1.7, 0.35])
            step_size = values["step size"]
            max_steps = int(values["max steps"])
            _, _, path = leapfrog(start, momentum, step_size, max_steps, lambda q: bivariate_gradient(q, rho))
            flags = [u_turn(start, path[i], (path[i] - path[i - 1]) / step_size) for i in range(1, len(path))]
            stop = next((i + 1 for i, flag in enumerate(flags) if flag), len(path) - 1)
            displacement = path[stop] - start
            current_momentum = (path[stop] - path[max(0, stop - 1)]) / step_size
            grid = np.linspace(-4, 4, 160)
            xx, yy = np.meshgrid(grid, grid)
            zz = np.exp(np.vectorize(lambda x, y: bivariate_logpdf(np.array([x, y]), rho))(xx, yy))
            fig, ax = plt.subplots(figsize=(7, 5.2))
            ax.contour(xx, yy, zz, levels=9, colors="#94A3B8")
            ax.plot(path[: stop + 1, 0], path[: stop + 1, 1], "o-", ms=2.5, color=COLORS["sample"])
            ax.quiver(start[0], start[1], displacement[0], displacement[1], color=COLORS["posterior"], angles="xy", scale_units="xy", scale=1, label="displacement")
            ax.quiver(path[stop, 0], path[stop, 1], current_momentum[0], current_momentum[1], color=COLORS["accent"], angles="xy", scale_units="xy", scale=1, label="momentum")
            style_axes(ax, "Stop when motion starts to turn back", "θ₁", "θ₂")
            ax.legend(frameon=False)
            dot = float(np.dot(displacement, current_momentum))
            return RunResult(finish_figure(fig), f"First detected U-turn: step {stop}; displacement · momentum = {dot:.3f}. Production NUTS grows a tree in both directions.")

        if index == 13:
            separation = values["broken separation"]
            chains = []
            for chain_id, start in enumerate((-5.0, -2.0, 2.0, 5.0)):
                target_mean = separation * (-1 if chain_id < 2 else 1)
                chain, _ = metropolis_1d(lambda x, m=target_mean: -0.5 * (x - m) ** 2, start, 0.8, 1600, rng)
                chains.append(chain[400:])
            chains = np.asarray(chains)
            fig, axes = plt.subplots(2, 2, figsize=(11, 6))
            for chain_id, chain in enumerate(chains):
                axes[0, 0].plot(chain[:500], lw=0.65, label=f"chain {chain_id+1}")
                axes[0, 1].hist(chain, bins=35, density=True, histtype="step", lw=1.3)
                axes[1, 0].plot(np.cumsum(chain) / np.arange(1, len(chain) + 1), lw=0.9)
            axes[1, 1].bar(range(1, 5), [approximate_ess(chain) for chain in chains], color=COLORS["sample"], alpha=0.75)
            for ax, title in zip(axes.ravel(), ("Trace", "Chain histograms", "Cumulative means", "Approximate ESS")):
                style_axes(ax, title)
            axes[0, 0].legend(frameon=False, ncol=2)
            rhat = rhat_basic(chains)
            mcse = chains.reshape(-1).std() / np.sqrt(sum(approximate_ess(c) for c in chains))
            return RunResult(finish_figure(fig), f"Basic R-hat = {rhat:.3f}; rough Monte Carlo standard error = {mcse:.3f}. Increase broken separation to create false stationary regimes.")

        if index == 14:
            grid = np.linspace(-4, 5, 700)
            logp = lambda x: -0.5 * np.asarray(x) ** 2 + 1.5 * np.tanh(np.asarray(x))
            target = np.exp(logp(grid) - np.max(logp(grid)))
            target /= np.trapezoid(target, grid)
            manual_mean, manual_sd = values["q mean"], values["q sd"]
            optimized_mean, optimized_sd, engine = _optimize_vi(logp, seed, is_wasm, importlib, np)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(grid, target, color=COLORS["posterior"], lw=3, label="target")
            ax.plot(grid, np.exp(-0.5 * ((grid - manual_mean) / manual_sd) ** 2) / (manual_sd * np.sqrt(2 * np.pi)), color=COLORS["accent"], lw=2, ls="--", label="your q")
            ax.plot(grid, np.exp(-0.5 * ((grid - optimized_mean) / optimized_sd) ** 2) / (optimized_sd * np.sqrt(2 * np.pi)), color=COLORS["sample"], lw=2, label="optimized q")
            style_axes(ax, "Gaussian q fitted to a skewed target", "θ", "density")
            ax.legend(frameon=False)
            return RunResult(finish_figure(fig), f"{engine} optimized q to mean {optimized_mean:.2f}, SD {optimized_sd:.2f}. Compare its missing tail with the target.")

        if index == 15:
            target_name = values["target"]
            fig, ax = plt.subplots(figsize=(7, 5.2))
            if target_name == "skewed":
                x = np.linspace(-3, 6, 600)
                density = np.exp(-0.5 * x**2 + 1.5 * np.tanh(x)); density /= np.trapezoid(density, x)
                ax.plot(x, density, color=COLORS["posterior"], lw=3, label="skewed target")
                ax.plot(x, norm_pdf(x, 0.75, 0.85), color=COLORS["accent"], lw=2, ls="--", label="Gaussian q")
                style_axes(ax, "A Gaussian misses skew and tail balance", "θ", "density")
            elif target_name == "multimodal":
                x = np.linspace(-6, 6, 700)
                density = 0.5 * norm_pdf(x, -2.2, 0.65) + 0.5 * norm_pdf(x, 2.2, 0.65)
                ax.plot(x, density, color=COLORS["posterior"], lw=3, label="two modes")
                ax.plot(x, norm_pdf(x, -2.2, 0.7), color=COLORS["accent"], lw=2, ls="--", label="one VI solution")
                style_axes(ax, "Reverse-KL VI can select one mode", "θ", "density")
            else:
                z = rng.normal(size=(3000, 2))
                target = np.column_stack([z[:, 0], z[:, 1] + 0.45 * (z[:, 0] ** 2 - 1)])
                q = rng.multivariate_normal(target.mean(0), np.cov(target.T), size=1800)
                ax.scatter(target[:, 0], target[:, 1], s=5, alpha=0.12, color=COLORS["posterior"], label="banana target")
                ax.scatter(q[:, 0], q[:, 1], s=5, alpha=0.10, color=COLORS["accent"], label="Gaussian q")
                style_axes(ax, "Linear covariance cannot bend", "θ₁", "θ₂")
            ax.legend(frameon=False)
            return RunResult(finish_figure(fig), "The approximation family sets a hard shape limit. Optimization quality cannot remove this limit.")

        if index == 16:
            data = np.r_[rng.normal(-2.2, 0.7, 90), rng.normal(2.0, 0.7, 110)]
            rng.shuffle(data)
            means, weights, responsibilities, history = gaussian_mixture_em(data, [values["initial left mean"], values["initial right mean"]], [0.5, 0.5], 0.7, int(values["iterations"]))
            colors = np.column_stack([responsibilities[:, 0], np.full(len(data), 0.25), responsibilities[:, 1], np.full(len(data), 0.65)])
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].scatter(data, np.zeros_like(data), c=colors, s=28)
            axes[0].scatter(means, [0, 0], marker="X", s=180, color=[COLORS["accent"], COLORS["sample"]], edgecolor="white")
            axes[1].plot(np.arange(1, len(history) + 1), history, "o-", color=COLORS["posterior"])
            style_axes(axes[0], "Soft responsibilities", "x", "")
            style_axes(axes[1], "Observed-data log likelihood", "EM iteration", "log likelihood")
            monotone = np.all(np.diff(history) >= -1e-8)
            return RunResult(finish_figure(fig), f"Means = {means.round(2).tolist()}; weights = {weights.round(2).tolist()}; log likelihood non-decreasing = {monotone}.")

        if index == 17:
            uncertainty = values["uncertainty required"]
            rows = [
                {"method": "MAP", "learns": "one point", "uncertainty": "no", "main risk": "missed width or modes"},
                {"method": "EM", "learns": "point + responsibilities", "uncertainty": "limited", "main risk": "local optimum"},
                {"method": "VI", "learns": "approximate q", "uncertainty": "approximate", "main risk": "family mismatch"},
                {"method": "MCMC", "learns": "dependent draws", "uncertainty": "yes", "main risk": "poor mixing"},
                {"method": "NUTS", "learns": "HMC draws", "uncertainty": "yes", "main risk": "bad geometry/divergence"},
            ]
            recommended = "NUTS/MCMC" if uncertainty == "yes" else "MAP or EM"
            return RunResult(None, f"Your current requirement suggests {recommended}. This table compares output objects, not only speed.", rows)

        if index == 18:
            raw = np.array([0.70, 0.70, 1.00])
            n = np.array([100, 10, 1])
            se = np.sqrt(np.maximum(raw * (1 - raw), 0.08) / n)
            pooled, weight = shrinkage_estimates(raw, se, 0.68, values["population sd"])
            fig, ax = plt.subplots(figsize=(8, 4.4))
            y = np.arange(3)
            ax.scatter(raw, y, s=90, color=COLORS["likelihood"], label="raw")
            ax.scatter(pooled, y, s=90, color=COLORS["posterior"], label="partial pooling")
            for i in range(3):
                ax.annotate("", xy=(pooled[i], y[i]), xytext=(raw[i], y[i]), arrowprops={"arrowstyle": "->", "color": COLORS["ink"]})
            ax.axvline(0.68, color=COLORS["prior"], ls="--", label="population mean")
            ax.set_yticks(y, ["A: 70/100", "B: 7/10", "C: 1/1"])
            ax.set_xlim(0.5, 1.03)
            style_axes(ax, "Data-poor groups shrink more", "rate")
            ax.legend(frameon=False)
            rows = [{"group": name, "raw": f"{r:.3f}", "pooled": f"{p:.3f}", "data weight": f"{w:.2f}"} for name, r, p, w in zip("ABC", raw, pooled, weight)]
            return RunResult(finish_figure(fig), "As population scale approaches zero, all estimates approach the population mean.", rows)

        if index == 19:
            if values["engine"] == "PyMC NUTS":
                if is_wasm:
                    return RunResult(None, "The public browser build uses the fast conjugate view. Run locally to fit the full hierarchical coupon model with PyMC NUTS.")
                return _run_coupon_pymc(importlib, np, plt, style_axes, finish_figure, COLORS, seed, values["prior odds"])
            k_pos, n_pos = np.array([14, 6, 2]), np.array([20, 10, 2])
            k_neg, n_neg = np.array([2, 4, 1]), np.array([100, 80, 40])
            draws = 12000
            s_draws = np.column_stack([rng.beta(k_pos[j] + 1.5, n_pos[j] - k_pos[j] + 1.5, draws) for j in range(3)])
            f_draws = np.column_stack([rng.beta(k_neg[j] + 1.2, n_neg[j] - k_neg[j] + 8.0, draws) for j in range(3)])
            ratios = s_draws / np.maximum(f_draws, 1e-6)
            rule = int(values["rule"]) - 1
            posterior_odds = values["prior odds"] * ratios[:, rule]
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
            for j in range(3):
                axes[0].hist(s_draws[:, j], bins=45, density=True, histtype="step", label=f"rule {j+1}")
                axes[1].hist(f_draws[:, j], bins=45, density=True, histtype="step", label=f"rule {j+1}")
            axes[2].hist(posterior_odds, bins=70, density=True, color=COLORS["posterior"], alpha=0.65)
            for ax, title in zip(axes, ("Sensitivity sⱼ", "False-fire rate fⱼ", f"Rule {rule+1}: posterior odds")):
                style_axes(ax, title)
            axes[0].legend(frameon=False)
            q = np.quantile(posterior_odds, [0.05, 0.5, 0.95])
            rows = [{"rule": j + 1, "median s": f"{np.median(s_draws[:,j]):.3f}", "median f": f"{np.median(f_draws[:,j]):.3f}", "median s/f": f"{np.median(ratios[:,j]):.1f}"} for j in range(3)]
            return RunResult(finish_figure(fig), f"Selected-rule posterior odds 5/50/95% = {q.round(2).tolist()}. This fast view is conjugate and only approximates hierarchical pooling.", rows)

        if index == 20:
            if values["engine"] == "PyMC NUTS":
                if is_wasm:
                    return RunResult(None, "PyMC NUTS requires the local environment. The geometry view remains fully interactive in the browser.")
                return _run_funnel_pymc(importlib, np, plt, style_axes, finish_figure, COLORS, seed)
            n = 7000
            v = rng.normal(0, 3, n)
            x = rng.normal(0, np.exp(v / 2))
            fig, ax = plt.subplots(figsize=(7, 5.2))
            ax.scatter(v, x, s=4, alpha=0.08, color=COLORS["sample"])
            neck = values["neck depth"]
            ax.axvline(neck, color=COLORS["reject"], ls="--", label=f"inspect v={neck:.1f}")
            ax.set_ylim(-15, 15)
            style_axes(ax, "Neal's funnel", "v (log variance)", "x")
            ax.legend(frameon=False)
            return RunResult(finish_figure(fig), f"At v={neck:.1f}, the conditional SD of x is exp(v/2) = {np.exp(neck/2):.4f}. One global step size must handle mouth and neck.")

        if index == 21:
            if values["engine"] == "PyMC compare":
                if is_wasm:
                    return RunResult(None, "The browser shows the coordinate transform. Run locally for the two live PyMC fits.")
                return _run_parameterization_compare(importlib, np, plt, style_axes, finish_figure, COLORS, seed)
            sigma = values["population sd"]
            z = rng.normal(size=(2500, 2))
            centered = np.column_stack([np.exp(z[:, 0]), np.exp(z[:, 0]) * z[:, 1]])
            noncentered = np.column_stack([np.exp(z[:, 0]), z[:, 1]])
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
            axes[0].scatter(centered[:, 0], centered[:, 1], s=4, alpha=0.12, color=COLORS["sample"])
            axes[1].scatter(noncentered[:, 0], noncentered[:, 1], s=4, alpha=0.12, color=COLORS["posterior"])
            axes[0].axvline(sigma, color=COLORS["accent"], ls="--")
            axes[1].axvline(sigma, color=COLORS["accent"], ls="--")
            style_axes(axes[0], "Centered coordinates (σ, η)", "σ", "η")
            style_axes(axes[1], "Non-centered coordinates (σ, z)", "σ", "z")
            axes[0].set_xlim(0, 5); axes[1].set_xlim(0, 5)
            return RunResult(finish_figure(fig), "The deterministic transform η=μ+σz keeps the model meaning but removes the narrowing in sampling coordinates.")

        if index == 22:
            l2, steps = values["L2 strength"], int(values["training steps"])
            x = np.linspace(-2, 2, 80)[:, None]
            y = np.sin(2.2 * x[:, 0]) + rng.normal(0, 0.12, len(x))
            if not is_wasm:
                torch = _import_optional("torch")
                torch.manual_seed(seed)
                xt = torch.tensor(x, dtype=torch.float32)
                yt = torch.tensor(y[:, None], dtype=torch.float32)
                model = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
                optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
                losses = []
                for _ in range(steps):
                    optimizer.zero_grad(); prediction = model(xt)
                    data_loss = torch.mean((prediction - yt) ** 2)
                    penalty = 0.5 * l2 * sum(torch.sum(p**2) for p in model.parameters()) / len(x)
                    loss = data_loss + penalty; loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                prediction = model(xt).detach().numpy()[:, 0]
                weight_norm = float(np.sqrt(sum(float(torch.sum(p.detach() ** 2)) for p in model.parameters())))
                engine = "Torch autograd"
            else:
                features = np.column_stack([np.ones(len(x)), x[:, 0], x[:, 0] ** 2, x[:, 0] ** 3, x[:, 0] ** 4, x[:, 0] ** 5])
                weights = np.linalg.solve(features.T @ features + l2 * np.eye(features.shape[1]), features.T @ y)
                prediction = features @ weights; weight_norm = float(np.linalg.norm(weights)); losses = [float(np.mean((prediction-y)**2))]
                engine = "browser ridge fallback"
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
            axes[0].scatter(x[:, 0], y, s=16, alpha=0.5, color=COLORS["sample"])
            axes[0].plot(x[:, 0], prediction, color=COLORS["accent"], lw=2.5)
            axes[1].plot(losses, color=COLORS["posterior"], lw=2)
            style_axes(axes[0], f"Tiny regression model ({engine})", "x", "y")
            style_axes(axes[1], "MAP-style training objective", "step", "loss")
            return RunResult(finish_figure(fig), f"Weight norm = {weight_norm:.2f}. L2 strength acts like the precision of a zero-mean Gaussian weight prior.")

        if index == 23:
            scenario = values["scenario"]
            mapping = {
                "fast point forecast": ("MAP", "one parameter point", "negative log posterior", "no", "local minima or missing uncertainty"),
                "latent mixture": ("EM", "parameters + responsibilities", "expected complete-data objective", "limited", "local optimum or label symmetry"),
                "calibrated small model": ("NUTS", "posterior draws", "posterior density", "yes", "divergences or poor mixing"),
                "large Bayesian model": ("VI", "approximate posterior q", "ELBO", "approximate", "underestimated or missing uncertainty"),
            }
            method, obj, target, uncertainty, failure = mapping[scenario]
            rows = [
                {"question": "1. learned object", "expert answer": obj},
                {"question": "2. objective or target", "expert answer": target},
                {"question": "3. uncertainty preserved", "expert answer": uncertainty},
                {"question": "4. gradient role", "expert answer": "optimize directly or integrate HMC dynamics"},
                {"question": "5. likely failure", "expert answer": failure},
            ]
            return RunResult(None, f"For this compact scenario, the expert first test is {method}. A real choice also uses compute limits and failure cost.", rows)

        raise ValueError(f"Late experiment does not support lab {index}")

    def norm_pdf(x, mean, sd):
        return np.exp(-0.5 * ((np.asarray(x) - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    def _optimize_vi(logp, seed, is_wasm, importlib, np):
        if not is_wasm:
            torch = _import_optional("torch")
            torch.manual_seed(seed)
            mean = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            log_sd = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.Adam([mean, log_sd], lr=0.04)
            eps = torch.randn(1600, dtype=torch.float64)
            for _ in range(180):
                optimizer.zero_grad(); sample = mean + torch.exp(log_sd) * eps
                log_target = -0.5 * sample**2 + 1.5 * torch.tanh(sample)
                log_q = -0.5 * eps**2 - log_sd - 0.5 * np.log(2 * np.pi)
                loss = -(log_target - log_q).mean(); loss.backward(); optimizer.step()
            return float(mean.detach()), float(torch.exp(log_sd).detach()), "Torch autograd"
        eps = np.random.default_rng(seed).normal(size=2500)
        params = np.array([0.0, 0.0])
        def objective(p):
            sample = p[0] + np.exp(p[1]) * eps
            log_q = -0.5 * eps**2 - p[1] - 0.5 * np.log(2 * np.pi)
            return np.mean(logp(sample) - log_q)
        for _ in range(120):
            grad = np.zeros(2)
            for j in range(2):
                step = np.zeros(2); step[j] = 1e-4
                grad[j] = (objective(params + step) - objective(params - step)) / 2e-4
            params += 0.035 * grad
        return params[0], np.exp(params[1]), "NumPy finite-difference VI"

    def _run_coupon_pymc(importlib, np, plt, style_axes, finish_figure, colors, seed, prior_odds):
        pm = _import_optional("pymc")
        k_pos, n_pos = np.array([14, 6, 2]), np.array([20, 10, 2])
        k_neg, n_neg = np.array([2, 4, 1]), np.array([100, 80, 40])
        with pm.Model():
            mu_s = pm.Normal("mu_s", 0, 2); sigma_s = pm.HalfNormal("sigma_s", 1)
            eta = pm.Normal("eta", mu_s, sigma_s, shape=3); s = pm.Deterministic("s", pm.math.sigmoid(eta))
            pm.Binomial("positive_fires", n=n_pos, p=s, observed=k_pos)
            mu_f = pm.Normal("mu_f", 0, 2); sigma_f = pm.HalfNormal("sigma_f", 1)
            xi = pm.Normal("xi", mu_f, sigma_f, shape=3); f = pm.Deterministic("f", pm.math.sigmoid(xi))
            pm.Binomial("negative_fires", n=n_neg, p=f, observed=k_neg)
            idata = pm.sample(draws=400, tune=800, chains=2, cores=1, target_accept=0.97, random_seed=seed, progressbar=False)
        s_draws = np.asarray(idata.posterior["s"]).reshape(-1, 3)
        f_draws = np.asarray(idata.posterior["f"]).reshape(-1, 3)
        ratios = s_draws / np.maximum(f_draws, 1e-9)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        for j in range(3):
            axes[0].hist(s_draws[:, j], bins=35, density=True, histtype="step", label=f"rule {j+1}")
            axes[1].hist(f_draws[:, j], bins=35, density=True, histtype="step")
            axes[2].hist(prior_odds * ratios[:, j], bins=45, density=True, histtype="step")
        for ax, title in zip(axes, ("Hierarchical sensitivity", "Hierarchical false-fire rate", "Transaction posterior odds")):
            style_axes(ax, title)
        axes[0].legend(frameon=False)
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
        rows = [{"rule": j+1, "s median": f"{np.median(s_draws[:,j]):.3f}", "f median": f"{np.median(f_draws[:,j]):.3f}", "s/f median": f"{np.median(ratios[:,j]):.1f}"} for j in range(3)]
        diagnostic = (
            "Do not trust the downstream odds yet; Labs 20 and 21 explain and repair this geometry."
            if divergences
            else "The samples preserve dependence through the odds update."
        )
        return RunResult(finish_figure(fig), f"Full PyMC hierarchy completed with {divergences} divergences. {diagnostic}", rows)

    def _run_funnel_pymc(importlib, np, plt, style_axes, finish_figure, colors, seed):
        pm = _import_optional("pymc")
        with pm.Model():
            v = pm.Normal("v", 0, 3)
            pm.Normal("x", 0, pm.math.exp(v / 2), shape=5)
            idata = pm.sample(draws=350, tune=350, chains=2, cores=1, target_accept=0.75, random_seed=seed, progressbar=False)
        v_draws = np.asarray(idata.posterior["v"]).reshape(-1)
        x_draws = np.asarray(idata.posterior["x"]).reshape(-1, 5)[:, 0]
        divergent = np.asarray(idata.sample_stats["diverging"]).reshape(-1).astype(bool)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].scatter(v_draws[~divergent], x_draws[~divergent], s=8, alpha=0.2, color=colors["sample"])
        axes[0].scatter(v_draws[divergent], x_draws[divergent], s=20, color=colors["reject"], label="divergence")
        axes[1].plot(v_draws, lw=0.6, color=colors["sample"])
        style_axes(axes[0], "NUTS on centered funnel", "v", "x₁"); style_axes(axes[1], "v trace", "draw", "v")
        axes[0].legend(frameon=False)
        return RunResult(finish_figure(fig), f"Divergences = {divergent.sum()}. Their location can identify geometry that leapfrog could not resolve.")

    def _run_parameterization_compare(importlib, np, plt, style_axes, finish_figure, colors, seed):
        pm = _import_optional("pymc")
        observed = np.array([0.2, -0.1, 0.15])
        results = []
        for centered in (True, False):
            with pm.Model():
                mu = pm.Normal("mu", 0, 1); sigma = pm.HalfNormal("sigma", 1)
                if centered:
                    eta = pm.Normal("eta", mu, sigma, shape=3)
                else:
                    z = pm.Normal("z", 0, 1, shape=3); eta = pm.Deterministic("eta", mu + sigma * z)
                pm.Normal("obs", eta, 1.5, observed=observed)
                idata = pm.sample(draws=280, tune=350, chains=2, cores=1, target_accept=0.85, random_seed=seed + int(centered), progressbar=False)
            results.append(idata)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        rows = []
        for ax, idata, name in zip(axes, results, ("centered", "non-centered")):
            sigma_draws = np.asarray(idata.posterior["sigma"]).reshape(-1)
            ax.plot(sigma_draws, lw=0.6, color=colors["sample"])
            div = int(np.asarray(idata.sample_stats["diverging"]).sum())
            rows.append({"parameterization": name, "divergences": div, "rough ESS σ": f"{approximate_ess(sigma_draws):.0f}"})
            style_axes(ax, f"{name}: σ trace", "draw", "σ")
        return RunResult(finish_figure(fig), "Weak group data often make the non-centered coordinates easier. Compare both diagnostics on this same model.", rows)

    return (run_experiment_late,)


@app.cell(hide_code=True)
def _(np):
    def check_exercise(index, implementations):
        def require(condition, message):
            if not bool(condition):
                raise AssertionError(message)

        try:
            if index == 0:
                (learning_loop_impl,) = implementations
                require(
                    learning_loop_impl()
                    == ["predict", "simulate", "inspect", "code", "explain"],
                    "Return the five exact stage strings in the required order.",
                )
            elif index == 1:
                (normal_log_density_impl,) = implementations
                got = normal_log_density_impl(np.array([0.0, 1.0]), 0.0, 1.0)
                expected = -0.5 * np.log(2 * np.pi) - 0.5 * np.array([0.0, 1.0]) ** 2
                require(got is not None, "Return the log-density. The current result is None.")
                require(
                    np.allclose(got, expected),
                    f"Expected approximately {expected}; received {got!r}.",
                )
            elif index == 2:
                log_likelihood_impl, log_posterior_impl = implementations
                ll = log_likelihood_impl(0.5, np.array([0.0, 1.0]), 1.0)
                lp = log_posterior_impl(0.5, np.array([0.0, 1.0]), 1.0, 0.0, 2.0)
                require(ll is not None and lp is not None, "Both functions must return scalar scores, not None.")
                require(np.isfinite(ll) and np.isfinite(lp) and lp < ll, "Return finite scores and add the log-prior to the likelihood.")
            elif index == 3:
                (gradient_ascent_impl,) = implementations
                path = gradient_ascent_impl(lambda x: -(x - 2.0) ** 2, -2.0, 0.1, 80)
                require(path is not None, "Return the full NumPy path, not None.")
                require(len(path) == 81 and abs(path[-1] - 2.0) < 0.05, "The path must contain 81 values and finish near x=2.")
            elif index == 4:
                (laplace_sd_impl,) = implementations
                sd = laplace_sd_impl(lambda x: -0.5 * (x / 2.0) ** 2, 0.0)
                require(sd is not None and np.isclose(sd, 2.0, rtol=0.02), "Return a width close to 2.0.")
            elif index == 5:
                (metropolis_impl,) = implementations
                samples, acceptance = metropolis_impl(lambda x: -0.5 * x**2, 0.0, 1.0, 5000, np.random.default_rng(4))
                require(len(samples) == 5000, "Store exactly 5,000 states, including repeats.")
                require(abs(samples.mean()) < 0.15 and 0.2 < acceptance < 0.95, "The sample mean or acceptance rate is outside the expected range.")
            elif index == 6:
                (autocorrelation_impl,) = implementations
                require(np.isclose(autocorrelation_impl(np.arange(8.0), 0), 1.0), "Lag-zero autocorrelation must equal 1.")
                require(autocorrelation_impl(np.arange(8.0), 1) > 0.4, "Lag-one autocorrelation must be positive for an increasing sequence.")
            elif index == 7:
                (covariance_impl,) = implementations
                matrix = covariance_impl(0.7)
                require(matrix is not None, "Return the matrix, not None.")
                require(np.allclose(matrix, [[1.0, 0.7], [0.7, 1.0]]), "Put 1 on the diagonal and rho off the diagonal.")
                require(np.all(np.linalg.eigvalsh(matrix) > 0), "The test matrix must be positive definite.")
            elif index == 8:
                (numerical_gradient_impl,) = implementations

                def coupled_quadratic(x):
                    vector = np.asarray(x, dtype=float)
                    if vector.shape != (2,):
                        return np.nan
                    return (
                        vector[0] ** 2
                        + 3.0 * vector[0] * vector[1]
                        + 2.0 * vector[1] ** 2
                    )

                point = np.array([1.0, 2.0])
                original = point.copy()
                grad = numerical_gradient_impl(coupled_quadratic, point)
                require(grad is not None, "Return the gradient vector, not None.")
                require(
                    np.asarray(grad).shape == point.shape,
                    "Return one derivative for each coordinate, with the same shape as point.",
                )
                require(
                    np.array_equal(point, original),
                    "Do not modify the input point in place.",
                )
                require(
                    np.allclose(grad, [8.0, 11.0], atol=1e-3),
                    "Expected gradient [8, 11]. For coordinate i, copy the full point, "
                    "change only coordinate i by +eps and -eps, and evaluate fn on both full vectors.",
                )
            elif index == 9:
                (energy_impl,) = implementations
                total = energy_impl(lambda q: -0.5 * np.dot(q, q), np.array([1.0, 0.0]), np.array([0.0, 2.0]))
                require(total is not None and np.isclose(total, 2.5), "Expected total energy 2.5.")
            elif index == 10:
                (leapfrog_step_impl,) = implementations
                grad = lambda q: -q
                q0 = np.array([1.0])
                p0 = np.array([0.3])
                q, p = leapfrog_step_impl(q0, p0, 0.01, grad)
                require(q.shape == p.shape == (1,), "Return q and p with their original shapes.")
                old_energy = 0.5 * q0[0] ** 2 + 0.5 * p0[0] ** 2
                new_energy = 0.5 * q[0] ** 2 + 0.5 * p[0] ** 2
                require(
                    abs(new_energy - old_energy) < 1e-5,
                    "Energy error is too large for this small step.",
                )
                q_forward, p_forward = leapfrog_step_impl(q0, p0, 0.15, grad)
                q_back, p_back = leapfrog_step_impl(
                    q_forward, p_forward, -0.15, grad
                )
                require(
                    np.allclose(q_back, q0, atol=1e-10)
                    and np.allclose(p_back, p0, atol=1e-10),
                    "The step must retrace when run backward. Use matching half momentum updates on both sides of the position update.",
                )
            elif index == 11:
                (one_hmc_transition_impl,) = implementations

                class FixedRng:
                    def __init__(self, momentum, uniform):
                        self.momentum = np.asarray(momentum, dtype=float)
                        self.uniform_value = float(uniform)

                    def normal(self, size):
                        require(tuple(size) == self.momentum.shape, "Draw momentum with the same shape as position.")
                        return self.momentum.copy()

                    def uniform(self):
                        return self.uniform_value

                observed = {}

                def equal_energy_integrator(q, p):
                    observed["momentum"] = p.copy()
                    return q.copy(), p.copy()

                current = np.array([0.5, -0.25])
                next_q, accepted = one_hmc_transition_impl(
                    current,
                    lambda q: -0.5 * float(np.dot(q, q)),
                    equal_energy_integrator,
                    FixedRng([0.4, -0.3], 0.9),
                )
                require(
                    np.allclose(observed.get("momentum"), [0.4, -0.3]),
                    "Draw fresh momentum and pass it to the trajectory integrator.",
                )
                require(
                    bool(accepted) and np.allclose(next_q, current),
                    "An equal-energy proposal must be accepted.",
                )

                def high_energy_integrator(q, p):
                    return q + np.array([10.0, 0.0]), p

                rejected_q, accepted = one_hmc_transition_impl(
                    current,
                    lambda q: -0.5 * float(np.dot(q, q)),
                    high_energy_integrator,
                    FixedRng([0.0, 0.0], 0.5),
                )
                require(
                    not bool(accepted),
                    "A proposal with this very large positive Hamiltonian error must be rejected.",
                )
                require(
                    np.allclose(rejected_q, current),
                    "On rejection, return the old position so the chain stores a repeated state.",
                )
            elif index == 12:
                (is_uturn_impl,) = implementations
                require(is_uturn_impl(np.zeros(2), np.array([2.0, 0.0]), np.array([-1.0, 0.0])), "Opposing displacement and momentum must report a U-turn.")
                require(not is_uturn_impl(np.zeros(2), np.array([2.0, 0.0]), np.array([1.0, 0.0])), "Aligned displacement and momentum must not report a U-turn.")
            elif index == 13:
                (basic_rhat_impl,) = implementations
                chains = np.random.default_rng(2).normal(size=(4, 1000))
                value = basic_rhat_impl(chains)
                require(
                    value is not None and np.ndim(value) == 0 and np.isfinite(value),
                    "Return one finite scalar R-hat value.",
                )
                require(
                    0.98 < value < 1.03,
                    "Independent Normal chains must give R-hat near 1.",
                )

                hand_chains = np.array([[0.0, 2.0], [2.0, 4.0]])
                hand_value = basic_rhat_impl(hand_chains)
                require(
                    hand_value is not None
                    and np.ndim(hand_value) == 0
                    and np.isclose(hand_value, np.sqrt(1.5)),
                    "For [[0, 2], [2, 4]], W=2 and B=4, so R-hat must be sqrt(1.5), approximately 1.225.",
                )

                base = np.linspace(-1.0, 1.0, 100)
                separated_chains = np.vstack(
                    [base - 5.0, base - 2.0, base + 2.0, base + 5.0]
                )
                separated_value = basic_rhat_impl(separated_chains)
                require(
                    separated_value is not None and separated_value > 2.0,
                    "Chains with far-apart means must have a large between-chain variance B and an R-hat well above 1.",
                )
            elif index == 14:
                (mc_elbo_impl,) = implementations
                eps = np.random.default_rng(2).normal(size=2000)
                value = mc_elbo_impl(lambda z: -0.5 * z**2, 0.0, 0.0, eps)
                require(value is not None and np.isfinite(value), "Return one finite ELBO estimate.")
            elif index == 15:
                (diagonal_gaussian_sample_impl,) = implementations
                samples = diagonal_gaussian_sample_impl([1.0, 2.0], [0.0, 0.0], np.zeros((5, 2)))
                require(samples is not None and samples.shape == (5, 2), "Return five two-dimensional samples.")
                require(np.allclose(samples, [1.0, 2.0]), "Zero noise must return the mean for every sample.")
            elif index == 16:
                (m_step_impl,) = implementations
                x = np.array([-2.0, -1.0, 1.0, 2.0])
                responsibilities = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
                weights, means = m_step_impl(x, responsibilities)
                require(np.allclose(weights, [0.5, 0.5]), "Expected equal component weights.")
                require(np.allclose(means, [-1.5, 1.5]), "Expected means [-1.5, 1.5].")
            elif index == 17:
                (output_object_impl,) = implementations
                require(output_object_impl("MAP") == "point", "MAP must map to 'point'.")
                require("samples" in output_object_impl("MCMC"), "MCMC output must contain 'samples'.")
            elif index == 18:
                (shrink_impl,) = implementations
                weak = shrink_impl(1.0, 0.5, 0.6, 0.05)
                strong = shrink_impl(1.0, 0.05, 0.6, 0.5)
                require(weak is not None and strong is not None, "Return both shrinkage means, not None.")
                require(abs(weak - 0.6) < abs(strong - 0.6), "The weak group under a tight population must shrink more.")
            elif index == 19:
                (classify_variables_impl,) = implementations
                roles = classify_variables_impl()
                require(roles is not None, "Return the role dictionary, not None.")
                require("k_pos" in roles["observed"], "Put k_pos in observed.")
                require("eta" in roles["latent"], "Put eta in latent.")
                require("s_over_f" in roles["deterministic"], "Put s_over_f in deterministic.")
            elif index == 20:
                (funnel_sample_impl,) = implementations
                result = funnel_sample_impl(-8.0, np.array([1.0, -1.0]))
                require(result is not None and np.allclose(result, np.exp(-4) * np.array([1.0, -1.0])), "Scale z by exp(v/2).")
            elif index == 21:
                (noncenter_impl,) = implementations
                result = noncenter_impl(2.0, 0.5, np.array([-1.0, 1.0]))
                require(result is not None and np.allclose(result, [1.5, 2.5]), "Expected [1.5, 2.5].")
            elif index == 22:
                (map_loss_impl,) = implementations
                low = map_loss_impl(1.0, np.array([1.0, 2.0]), 0.0)
                high = map_loss_impl(1.0, np.array([1.0, 2.0]), 2.0)
                require(low is not None and high is not None, "Return scalar losses, not None.")
                require(np.isclose(low, 1.0) and high > low, "Zero L2 must keep the data loss; positive L2 must increase it.")
            elif index == 23:
                (choose_method_impl,) = implementations
                require(choose_method_impl(True, False, True) == "NUTS", "Uncertainty plus hard geometry must select NUTS.")
                require(choose_method_impl(False, True, False) == "EM", "A latent mixture without uncertainty must select EM.")
            return "success", f"PASS · Lab {index:02d}. Your implementation has the required behavior."
        except Exception as exc:
            detail = str(exc) or "The function did not satisfy the checkpoint."
            return "danger", f"FAIL · Lab {index:02d} · {type(exc).__name__}: {detail}"


    return (check_exercise,)


@app.cell(hide_code=True)
def orientation_views(mo):
    def make_orientation_view(index):
        views = [
            (
                "The course learning loop",
                "Read from left to right. A prediction records your current mental model. The simulation creates evidence. Inspection finds the mismatch. Code exposes the mechanism. Explanation checks whether the new idea transfers to a new case.",
                "This is a route map, not an experiment result. Every later lab returns to these five stages.",
                """
                <rect x="25" y="88" width="120" height="58" rx="12" class="box"/><text x="85" y="122">predict</text>
                <path d="M145 117 H180" class="arrow"/><rect x="180" y="88" width="120" height="58" rx="12" class="box"/><text x="240" y="122">simulate</text>
                <path d="M300 117 H335" class="arrow"/><rect x="335" y="88" width="120" height="58" rx="12" class="box"/><text x="395" y="122">inspect</text>
                <path d="M455 117 H490" class="arrow"/><rect x="490" y="88" width="105" height="58" rx="12" class="box"/><text x="542" y="122">code</text>
                <path d="M595 117 H630" class="arrow"/><rect x="630" y="88" width="105" height="58" rx="12" class="box"/><text x="682" y="122">explain</text>
                """,
            ),
            (
                "A demand density and an interval probability",
                "Read the horizontal axis as tomorrow's requested loaves. The curve height is density, not a point probability. The shaded area is the probability that demand lies between 50 and 70 loaves. The vertical line marks the mean and mode in this symmetric example.",
                "The fixed picture gives you visual landmarks. After you predict, the controls redraw the same curve and shaded interval with new values.",
                """
                <path d="M70 215 H715 M70 215 V35" class="axis"/><text x="660" y="244">loaves requested</text><text x="16" y="45" class="small">density</text>
                <path d="M120 214 C230 213 270 190 320 112 C350 67 380 48 400 48 C420 48 450 67 480 112 C530 190 570 213 680 214" class="curve teal"/>
                <path d="M320 214 L320 112 C350 67 380 48 400 48 C420 48 450 67 480 112 L480 214 Z" class="fill"/>
                <path d="M400 215 V48" class="dash"/><text x="408" y="66" class="label">mean and mode: 60</text>
                <text x="326" y="190" class="label">probability mass</text><text x="300" y="233" class="small">50</text><text x="463" y="233" class="small">70</text>
                """,
            ),
            (
                "Existing sales build a posterior; one new day can shift it",
                "The horizontal axis contains candidate values of average demand, not future sales. Purple is the prior and orange is the likelihood from existing days. Blue is the posterior before one highlighted new day. Teal is the posterior after that day. The distance between the blue and teal centers is the update you must explain.",
                "The live experiment adds a slider for the new sales value. Hold that value fixed, compare narrow and wide priors, and read the before-to-after shift in the right panel.",
                """
                <path d="M70 215 H715 M70 215 V35" class="axis"/><text x="625" y="244">candidate mean demand</text>
                <path d="M100 214 C210 210 260 130 330 68 C390 25 455 150 520 208" class="curve purple"/>
                <path d="M250 214 C320 210 375 110 430 52 C480 110 520 206 610 214" class="curve orange"/>
                <path d="M235 214 C320 210 350 95 390 52 C430 95 460 210 545 214" class="curve blue"/>
                    <path d="M285 214 C365 210 395 96 435 56 C475 96 505 210 590 214" class="curve teal"/>
                    <path d="M560 214 V165" class="dash red-stroke"/>
                <text x="205" y="72" class="purple-text">prior</text><text x="500" y="96" class="orange-text">likelihood</text>
                    <text x="355" y="42" class="blue-text">before</text><text x="470" y="60" class="teal-text">after</text>
                    <text x="590" y="165" class="red-text">new day</text>
                """,
            ),
            (
                "Climbing a posterior to one operating value",
                "The black curve is a log-posterior score. Higher is better. The blue dots are successive parameter values from gradient ascent. Their spacing shows the size of each update, and the final dot is the point estimate returned by optimization.",
                "The experiment changes the starting point, learning rate, and step count. Watch whether the path approaches, crosses, or escapes the peak.",
                """
                <path d="M70 215 H715 M70 215 V35" class="axis"/><path d="M95 205 Q390 -40 685 205" class="curve ink"/>
                <polyline points="130,185 225,130 305,88 360,65 390,57" class="path blue"/>
                <circle cx="130" cy="185" r="7" class="dot blue-fill"/><circle cx="225" cy="130" r="7" class="dot blue-fill"/><circle cx="305" cy="88" r="7" class="dot blue-fill"/><circle cx="360" cy="65" r="7" class="dot blue-fill"/><circle cx="390" cy="57" r="8" class="dot accent-fill"/>
                <text x="112" y="207" class="label">start</text><text x="400" y="52" class="label">MAP</text>
                """,
            ),
            (
                "The same MAP can hide different tail risk",
                "Both curves peak at 60 loaves, so MAP returns the same point. The narrow curve keeps most plausible values near 60. The broad curve assigns more mass to low and high demand, which changes waste and stockout probabilities.",
                "The experiment changes only the broad width. Compare intervals and tails, not only the shared peak location.",
                """
                <path d="M70 215 H715 M70 215 V35" class="axis"/><path d="M100 214 C190 210 280 150 390 86 C500 150 590 210 680 214" class="curve purple"/>
                <path d="M260 214 C325 207 350 75 390 42 C430 75 455 207 520 214" class="curve teal"/>
                <path d="M390 215 V42" class="dash"/><text x="400" y="54" class="label">same MAP</text><text x="540" y="160" class="purple-text">broad</text><text x="438" y="96" class="teal-text">narrow</text>
                """,
            ),
            (
                "A random walk proposes, accepts, or stays still",
                "The curve is the target posterior. The blue point is the current demand value. A nearby teal proposal is accepted; a distant red proposal is rejected. On rejection, the saved next state is the current point again.",
                "After prediction, proposal standard deviation changes the typical arrow length. Inspect movement and acceptance together.",
                """
                <path d="M70 215 H715 M70 215 V35" class="axis"/><path d="M110 214 C230 210 300 55 390 48 C480 55 550 210 680 214" class="curve ink"/>
                <circle cx="350" cy="68" r="9" class="dot blue-fill"/><text x="315" y="51" class="blue-text">current</text>
                <path d="M357 72 L430 92" class="arrow teal-stroke"/><circle cx="438" cy="94" r="8" class="dot teal-fill"/><text x="446" y="98" class="teal-text">accept</text>
                <path d="M345 74 L175 175" class="arrow red-stroke"/><circle cx="165" cy="181" r="8" class="dot red-fill"/><text x="105" y="177" class="red-text">reject</text>
                """,
            ),
            (
                "Saved rows are not always independent information",
                "The upper trace moves slowly and keeps nearby values similar. The lower trace changes direction more often. Both can contain the same number of saved rows, but the upper chain has stronger autocorrelation and therefore less effective information.",
                "The experiment compares proposal scales. Read the trace first, then the autocorrelation bars and approximate ESS.",
                """
                <text x="25" y="52" class="label">sticky trace</text><polyline points="140,45 190,50 240,47 290,58 340,55 390,62 440,58 490,66 540,62 590,70 650,68 715,73" class="path red"/>
                <text x="25" y="132" class="label">mobile trace</text><polyline points="140,128 190,95 240,145 290,108 340,160 390,100 440,138 490,92 540,150 590,112 650,142 715,102" class="path teal"/>
                <text x="25" y="211" class="label">ACF</text><rect x="145" y="178" width="22" height="42" class="red-fill"/><rect x="180" y="189" width="22" height="31" class="red-fill"/><rect x="215" y="199" width="22" height="21" class="red-fill"/>
                <rect x="330" y="209" width="22" height="11" class="teal-fill"/><rect x="365" y="214" width="22" height="6" class="teal-fill"/><text x="430" y="210" class="small">less correlation → more effective draws</text>
                """,
            ),
            (
                "A narrow diagonal posterior valley",
                "Each ellipse is a contour of equal posterior density for baseline demand and price effect. The target is long in one direction and narrow in the other. The red random walk wastes many moves across the narrow walls.",
                "The correlation control rotates and narrows this valley. Predict how an isotropic proposal will behave before you run it.",
                """
                <g transform="rotate(-28 390 130)"><ellipse cx="390" cy="130" rx="245" ry="65" class="contour"/><ellipse cx="390" cy="130" rx="175" ry="43" class="contour"/><ellipse cx="390" cy="130" rx="95" ry="23" class="contour"/></g>
                <polyline points="160,190 210,155 265,173 300,136 350,149 395,113 450,126 505,88 555,105 610,72" class="path red"/>
                <text x="38" y="235" class="small">baseline demand</text><text x="25" y="30" class="small">price effect</text>
                """,
            ),
            (
                "The gradient field points uphill",
                "Contours show equal posterior score. Each arrow is the local gradient of log posterior. Arrow direction shows the steepest local increase; arrow length shows slope magnitude. Near the mode, the arrows become short.",
                "You will move one point by the field and compare numerical gradients with automatic differentiation later.",
                """
                <ellipse cx="390" cy="130" rx="230" ry="88" class="contour"/><ellipse cx="390" cy="130" rx="150" ry="57" class="contour"/><ellipse cx="390" cy="130" rx="70" ry="27" class="contour"/>
                <path d="M150 60 L225 82 M160 195 L235 168 M630 55 L555 82 M625 200 L550 170 M390 35 L390 78 M390 225 L390 182 M260 130 L315 130 M520 130 L465 130" class="arrow teal-stroke"/>
                <circle cx="390" cy="130" r="7" class="accent-fill"/><text x="405" y="125" class="label">mode</text>
                """,
            ),
            (
                "Two state updates create HMC motion",
                "The posterior gradient does not move position directly. First it changes the momentum vector r. Then r changes the parameter position θ. The labels r₁ and r₂ name the horizontal and vertical components of that one vector.",
                "The live experiment compares posterior force on with posterior force off. This makes the zero-initial-momentum case testable instead of leaving it as a verbal claim.",
                """
                <rect x="35" y="72" width="190" height="100" rx="14" class="box"/><text x="130" y="103">current position θ</text><text x="130" y="132" class="small">gradient is measured here</text><text x="130" y="154" class="small">start is away from mode</text>
                <path d="M225 122 H300" class="arrow red-stroke"/><text x="262" y="101" class="red-text">∇ log p(θ)</text>
                <rect x="300" y="72" width="190" height="100" rx="14" class="box"/><text x="395" y="103">change momentum r</text><text x="395" y="132" class="small">r = (r₁, r₂)</text><text x="395" y="154" class="small">r₀ can start at zero</text>
                <path d="M490 122 H565" class="arrow teal-stroke"/><text x="528" y="101" class="teal-text">new r</text>
                <rect x="565" y="72" width="180" height="100" rx="14" class="box"/><text x="655" y="103">change position θ</text><text x="655" y="132" class="small">movement now occurs</text><text x="655" y="154" class="small">repeat both updates</text>
                <text x="180" y="218" class="label">zero r₀ + nonzero gradient → nonzero r → changed θ</text>
                """,
            ),
            (
                "Symmetry first; accept or repeat second",
                "A one-sided update uses one force value for the full momentum change. Leapfrog uses half of the old-position force before movement and half of the new-position force after movement. The matching ends make the sequence retrace when the step sign is reversed.",
                "The live experiment compares backward return error and energy error. It then converts the selected leapfrog path's signed energy error into an explicit acceptance probability.",
                """
                <rect x="25" y="42" width="150" height="70" rx="12" class="box"/><text x="100" y="69">½ momentum</text><text x="100" y="93" class="small">force at old q</text>
                <path d="M175 77 H225" class="arrow"/><rect x="225" y="42" width="150" height="70" rx="12" class="box"/><text x="300" y="69">full position</text><text x="300" y="93" class="small">use half-step p</text>
                <path d="M375 77 H425" class="arrow"/><rect x="425" y="42" width="150" height="70" rx="12" class="box"/><text x="500" y="69">½ momentum</text><text x="500" y="93" class="small">force at new q</text>
                <path d="M575 77 H650" class="arrow teal-stroke"/><text x="685" y="69" class="teal-text">reversible</text><text x="685" y="93" class="small">use -ε to undo</text>
                <rect x="110" y="153" width="170" height="70" rx="12" class="box"/><text x="195" y="180">compute ΔH</text><text x="195" y="204" class="small">Hnew - Hold</text>
                <path d="M280 188 H345" class="arrow"/><rect x="345" y="153" width="190" height="70" rx="12" class="box"/><text x="440" y="180">α = min(1, e⁻Δᴴ)</text><text x="440" y="204" class="small">draw u from 0 to 1</text>
                <path d="M535 188 H600" class="arrow"/><text x="675" y="176" class="teal-text">u &lt; α: accept</text><text x="675" y="205" class="red-text">u ≥ α: repeat old q</text>
                """,
            ),
            (
                "One trajectory is not yet an HMC chain",
                "Read each box as one layer. Fresh momentum is random. The leapfrog path is then deterministic. The Metropolis draw accepts the proposed endpoint or repeats the old position. Only that final position enters the chain.",
                "After you lock a prediction, the live experiment will compare complete transitions with one no-refresh path. Time colors and an energy-error inset will then give you evidence for judging the two processes.",
                """
                <rect x="20" y="77" width="125" height="88" rx="12" class="box"/><text x="82" y="105">current qₜ</text><text x="82" y="134" class="small">stored position</text>
                <path d="M145 121 H180" class="arrow"/><rect x="180" y="77" width="135" height="88" rx="12" class="box"/><text x="247" y="105">draw p₀</text><text x="247" y="134" class="small">Normal randomness</text>
                <path d="M315 121 H350" class="arrow"/><rect x="350" y="77" width="145" height="88" rx="12" class="box"/><text x="422" y="105">L leapfrog</text><text x="422" y="134" class="small">deterministic path</text>
                <path d="M495 121 H530" class="arrow"/><rect x="530" y="62" width="190" height="118" rx="12" class="box"/><text x="625" y="91">draw u; compare α</text><text x="625" y="122" class="teal-text">accept: store q*</text><text x="625" y="151" class="red-text">reject: repeat qₜ</text>
                <path d="M720 121 C748 121 748 218 82 218 C48 218 48 183 62 165" class="arrow teal-stroke"/>
                <text x="382" y="245" class="label">repeat complete transitions → ordered HMC chain</text>
                """,
            ),
            (
                "A geometric U-turn",
                "The long curve is an HMC trajectory grown from the start point. The displacement arrow points from the start to the current endpoint. The momentum arrow points in the current travel direction. A negative dot product means the path has started to return.",
                "The experiment changes step size and geometry. Watch where useful forward travel becomes doubling back.",
                """
                <path d="M125 180 C190 70 345 42 500 80 C640 115 650 205 535 210 C455 214 410 178 435 145" class="path teal"/>
                <circle cx="125" cy="180" r="8" class="blue-fill"/><text x="82" y="205" class="label">start</text><circle cx="435" cy="145" r="8" class="accent-fill"/>
                <path d="M125 180 L435 145" class="arrow blue-stroke"/><text x="250" y="145" class="blue-text">displacement</text>
                <path d="M435 145 L380 172" class="arrow red-stroke"/><text x="425" y="188" class="red-text">momentum turns back</text>
                """,
            ),
            (
                "Four chains must tell one stable story",
                "Each row is one chain started from a different point. Good chains lose the memory of their starts and visit the same region. The dashed vertical line marks the end of warmup; diagnostics apply to the stationary sampling region after it.",
                "The experiment includes a broken case. Diagnose trace overlap, cumulative means, R-hat, ESS, and divergences together.",
                """
                <path d="M150 35 V230" class="dash"/><text x="95" y="248" class="small">warmup</text><text x="430" y="248" class="small">sampling regime</text>
                <polyline points="35,55 80,30 120,62 160,48 220,58 280,44 340,62 400,49 460,57 520,46 590,61 660,48 725,55" class="path blue"/>
                <polyline points="35,105 80,138 120,96 160,112 220,101 280,117 340,98 400,113 460,102 520,118 590,99 660,111 725,103" class="path teal"/>
                <polyline points="35,165 80,142 120,178 160,157 220,170 280,153 340,174 400,158 460,169 520,154 590,171 660,156 725,166" class="path purple"/>
                <polyline points="35,215 80,185 120,224 160,205 220,214 280,199 340,220 400,202 460,216 520,198 590,218 660,201 725,211" class="path orange"/>
                """,
            ),
            (
                "A tractable distribution approximates a skewed target",
                "The black curve is a non-Gaussian target posterior. The teal curve is a Gaussian approximation q with adjustable mean and width. One Gaussian can move and stretch, but it cannot reproduce the target's asymmetric tail exactly.",
                "First adjust q by hand. Then run optimization and compare peak location, width, and tail coverage with MCMC samples.",
                """
                <path d="M70 215 H715 M70 215 V35" class="axis"/><path d="M100 214 C185 210 245 70 330 54 C385 44 420 110 470 150 C525 192 590 207 690 214" class="curve ink"/>
                <path d="M185 214 C250 207 300 93 380 72 C460 93 510 207 575 214" class="curve teal"/>
                <text x="260" y="47" class="label">target p</text><text x="425" y="91" class="teal-text">Gaussian q</text><text x="545" y="176" class="label">target tail</text>
                """,
            ),
            (
                "Approximation families fail in recognizable ways",
                "The three panels show a skewed target, two separated modes, and curved dependence. A diagonal Gaussian is symmetric, has one mode, and cannot rotate. A full-covariance Gaussian can rotate one ellipse, but it still cannot bend or split.",
                "Select one geometry in the experiment. Name the missing shape before you look at the fitted approximation.",
                """
                <rect x="20" y="30" width="225" height="190" rx="10" class="panel"/><text x="132" y="52">skew</text><path d="M40 195 C80 190 95 75 135 68 C180 64 185 165 225 195" class="curve ink"/><ellipse cx="132" cy="135" rx="58" ry="28" class="contour teal-stroke"/>
                <rect x="267" y="30" width="225" height="190" rx="10" class="panel"/><text x="379" y="52">two modes</text><path d="M285 195 C305 190 320 95 350 88 C380 95 390 190 405 195 C425 190 435 82 465 76" class="curve ink"/><ellipse cx="380" cy="135" rx="70" ry="32" class="contour teal-stroke"/>
                <rect x="514" y="30" width="225" height="190" rx="10" class="panel"/><text x="626" y="52">banana</text><path d="M545 90 C570 190 660 205 712 120" class="path ink"/><ellipse cx="625" cy="140" rx="74" ry="34" class="contour teal-stroke"/>
                """,
            ),
            (
                "Soft assignments connect the E-step and M-step",
                "Each dot is a customer visit time. Purple and orange show two hidden arrival patterns. Mixed color means uncertain membership. The E-step updates these soft responsibilities; the M-step uses them as weights when it moves component means and weights.",
                "The experiment animates repeated E and M updates. Watch responsibilities first and parameter movement second.",
                """
                <path d="M70 200 H715" class="axis"/><text x="610" y="232" class="small">visit time</text>
                <circle cx="150" cy="175" r="10" class="purple-fill"/><circle cx="205" cy="150" r="10" class="purple-fill"/><circle cx="270" cy="168" r="10" class="blend-fill"/><circle cx="330" cy="130" r="10" class="blend-fill"/><circle cx="410" cy="142" r="10" class="blend-fill"/><circle cx="485" cy="112" r="10" class="orange-fill"/><circle cx="560" cy="125" r="10" class="orange-fill"/><circle cx="630" cy="92" r="10" class="orange-fill"/>
                <path d="M220 190 V70" class="dash purple-stroke"/><path d="M540 190 V70" class="dash orange-stroke"/><text x="165" y="55" class="purple-text">morning mean</text><text x="515" y="55" class="orange-text">afternoon mean</text>
                """,
            ),
            (
                "Inference methods return different objects",
                "Each card names the object that leaves an algorithm. MAP returns one point. EM returns fitted parameters and soft assignments. VI returns a parameterized approximation. MCMC and NUTS return dependent posterior draws plus diagnostics.",
                "The experiment fills a comparison table from actual small runs. Start method selection from the object your decision needs.",
                """
                <rect x="25" y="45" width="130" height="150" rx="12" class="box"/><text x="90" y="75">MAP</text><circle cx="90" cy="125" r="12" class="blue-fill"/><text x="90" y="170" class="small">one point</text>
                <rect x="170" y="45" width="130" height="150" rx="12" class="box"/><text x="235" y="75">EM</text><circle cx="215" cy="120" r="9" class="purple-fill"/><circle cx="255" cy="135" r="9" class="orange-fill"/><text x="235" y="170" class="small">fit + roles</text>
                <rect x="315" y="45" width="130" height="150" rx="12" class="box"/><text x="380" y="75">VI</text><ellipse cx="380" cy="127" rx="42" ry="25" class="contour teal-stroke"/><text x="380" y="170" class="small">approximation q</text>
                <rect x="460" y="45" width="130" height="150" rx="12" class="box"/><text x="525" y="75">MCMC</text><g class="blue-fill"><circle cx="490" cy="115" r="5"/><circle cx="510" cy="135" r="5"/><circle cx="530" cy="112" r="5"/><circle cx="550" cy="140" r="5"/></g><text x="525" y="170" class="small">draws</text>
                <rect x="605" y="45" width="130" height="150" rx="12" class="box"/><text x="670" y="75">NUTS</text><g class="teal-fill"><circle cx="640" cy="115" r="5"/><circle cx="660" cy="135" r="5"/><circle cx="680" cy="112" r="5"/><circle cx="700" cy="140" r="5"/></g><text x="670" y="170" class="small">draws + checks</text>
                """,
            ),
            (
                "Partial pooling moves uncertain branches more",
                "The hollow points are raw branch estimates. The teal line is the shared population mean. Arrows show posterior shrinkage. The branch with little data moves farther because its own estimate is less precise.",
                "The population standard deviation controls how strongly branches can differ. Predict the low-data branch before moving the control.",
                """
                <path d="M120 55 H700 M120 125 H700 M120 195 H700" class="grid"/><text x="25" y="60" class="label">Branch A</text><text x="25" y="130" class="label">Branch B</text><text x="25" y="200" class="label">Branch C</text>
                <path d="M410 25 V225" class="dash teal-stroke"/><text x="420" y="42" class="teal-text">population mean</text>
                <circle cx="500" cy="55" r="10" class="hollow"/><path d="M490 55 L445 55" class="arrow teal-stroke"/><circle cx="440" cy="55" r="8" class="teal-fill"/>
                <circle cx="520" cy="125" r="10" class="hollow"/><path d="M510 125 L455 125" class="arrow teal-stroke"/><circle cx="450" cy="125" r="8" class="teal-fill"/>
                <circle cx="670" cy="195" r="10" class="hollow"/><path d="M660 195 L470 195" class="arrow teal-stroke"/><circle cx="465" cy="195" r="8" class="teal-fill"/>
                """,
            ),
            (
                "A hierarchy turns coupon review counts into purchase evidence",
                "Population nodes describe what coupon rules share. Rule-level latent logits become buyer coverage s and nonbuyer offer rate f. Observed review counts update them. Their ratio s/f carries uncertainty into posterior purchase odds.",
                "The experiment fits the complete model. Read from population to rules to counts, and then from posterior draws to the final odds.",
                """
                <rect x="25" y="80" width="135" height="80" rx="12" class="box"/><text x="92" y="108">population</text><text x="92" y="137" class="small">μ and σ</text><path d="M160 120 H215" class="arrow"/>
                <rect x="215" y="55" width="150" height="130" rx="12" class="box"/><text x="290" y="88">coupon rules</text><text x="290" y="118" class="small">η and ξ</text><text x="290" y="148" class="small">s and f</text><path d="M365 120 H420" class="arrow"/>
                <rect x="420" y="70" width="130" height="100" rx="12" class="box"/><text x="485" y="105">review counts</text><text x="485" y="137" class="small">k / n</text><path d="M550 120 H605" class="arrow"/>
                <rect x="605" y="70" width="130" height="100" rx="12" class="box"/><text x="670" y="103">purchase odds</text><text x="670" y="137" class="small">prior × s/f</text>
                """,
            ),
            (
                "A hierarchical funnel has a wide mouth and a narrow neck",
                "The vertical axis is a top-level log scale. When it is large, lower-level parameters can spread widely. When it is small, they must fit through a very narrow region. One global HMC step size must handle both parts.",
                "The experiment samples this geometry and marks divergences. Look for failures near the narrow neck.",
                """
                <path d="M390 25 C250 75 130 120 105 220 M390 25 C530 75 650 120 675 220" class="curve purple"/><path d="M390 25 V225" class="dash"/>
                <g class="blue-fill" opacity=".72"><circle cx="385" cy="45" r="4"/><circle cx="397" cy="60" r="4"/><circle cx="370" cy="80" r="4"/><circle cx="420" cy="95" r="4"/><circle cx="340" cy="115" r="4"/><circle cx="455" cy="135" r="4"/><circle cx="280" cy="160" r="4"/><circle cx="510" cy="175" r="4"/><circle cx="190" cy="205" r="4"/><circle cx="610" cy="215" r="4"/></g>
                <text x="405" y="48" class="label">neck</text><text x="405" y="215" class="label">wide mouth</text>
                """,
            ),
            (
                "Non-centered coordinates straighten difficult dependence",
                "The left panel shows centered funnel coordinates: scale and group effects are tightly coupled. The right panel shows standard z coordinates: the same model becomes closer to a round cloud. The probability model is unchanged; only coordinates change.",
                "The experiment fits both forms. Compare divergences, ESS, and trace behavior, not only posterior means.",
                """
                <rect x="25" y="30" width="335" height="200" rx="12" class="panel"/><text x="192" y="55">centered</text><path d="M192 70 C120 115 80 150 65 210 M192 70 C265 115 305 150 320 210" class="curve purple"/>
                <rect x="400" y="30" width="335" height="200" rx="12" class="panel"/><text x="567" y="55">non-centered</text><ellipse cx="567" cy="140" rx="92" ry="62" class="contour teal-stroke"/>
                <g class="teal-fill" opacity=".65"><circle cx="520" cy="115" r="5"/><circle cx="555" cy="95" r="5"/><circle cx="600" cy="120" r="5"/><circle cx="535" cy="155" r="5"/><circle cx="580" cy="165" r="5"/><circle cx="615" cy="145" r="5"/></g><path d="M360 130 H400" class="arrow"/>
                """,
            ),
            (
                "Neural-network training is point inference on weights",
                "Inputs flow through a tiny network to a demand prediction. Cross-entropy or squared error supplies a negative log likelihood. L2 adds the penalty implied by a Gaussian prior. SGD or Adam still returns one optimized weight vector.",
                "The experiment changes regularization and compares the ordinary point estimate with the Bayesian interpretation.",
                """
                <circle cx="70" cy="90" r="20" class="hollow"/><circle cx="70" cy="170" r="20" class="hollow"/><text x="70" y="230" class="small">inputs</text>
                <g class="teal-fill"><circle cx="250" cy="65" r="18"/><circle cx="250" cy="130" r="18"/><circle cx="250" cy="195" r="18"/></g><text x="250" y="230" class="small">hidden units</text>
                <circle cx="430" cy="130" r="24" class="blue-fill"/><text x="430" y="175" class="small">demand</text><path d="M90 90 L232 65 M90 90 L232 130 M90 170 L232 130 M90 170 L232 195 M268 65 L408 130 M268 130 L408 130 M268 195 L408 130" class="line"/>
                <path d="M455 130 H530" class="arrow"/><rect x="530" y="75" width="195" height="110" rx="12" class="box"/><text x="627" y="108">training objective</text><text x="627" y="138" class="small">negative log likelihood</text><text x="627" y="163" class="small">+ Gaussian-prior penalty</text>
                """,
            ),
            (
                "Choose a method by following the full inference route",
                "Read from left to right. A decision defines the needed output. The probabilistic model defines latent and observed variables. The target can be a point objective or posterior. The algorithm produces an object that must be checked with suitable diagnostics.",
                "The final challenge gives a new case. Use this route before you name MAP, EM, VI, MCMC, or NUTS.",
                """
                <rect x="20" y="85" width="120" height="70" rx="12" class="box"/><text x="80" y="125">decision</text><path d="M140 120 H170" class="arrow"/>
                <rect x="170" y="85" width="120" height="70" rx="12" class="box"/><text x="230" y="112">probabilistic</text><text x="230" y="137">model</text><path d="M290 120 H320" class="arrow"/>
                <rect x="320" y="85" width="120" height="70" rx="12" class="box"/><text x="380" y="125">target</text><path d="M440 120 H470" class="arrow"/>
                <rect x="470" y="85" width="120" height="70" rx="12" class="box"/><text x="530" y="125">algorithm</text><path d="M590 120 H620" class="arrow"/>
                <rect x="620" y="85" width="120" height="70" rx="12" class="box"/><text x="680" y="112">output and</text><text x="680" y="137">diagnostics</text>
                """,
            ),
        ]
        title, read_text, change_text, drawing = views[index]
        orientation = mo.Html(
            f"""
            <div style="border:1px solid #D9E2EC;border-radius:16px;padding:1rem 1.1rem;background:#FFFFFF">
              <div style="font-size:.76rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#0B6E75">Orientation view · fixed example</div>
              <h3 style="margin:.25rem 0 .8rem;color:#172033">{title}</h3>
              <svg viewBox="0 0 760 260" role="img" aria-label="{title}" style="width:100%;max-height:300px;background:#FCFCFD;border-radius:12px">
                <defs><marker id="arrowhead-{index}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#52606D"/></marker></defs>
                <style>
                  text{{font:16px system-ui;text-anchor:middle;fill:#172033}} .small{{font-size:13px;fill:#52606D}} .label{{font-size:14px}}
                  .axis{{stroke:#52606D;stroke-width:2;fill:none}} .grid{{stroke:#CBD5E1;stroke-width:1.5}} .line{{stroke:#94A3B8;stroke-width:2;fill:none}}
                  .curve{{fill:none;stroke-width:4}} .path{{fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}}
                  .ink{{stroke:#172033}} .teal{{stroke:#0EA5A8}} .blue{{stroke:#2563EB}} .red{{stroke:#DC2626}} .purple{{stroke:#6D5BD0}} .orange{{stroke:#F59E0B}}
                  .box{{fill:#F8FAFC;stroke:#94A3B8;stroke-width:2}} .panel{{fill:#FFFFFF;stroke:#CBD5E1;stroke-width:2}}
                  .contour{{fill:none;stroke:#94A3B8;stroke-width:2}} .fill{{fill:#99F6E4;opacity:.55;stroke:none}} .hollow{{fill:#FFFFFF;stroke:#172033;stroke-width:3}}
                  .dot{{stroke:#FFFFFF;stroke-width:2}} .blue-fill{{fill:#2563EB}} .teal-fill{{fill:#0EA5A8}} .red-fill{{fill:#DC2626}} .purple-fill{{fill:#6D5BD0}} .orange-fill{{fill:#F59E0B}} .accent-fill{{fill:#EC4899}} .blend-fill{{fill:#B76B78}}
                  .blue-text{{fill:#2563EB}} .teal-text{{fill:#0B7F82}} .red-text{{fill:#DC2626}} .purple-text{{fill:#6D5BD0}} .orange-text{{fill:#C77800}}
                  .teal-stroke{{stroke:#0EA5A8}} .blue-stroke{{stroke:#2563EB}} .red-stroke{{stroke:#DC2626}} .purple-stroke{{stroke:#6D5BD0}} .orange-stroke{{stroke:#F59E0B}}
                  .dash{{fill:none;stroke:#64748B;stroke-width:2;stroke-dasharray:7 6}} .arrow{{fill:none;stroke:#52606D;stroke-width:3;marker-end:url(#arrowhead-{index})}}
                </style>
                {drawing}
              </svg>
              <p style="margin:.8rem 0 .3rem"><strong>How to read it.</strong> {read_text}</p>
              <p style="margin:.3rem 0 0"><strong>What the live experiment adds.</strong> {change_text}</p>
            </div>
            """
        )
        if index == 12:
            nuts_deep_dive = mo.md(
                r"""
    ### The U-turn test is only one part of NUTS

    The simple dot-product rule in the main lesson gives the central geometric idea. A complete NUTS transition must solve a harder problem: **How can the sampler choose its path length from the path itself without favoring some states?**

    NUTS uses a balanced binary tree. Here, a **tree** is a record of leapfrog states and their links. It is not a decision tree and it is not a model parameter.

    ### 1. Start one HMC transition

    Begin at the current position $q_0$ and draw fresh momentum $p_0$. Together they define the initial Hamiltonian:

    $$H(q_0,p_0)=U(q_0)+K(p_0).$$

    NUTS uses the same reversible leapfrog step as HMC. The new logic controls how the path grows and how the sampler selects the next stored position.

    ### 2. Grow the path by doubling

    At each tree depth, NUTS randomly selects a time direction: forward or backward. It adds a subtree with $1, 2, 4, 8, \ldots$ new leapfrog steps. The available path length grows quickly, but you do not have to select a fixed number of steps in advance.

    The tree keeps:

    - the left endpoint $(q_-,p_-)$,
    - the right endpoint $(q_+,p_+)$,
    - valid candidate states visited inside the tree,
    - diagnostic values such as energy error.

    The random direction is important. If the algorithm always extended forward, one end of the path would have a special role.

    ### 3. Test both ends for a U-turn

    Define the displacement across the complete tree as:

    $$\Delta q=q_+-q_-.$$

    With mass matrix $M$, momentum becomes velocity through $v=M^{-1}p$. NUTS stops growing when either endpoint points back into the path:

    $$\Delta q^T M^{-1}p_- < 0
    \quad\text{or}\quad
    \Delta q^T M^{-1}p_+ < 0.$$

    For the identity mass matrix, $M=I$, this becomes the dot-product rule from the exercise. The algorithm checks both endpoints because the tree can grow in both time directions. Production implementations also check completed subtrees. An internal part of the path can turn back before the two outer endpoints show the turn clearly.

    ### 4. Select a candidate, not only the final endpoint

    A naive algorithm could stop at the first U-turn and return that endpoint. This creates **endpoint bias** because the state that caused the stop receives a special chance of selection.

    NUTS selects from valid states visited in the tree. Classic NUTS uses a slice variable to define eligible states. Many current implementations use multinomial weights based on Hamiltonian energy. In both cases, the selected state can be inside the path. It does not have to be the final endpoint.

    ### 5. Stop unsafe or excessive growth

    Tree growth stops when one of these conditions occurs:

    1. The endpoint geometry indicates a U-turn.
    2. A large energy error indicates a divergent leapfrog path.
    3. The tree reaches its maximum depth.

    A divergence is not an ordinary early stop. It is evidence that the numerical path could not follow the target geometry accurately. Reaching maximum depth can mean that the allowed path was too short or that the geometry is difficult.

    ### 6. Warmup prepares the geometry

    During warmup, NUTS usually adapts two quantities:

    - **Step size:** Dual averaging searches for a step size with useful acceptance behavior.
    - **Mass matrix:** Adaptation rescales and sometimes rotates momentum so that movement better matches the posterior geometry.

    After warmup, these quantities are fixed while the sampler collects posterior draws. NUTS still builds a new tree and draws fresh momentum for each transition.

    ### A compact mental model

    ```text
    current q
      → draw fresh p
      → select forward or backward
      → add 1, 2, 4, 8, ... leapfrog steps
      → keep valid candidates and check energy
      → check U-turns at both ends and in subtrees
      → stop on a U-turn, divergence, or maximum depth
      → select one valid candidate as the next chain state
    ```

    Fixed-length HMC asks you to select the number of leapfrog steps. NUTS keeps the HMC mechanics, but adapts the useful trajectory length for each transition. The U-turn rule limits repeated travel. The tree and candidate-selection rules prevent the adaptive stopping decision from biasing the posterior samples.
                """
            )
            return mo.vstack(
                [
                    orientation,
                    mo.accordion(
                        {
                            "Deep dive · How NUTS builds and stops a trajectory": nuts_deep_dive,
                        }
                    ),
                ]
            )

        if index != 10:
            return orientation

        deep_dive = mo.md(
            r"""
    ### Start with two exact substeps

    Hamiltonian motion contains two simpler changes. A **kick** changes momentum while position stays fixed. A **drift** changes position while momentum stays fixed. Write them as:

    $$
    K_\epsilon: (q,p)\mapsto(q,\ p+\epsilon\nabla\log\pi(q)),
    $$

    $$
    D_\epsilon: (q,p)\mapsto(q+\epsilon p,\ p).
    $$

    Each substep is easy to compute and has an exact inverse: use the same substep with $-\epsilon$.

    ### Why full kick, then full drift is not self-reversing

    A one-sided step performs this sequence:

    $$
    K_\epsilon\;\longrightarrow\;D_\epsilon.
    $$

    To undo it, you must reverse both the signs **and the order**:

    $$
    D_{-\epsilon}\;\longrightarrow\;K_{-\epsilon}.
    $$

    Calling the same kick-then-drift routine with $-\epsilon$ instead gives $K_{-\epsilon}$ followed by $D_{-\epsilon}$. That is not the inverse. You could write and track a separate drift-then-kick inverse, but the standard HMC proposal would no longer have one simple self-reversing map.

    This point is not about choosing a positive or negative momentum. HMC samples momentum with arbitrary signs, and both update orders can process either sign. The issue is **time reversal**. After a momentum flip, HMC needs the proposal mechanism to retrace the path under the same update rule.

    ### Why the two half kicks solve the ordering problem

    Leapfrog uses a mirrored sequence:

    $$
    K_{\epsilon/2}\;\longrightarrow\;D_\epsilon\;\longrightarrow\;K_{\epsilon/2}.
    $$

    Its inverse is:

    $$
    K_{-\epsilon/2}\;\longrightarrow\;D_{-\epsilon}\;\longrightarrow\;K_{-\epsilon/2}.
    $$

    This has the same half-full-half structure. Thus the same routine can move forward or backward; only the sign of $\epsilon$ changes. This property is called **self-adjointness** or **time symmetry**. Together with volume preservation, it lets the HMC Metropolis correction use the Hamiltonian difference without an additional volume or proposal-density term.

    ### Why leapfrog is also more accurate

    The accuracy gain does not come only from making each kick smaller. It comes from the mirrored composition. The leading one-sided errors from the two ends cancel.

    - Full kick then full drift is a first-order splitting. Its local state error is $O(\epsilon^2)$ per step, and its global state error over a fixed simulated time is $O(\epsilon)$.
    - Leapfrog is a second-order splitting. Its local state error is $O(\epsilon^3)$ per step, and its global state error is $O(\epsilon^2)$.

    For a fixed simulated time, halving $\epsilon$ therefore reduces first-order global error by about a factor of 2, but second-order global error by about a factor of 4. Leapfrog is also symplectic, so its energy error usually oscillates in a bounded region instead of drifting steadily while the integration stays stable.

    ### The practical result for HMC

    Leapfrog gives HMC three useful properties:

    1. The same code can retrace a path by reversing the time direction.
    2. The symmetric split gives second-order accuracy and controlled energy error.
    3. Reversibility and volume preservation make the final Metropolis accept-or-repeat rule simple and valid.

    The Metropolis correction still matters because leapfrog is not exact. It corrects the distribution produced by the remaining numerical energy error; it does not make an arbitrary non-reversible update into valid standard HMC.
            """
        )
        return mo.vstack(
            [
                orientation,
                mo.accordion(
                    {
                        "Deep dive · Why use a symmetric leapfrog step?": deep_dive,
                    }
                ),
            ]
        )


    return (make_orientation_view,)


@app.cell(hide_code=True)
def _(
    CONTROL_SPECS,
    COURSE_PROSE,
    LABS,
    LAB_GUIDES,
    check_exercise,
    httpx,
    json,
    make_orientation_view,
    mo,
    np,
    os,
    plt,
    run_experiment_early,
    run_experiment_late,
):
    OPENROUTER_MODEL = (
        os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.8-flash").strip()
        or "qwen/qwen3.8-flash"
    )


    def get_openrouter_api_key():
        environment_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if environment_key:
            return environment_key
        try:
            from marimo._runtime.context import get_context

            provider = get_context().marimo_config.get("ai", {}).get("openrouter", {})
            configured_key = str(provider.get("api_key", "") or "").strip()
            return configured_key or None
        except Exception:
            return None


    def _openrouter_messages(messages):
        return [
            {"role": message.role, "content": str(message.content)}
            if hasattr(message, "role")
            else {"role": message["role"], "content": str(message["content"])}
            for message in messages
        ]


    def _openrouter_text(response_data):
        error = response_data.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or str(error)
            else:
                message = str(error)
            raise RuntimeError(f"OpenRouter error: {message}")
        choices = response_data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter returned no response choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(block.get("text") or block.get("content") or "")
                for block in content
                if isinstance(block, dict)
            )
        if not isinstance(content, str) or not content.strip():
            finish_reason = choices[0].get("finish_reason") or "unknown"
            raise RuntimeError(
                f"{OPENROUTER_MODEL} returned no text content "
                f"(finish reason: {finish_reason}). Select the feedback button again to retry."
            )
        return content.strip()


    def _call_openrouter(messages, *, structured=False):
        api_key = get_openrouter_api_key()
        if not api_key:
            raise RuntimeError(
                "No OpenRouter key is configured. Add it in Marimo Settings → AI → OpenRouter, "
                "or set OPENROUTER_API_KEY in .env and restart the notebook."
            )
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": _openrouter_messages(messages),
            "temperature": 0.2,
            "max_tokens": 8192,
        }
        if structured:
            payload["provider"] = {"require_parameters": True}
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "learning_feedback",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "what_you_got_right": {"type": "string"},
                            "what_to_improve": {"type": "string"},
                            "next_question": {"type": "string"},
                        },
                        "required": [
                            "summary",
                            "what_you_got_right",
                            "what_to_improve",
                            "next_question",
                        ],
                        "additionalProperties": False,
                    },
                },
            }
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=_openrouter_headers(api_key),
            json=payload,
            timeout=35.0,
        )
        try:
            response_data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"OpenRouter returned HTTP {response.status_code} with invalid JSON."
            ) from exc
        if response.is_error:
            _openrouter_text(response_data)
            response.raise_for_status()
        return _openrouter_text(response_data)


    def _openrouter_headers(api_key):
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ElArkk/learn-inference",
            "X-Title": "Bayesian Inference at the Corner Bakery",
        }


    def _stream_openrouter(messages):
        """Yield response text chunks from a streaming OpenRouter call."""
        api_key = get_openrouter_api_key()
        if not api_key:
            raise RuntimeError(
                "No OpenRouter key is configured. Add it in Marimo Settings → AI → "
                "OpenRouter, or set OPENROUTER_API_KEY in .env and restart the notebook."
            )
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": _openrouter_messages(messages),
            "temperature": 0.2,
            "max_tokens": 8192,
            "stream": True,
        }
        with httpx.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers=_openrouter_headers(api_key),
            json=payload,
            timeout=90.0,
        ) as response:
            if response.is_error:
                response.read()
                _openrouter_text(response.json())
                response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    _openrouter_text(chunk)
                choices = chunk.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    yield delta


    def _plain_tutor_math(expression):
        """Convert a small inline LaTeX expression to readable plain text."""
        import re

        text = expression.strip()
        text = re.sub(r"\\(?:widehat|hat)\{([^{}]+)\}", r"\1-hat", text)
        text = re.sub(r"\\bar\{([^{}]+)\}", r"\1-bar", text)
        text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
        text = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text
        )
        text = text.replace(r"\mathcal{N}", "Normal")
        text = text.replace(r"\mathbb{E}", "E")
        replacements = {
            r"\theta": "θ",
            r"\mu": "μ",
            r"\sigma": "σ",
            r"\tau": "τ",
            r"\phi": "φ",
            r"\rho": "ρ",
            r"\alpha": "α",
            r"\beta": "β",
            r"\lambda": "λ",
            r"\epsilon": "ε",
            r"\Delta": "Δ",
            r"\nabla": "∇",
            r"\pi": "π",
            r"\mid": " given ",
            r"\rightarrow": "→",
            r"\to": "→",
            r"\propto": "∝",
            r"\leq": "≤",
            r"\le": "≤",
            r"\geq": "≥",
            r"\ge": "≥",
            r"\times": "×",
            r"\cdot": "·",
        }
        for latex, plain in replacements.items():
            text = text.replace(latex, plain)
        text = text.replace(r"\left", "").replace(r"\right", "")
        text = re.sub(
            r"\\(?:operatorname|mathrm|text|mathbf|mathit|mathcal)\{([^{}]*)\}",
            r"\1",
            text,
        )
        text = re.sub(r"\\(?:widehat|hat)\s*([A-Za-z])", r"\1-hat", text)
        text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
        text = text.replace("{", "").replace("}", "")
        return re.sub(r"\s+", " ", text).strip()


    def _sanitize_tutor_line(line, in_display, in_code):
        """Sanitize one complete Markdown line and update parser state."""
        import re

        stripped = line.strip()
        if stripped.startswith("```"):
            return line, in_display, not in_code
        if in_code:
            return line, in_display, in_code
        if stripped == "$$":
            return line, not in_display, in_code
        if in_display:
            return line, in_display, in_code

        line = re.sub(
            r"\$\$([^$\n]+?)\$\$",
            lambda match: _plain_tutor_math(match.group(1)),
            line,
        )
        line = re.sub(
            r"(?<!\$)\$([^$\n]+?)\$(?!\$)",
            lambda match: _plain_tutor_math(match.group(1)),
            line,
        )
        line = re.sub(
            r"\\\(([^\n]+?)\\\)",
            lambda match: _plain_tutor_math(match.group(1)),
            line,
        )
        return line, in_display, in_code


    def sanitize_tutor_markdown(markdown):
        """Keep display equations and remove unsupported inline math delimiters."""
        in_display = False
        in_code = False
        output = []
        for line in str(markdown).splitlines(keepends=True):
            clean, in_display, in_code = _sanitize_tutor_line(
                line, in_display, in_code
            )
            output.append(clean)
        return "".join(output)


    def _sanitize_tutor_chunks(chunks):
        """Sanitize streamed output after each complete Markdown line."""
        buffer = ""
        in_display = False
        in_code = False
        for chunk in chunks:
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                clean, in_display, in_code = _sanitize_tutor_line(
                    line, in_display, in_code
                )
                yield clean + "\n"
        if buffer:
            clean, _in_display, _in_code = _sanitize_tutor_line(
                buffer, in_display, in_code
            )
            yield clean


    def parse_feedback_response(raw):
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("The tutor returned an empty feedback response.")
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise RuntimeError("The tutor response did not contain a JSON object.")
            result = json.loads(text[start : end + 1])
        if not isinstance(result, dict):
            raise TypeError("The tutor response was not a JSON object.")
        required = (
            "summary",
            "what_you_got_right",
            "what_to_improve",
            "next_question",
        )
        missing = [
            field
            for field in required
            if not isinstance(result.get(field), str) or not result[field].strip()
        ]
        if missing:
            raise RuntimeError(
                "The tutor response missed required feedback fields: " + ", ".join(missing)
            )
        return {
            field: sanitize_tutor_markdown(result[field].strip())
            for field in required
        }


    _feedback_cache = {}

    _TUTOR_MARKDOWN_RULES = (
        "Markdown rule: Put each mathematical equation in its own display block. "
        "Put a blank line before it, put $$ on a line by itself, put the equation "
        "on the next line, put $$ on a line by itself, and then add another blank "
        "line. Never use inline $...$ math. Never put dollar-delimited math in a "
        "heading, list item, table cell, bold label, or link. In those places, use "
        "plain words or plain Unicode symbols such as theta, θ, sigma, σ, or Delta H. "
        "If a list needs an equation, finish the list item in words and put the "
        "display equation in a separate paragraph below the list."
    )


    def grade_answer(index, answer):
        # Never mark an answer incomplete because it omits a concept from a later lab.
        cache_key = ("curriculum-v2", OPENROUTER_MODEL, index, answer.strip())
        if cache_key in _feedback_cache:
            return _feedback_cache[cache_key]
        guide = LAB_GUIDES[index]
        spec = LABS[index]
        introduced_terms = [
            term
            for chapter in COURSE_PROSE[: index + 1]
            for term, _definition in chapter["terms"]
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful Bayesian inference tutor. The learner is a strong DS/ML engineer "
                    "without deep formal mathematics. Assess reasoning, not spelling or wording. Identify "
                    "the learner's correct idea before correcting errors. Do not reveal the full reference "
                    "answer. Use only concepts listed as introduced through this lab. Never "
                    "mark an answer incomplete because it omits a term first taught in a later "
                    "lab. Treat later concepts as previews, not grading requirements. Use "
                    "concrete language and no more than 180 words in total. "
                    + _TUTOR_MARKDOWN_RULES
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Lab: {spec['title']}\nQuestion: {guide['quiz']}\n"
                    f"Concepts introduced through this lab: {', '.join(introduced_terms)}\n"
                    f"Current chapter overview: {COURSE_PROSE[index]['overview']}\n"
                    f"Reference reasoning: {spec['answer']}\nLearner answer: {answer}"
                ),
            },
        ]
        try:
            raw = _call_openrouter(messages, structured=True)
            result = {
                "ok": True,
                "model": OPENROUTER_MODEL,
                **parse_feedback_response(raw),
            }
            _feedback_cache[cache_key] = result
            return result
        except Exception as exc:
            return {
                "ok": False,
                "model": OPENROUTER_MODEL,
                "error": f"{type(exc).__name__}: {exc}",
            }


    def _function_source_from_text(source, name):
        import ast
        import textwrap

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        candidates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        if not candidates:
            return None
        node = candidates[-1]
        segment = ast.get_source_segment(source, node)
        if not segment:
            return None
        lines = segment.splitlines()
        if node.col_offset:
            prefix = " " * node.col_offset
            lines = [
                lines[0],
                *(line[node.col_offset:] if line.startswith(prefix) else line for line in lines[1:]),
            ]
        return textwrap.dedent("\n".join(lines)).strip()


    def implementation_snapshot(implementations):
        import inspect
        import linecache
        from pathlib import Path

        linecache.checkcache()
        saved_parts = []
        running_parts = []
        try:
            from marimo._runtime.context import get_context

            runtime = get_context()
            notebook_source = Path(runtime.filename).read_text()
            graph = runtime.graph
        except Exception:
            notebook_source = ""
            graph = None
        for implementation in implementations:
            name = getattr(implementation, "__name__", type(implementation).__name__)
            try:
                fallback = inspect.getsource(implementation).strip()
            except Exception:
                try:
                    signature = str(inspect.signature(implementation))
                except Exception:
                    signature = "(...)"
                fallback = f"def {name}{signature}:\n    # Source was not available to the tutor."
            saved = _function_source_from_text(notebook_source, name) or fallback
            running = fallback
            if graph is not None:
                defining_cells = graph.get_defining_cells(name)
                if defining_cells:
                    cell_id = next(iter(defining_cells))
                    running = (
                        _function_source_from_text(graph.cells[cell_id].code, name)
                        or fallback
                    )
            saved_parts.append(saved)
            running_parts.append(running)
        saved_code = "\n\n".join(saved_parts)[:12000]
        running_code = "\n\n".join(running_parts)[:12000]
        return {
            "saved_code": saved_code,
            "running_code": running_code,
            "in_sync": saved_code.strip() == running_code.strip(),
            "from_saved_file": bool(notebook_source),
        }


    def implementation_source(implementations):
        return implementation_snapshot(implementations)["saved_code"]


    def exercise_cells_from_text(source):
        """Map each lab index to the full saved source of its exercise cell."""
        import ast

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
        cells = {}
        for cell in tree.body:
            if not isinstance(cell, ast.FunctionDef):
                continue
            lab_index = next(
                (
                    node.args[0].value
                    for node in ast.walk(cell)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "exercise_ready"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ),
                None,
            )
            if lab_index is not None:
                segment = ast.get_source_segment(source, cell)
                if segment:
                    cells[lab_index] = segment
        return cells


    def saved_exercise_cells():
        from pathlib import Path

        try:
            from marimo._runtime.context import get_context

            source = Path(get_context().filename).read_text()
        except Exception:
            return {}
        return exercise_cells_from_text(source)


    def mentioned_lab(messages, cells):
        """Latest lab the chat refers to, by lab number or exercise function name."""
        import re

        function_labs = {
            match.group(1): index
            for index, cell_source in cells.items()
            for match in re.finditer(r"^\s*def (\w+)", cell_source, re.MULTILINE)
            if match.group(1) != "_"
        }
        for message in reversed(list(messages)):
            content = (
                message.content if hasattr(message, "content") else message["content"]
            )
            text = str(content)
            number = re.search(r"lab\s*#?\s*(\d{1,2})", text, re.IGNORECASE)
            if number and int(number.group(1)) in cells:
                return int(number.group(1))
            for name, index in function_labs.items():
                if name in text:
                    return index
        return None


    _TUTOR_CODE_RULES = (
        "The code comes from the saved notebook file; unsaved editor changes are not "
        "visible to you. Do not write a complete corrected exercise function; guide "
        "with hints and questions first."
    )


    def tutor_lab_context(messages):
        cells = saved_exercise_cells()
        if not cells:
            return ""
        index = mentioned_lab(messages, cells)
        if index is None:
            code = "\n\n".join(cells[i] for i in sorted(cells))
            return (
                "\n\nThe learner did not name one lab, so the saved exercise cell of "
                f"every lab is below. {_TUTOR_CODE_RULES}\n```python\n{code}\n```"
            )
        spec = LABS[index]
        guide = LAB_GUIDES[index]
        return (
            f"\n\nThe learner asks about Lab {index:02d}: {spec['title']}\n"
            f"Coding goal: {guide['mission']}\n"
            f"Exercise task:\n{guide['exercise']}\n"
            f"Key idea: {spec['intuition']}\n"
            f"Math: {spec['math']}\n"
            f"Reflection question: {spec['quiz']}\n\n"
            f"The learner's saved exercise cell for this lab is below. {_TUTOR_CODE_RULES}\n"
            f"```python\n{cells[index]}\n```"
        )


    def implementations_for_test(implementations, supporting_implementations=()):
        """Compile the exact saved learner exercise before the deterministic test."""
        snapshot = implementation_snapshot(implementations)
        supporting_snapshot = (
            implementation_snapshot(supporting_implementations)
            if supporting_implementations
            else {"saved_code": ""}
        )
        namespace = {"np": np}
        try:
            supporting_code = supporting_snapshot["saved_code"]
            if supporting_code:
                exec(
                    compile(supporting_code, "<saved supporting exercise>", "exec"),
                    namespace,
                    namespace,
                )
            exec(
                compile(snapshot["saved_code"], "<saved learner exercise>", "exec"),
                namespace,
                namespace,
            )
            current = tuple(
                namespace[implementation.__name__]
                for implementation in implementations
            )
            return {
                "implementations": current,
                "snapshot": snapshot,
                "error": None,
            }
        except Exception as exc:
            return {
                "implementations": (),
                "snapshot": snapshot,
                "error": f"{type(exc).__name__}: {exc}",
            }


    # Keyed on the exact code and failure text: a new failure always gets fresh
    # feedback, but an unrelated re-render (hint slider, done switch) does not
    # trigger another paid API call.
    _code_feedback_cache = {}


    def grade_code_failure(index, learner_code, test_failure, supporting_code=""):
        # Do not tell the learner to rerun the cell.
        # The saved source was compiled, so do not diagnose a stale-kernel problem.
        cache_key = (
            OPENROUTER_MODEL,
            index,
            learner_code,
            test_failure,
            supporting_code,
        )
        if cache_key in _code_feedback_cache:
            return _code_feedback_cache[cache_key]
        guide = LAB_GUIDES[index]
        spec = LABS[index]
        helper_context = (
            f"\n\nAvailable helper code from earlier labs:\n```python\n{supporting_code}\n```"
            if supporting_code
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful programming tutor for a Bayesian inference course. Diagnose "
                    "the learner's code from the exact deterministic test failure. The local test "
                    "compiled the exact saved code shown below. Do not tell the learner to rerun the "
                    "cell or claim that the kernel is stale. Follow the stated exercise and reuse "
                    "earlier helper functions. Do not recommend scipy.stats or another library shortcut "
                    "when the exercise asks for NumPy or a prior-lab helper. Identify something already "
                    "correct and give one next step. Do not provide a complete corrected function, a "
                    "full solution, or paste-ready answer code. End with one short understanding "
                    "question. Use at most 220 words. "
                    + _TUTOR_MARKDOWN_RULES
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Lab: {spec['title']}\nCoding goal: {guide['mission']}\n"
                    f"Expected task:\n{guide['exercise']}\n\nExact local test result:\n"
                    f"{test_failure}\n\nLearner's current saved code:\n```python\n"
                    f"{learner_code}\n```{helper_context}"
                ),
            },
        ]
        try:
            raw = _call_openrouter(messages, structured=True)
            result = {
                "ok": True,
                "model": OPENROUTER_MODEL,
                **parse_feedback_response(raw),
            }
            _code_feedback_cache[cache_key] = result
            return result
        except Exception as exc:
            return {
                "ok": False,
                "model": OPENROUTER_MODEL,
                "error": f"{type(exc).__name__}: {exc}",
            }


    def tutor_model(messages, _config):
        system = {
            "role": "system",
            "content": (
                "You are the tutor inside a Bayesian inference course set in a small bakery. Use the loop "
                "predict → simulate → inspect → code → explain. Ask one short diagnostic question before "
                "giving a full solution when possible. Explain symbols in plain language. Prefer NumPy, "
                "SciPy, Matplotlib, and Marimo, and follow each exercise's stated restrictions. "
                + _TUTOR_MARKDOWN_RULES
            ),
        }
        system["content"] += tutor_lab_context(messages)
        try:
            # Buffer only to line boundaries, so split LaTeX tokens cannot bypass
            # the deterministic Markdown sanitizer.
            yield from _sanitize_tutor_chunks(
                _stream_openrouter([system, *messages])
            )
        except Exception as exc:
            yield f"\n\nTutor unavailable: {type(exc).__name__}: {exc}"


    def _make_control(spec):
        name, kind, *args = spec
        if kind == "slider":
            start, stop, step, value = args
            return mo.ui.slider(
                start=start,
                stop=stop,
                step=step,
                value=value,
                show_value=True,
                label=name,
                full_width=True,
            )
        options, value = args
        return mo.ui.dropdown(options=options, value=value, label=name, full_width=True)


    def make_lab_ui(index):
        prediction = mo.ui.text_area(
            rows=3, label="Your prediction", full_width=True
        ).form(
            submit_button_label="Lock prediction",
            validate=lambda value: "Write a prediction before you continue."
            if not str(value or "").strip()
            else None,
        )
        answer = mo.ui.text_area(
            rows=4, label="Your explanation", full_width=True
        ).form(
            submit_button_label="Check my reasoning",
            validate=lambda value: "Write an answer before you request feedback."
            if not str(value or "").strip()
            else None,
        )
        elements = {
            "prediction": prediction,
            "run": mo.ui.run_button(label="Run experiment", kind="success", full_width=True),
            "test": mo.ui.run_button(label="Run tests", kind="success", full_width=True),
            "debug": mo.ui.run_button(
                label="Ask tutor about this failure", kind="warn", full_width=True
            ),
            "hint_level": mo.ui.slider(
                0, 3, 1, 0, show_value=True, label="Hints to show"
            ),
            "answer": answer,
            "done": mo.ui.switch(value=False, label=f"Mark Lab {index:02d} complete"),
        }
        elements.update(
            {
                f"control::{spec[0]}": _make_control(spec)
                for spec in CONTROL_SPECS[index]
            }
        )
        return mo.ui.dictionary(elements)


    def exercise_ready(index):
        return mo.callout(
            "Your implementation is saved in the notebook document. Select **Run tests** below. "
            "The test compiles the exact saved code, so it does not depend on a stale function "
            "object. To run only this cell in the live kernel, keep the cursor in it and press "
            "**Cmd+Enter** on macOS or **Ctrl+Enter** on Windows and Linux.",
            kind="success",
            title=f"Lab {index:02d} code saved",
        )


    # Results keyed on the exact control values. A re-render caused by an
    # unrelated widget (hint slider, done switch) reuses the stored result
    # instead of re-running the experiment - important for the PyMC labs,
    # where a NUTS fit blocks the kernel for seconds.
    _experiment_cache = {}


    def _run_experiment_cached(index, values, seed):
        cache_key = (index, seed, tuple(sorted(values.items())))
        if cache_key not in _experiment_cache:
            runner = run_experiment_early if index <= 11 else run_experiment_late
            if len(_experiment_cache) >= 32:
                _experiment_cache.pop(next(iter(_experiment_cache)))
            _experiment_cache[cache_key] = runner(index, values, seed)
        return _experiment_cache[cache_key]


    def render_lab_intro(index, ui):
        spec = LABS[index]
        guide = LAB_GUIDES[index]
        prose = COURSE_PROSE[index]
        controls = {
            control_spec[0]: ui[f"control::{control_spec[0]}"]
            for control_spec in CONTROL_SPECS[index]
        }
        prediction = ui["prediction"].value
        if not ui["run"].value:
            experiment = mo.callout(
                "Write and lock your prediction. Then adjust the controls and run the experiment.",
                kind="info",
                title="Result hidden until you predict",
            )
        elif not prediction:
            experiment = mo.callout(
                "Lock a prediction first. The comparison between prediction and result is part of the lesson.",
                kind="warn",
                title="Prediction required",
            )
        else:
            try:
                values = {name: widget.value for name, widget in controls.items()}
                result = _run_experiment_cached(index, values, 1729 + 101 * index)
                result_parts = []
                if result.figure is not None:
                    result_parts.append(result.figure)
                    # The object still renders; this only frees pyplot's registry.
                    plt.close(result.figure)
                result_parts.append(
                    mo.callout(
                        result.summary,
                        kind="success",
                        title="What the simulation showed",
                    )
                )
                if result.table is not None:
                    result_parts.append(
                        mo.ui.table(result.table, pagination=False, selection=None)
                    )
                result_parts.append(
                    mo.callout(
                        mo.md(guide["prediction_answer"]),
                        kind="info",
                        title="Written answer to the opening prediction",
                    )
                )
                experiment = mo.vstack(result_parts)
            except Exception as exc:
                experiment = mo.callout(
                    f"{type(exc).__name__}: {exc}",
                    kind="danger",
                    title="The experiment stopped safely",
                )

        control_panel = (
            mo.vstack(list(controls.values()))
            if controls
            else mo.md("This orientation lab has no simulation controls.")
        )
        term_lines = "\n".join(
            f"- **{term}:** {definition}" for term, definition in prose["terms"]
        )
        notation_lines = "\n".join(
            f"- **{symbol}:** {meaning}" for symbol, meaning in prose["notation"]
        )
        chapter_opening = (
            [mo.callout(mo.md(guide["chapter_opening"]), kind="info", title="Chapter opening")]
            if guide["chapter_opening"]
            else []
        )
        return mo.vstack(
            [
                mo.Html(f"<a id='lab-{index:02d}'></a>"),
                mo.Html(
                    f"""
                    <div class='lab-hero'>
                      <div style='font-size:.8rem;opacity:.8;text-transform:uppercase;letter-spacing:.12em'>{guide['act']}</div>
                      <h1 style='margin:.2rem 0 .1rem'>{spec['title']}</h1>
                      <div class='lab-loop'>predict → simulate → inspect → code → explain</div>
                    </div>
                    """
                ),
                *chapter_opening,
                mo.md(f"## Chapter overview\n\n{prose['overview']}"),
                mo.md(f"## The bakery story so far\n\n{guide['bridge']}"),
                mo.md(f"## Today's decision\n\n{guide['scenario']}"),
                mo.md(
                    "## What you will learn\n\n"
                    + "\n".join(f"- {goal}" for goal in guide["goals"])
                ),
                mo.md(f"## Technical terms in plain language\n\n{term_lines}"),
                mo.md(f"## Read the notation\n\n{notation_lines}"),
                mo.md(f"## Build the model\n\n{guide['model']}"),
                mo.md(f"## Walk through the mathematics\n\n{prose['math_story']}"),
                mo.accordion({"Compact mathematical statement": mo.md(spec["math"])}),
                mo.md(f"## What to look for\n\n{prose['visual_guide']}"),
                mo.md(
                    "## Meet the visual before you predict\n\nThe view below is a fixed "
                    "example. It teaches the visual language of this lab and does not use your controls."
                ),
                make_orientation_view(index),
                mo.md(f"### How the live experiment extends it\n\n{guide['experiment']}"),
                mo.md(
                    f"## Connection to machine-learning practice\n\n{prose['connection']}"
                ),
                mo.md(
                    f"## 1 · Predict\n\n**Answer this exact question before you run anything:**\n\n"
                    f"{guide['prediction']}"
                ),
                ui["prediction"],
                mo.md(
                    "## 2 · Simulate and inspect\n\nChange one control at a time. State which "
                    "visible result supports or contradicts your prediction."
                ),
                control_panel,
                ui["run"],
                experiment,
                mo.md(f"## 3 · Code — {guide['mission']}\n\n{guide['exercise_intro']}"),
                mo.callout(mo.md(guide["exercise"]), kind="info", title="Your coding task"),
                mo.callout(
                    mo.md(
                        "Edit the real Python cell directly below. It supports package completion, "
                        "function signatures, hover help, and go-to-definition. **Run tests** compiles "
                        "the exact saved code. To update only the live function object, use **Cmd+Enter** "
                        "on macOS or **Ctrl+Enter** on Windows and Linux."
                    ),
                    kind="info",
                    title="How to run your edited code",
                ),
            ]
        )


    def render_lab_wrapup(index, implementations, ui, supporting_implementations=()):
        spec = LABS[index]
        guide = LAB_GUIDES[index]
        tests_requested = bool(ui["test"].value or ui["debug"].value)
        test_kind = None
        test_message = None
        test_bundle = None
        if tests_requested:
            test_bundle = implementations_for_test(
                implementations, supporting_implementations
            )
            if test_bundle["error"]:
                test_kind = "danger"
                test_message = (
                    "FAIL · The current saved exercise code could not be compiled: "
                    + test_bundle["error"]
                )
            else:
                test_kind, test_message = check_exercise(
                    index, test_bundle["implementations"]
                )
            test_output = mo.callout(
                test_message,
                kind=test_kind,
                title=f"Lab {index:02d} test result",
            )
            snapshot = test_bundle["snapshot"]
            if not snapshot["from_saved_file"]:
                test_output = mo.vstack(
                    [
                        test_output,
                        mo.callout(
                            "The notebook file could not be read, so the test used the "
                            "running kernel code instead of the saved code. Save the "
                            "notebook and check the Marimo version if this persists.",
                            kind="warn",
                            title="Test fell back to kernel code",
                        ),
                    ]
                )
            elif not snapshot["in_sync"]:
                test_output = mo.vstack(
                    [
                        test_output,
                        mo.callout(
                            "The saved file and the running kernel differ for this "
                            "exercise. The test used the saved code. Press Cmd+Enter "
                            "(macOS) or Ctrl+Enter in the exercise cell to update the "
                            "kernel too.",
                            kind="warn",
                            title="Saved code and kernel differ",
                        ),
                    ]
                )
        else:
            test_output = mo.callout(
                "Edit the code cell above, then select **Run tests**. The test compiles the exact saved code before checking it.",
                kind="info",
                title="Tests not run",
            )

        code_feedback_parts = []
        if tests_requested and test_kind == "danger":
            code_feedback_parts.append(ui["debug"])
            if not ui["debug"].value:
                code_feedback_parts.append(
                    mo.callout(
                        "Read the exact test failure first. If its cause is still unclear, ask the tutor. "
                        "The tutor gives one next step without giving the reference implementation.",
                        kind="info",
                        title="Optional code feedback",
                    )
                )
            elif not get_openrouter_api_key():
                code_feedback_parts.append(
                    mo.callout(
                        "Add an OpenRouter key in Settings → AI → OpenRouter, then select the code-feedback button again.",
                        kind="warn",
                        title="Code tutor needs a key",
                    )
                )
            else:
                code_snapshot = test_bundle["snapshot"]
                learner_code = code_snapshot["saved_code"]
                supporting_code = implementation_source(supporting_implementations)
                code_result = grade_code_failure(
                    index,
                    learner_code,
                    test_message,
                    supporting_code,
                )
                if code_result["ok"]:
                    code_feedback_parts.extend(
                        [
                            mo.md(f"**Code feedback model:** `{code_result['model']}`"),
                            mo.callout(code_result["summary"], kind="warn", title="Why the test failed"),
                            mo.callout(code_result["what_you_got_right"], kind="info", title="What already works"),
                            mo.callout(code_result["what_to_improve"], kind="success", title="Your next implementation step"),
                            mo.callout(code_result["next_question"], kind="neutral", title="Question before your next edit"),
                        ]
                    )
                else:
                    code_feedback_parts.append(
                        mo.callout(
                            f"Model: `{code_result['model']}`\n\n{code_result['error']}\n\n"
                            "Your code is unchanged. Select **Ask tutor about this failure** again to retry.",
                            kind="danger",
                            title="Code feedback request failed",
                        )
                    )

        hint_level = int(ui["hint_level"].value)
        hint_parts = [ui["hint_level"]]
        hint_parts.extend(
            mo.callout(spec["hints"][hint], kind="info", title=f"Hint {hint + 1}")
            for hint in range(hint_level)
        )
        if tests_requested:
            fence = chr(96) * 3
            solution = mo.accordion(
                {"Reveal reference solution": mo.md(f"{fence}python\n{spec['solution']}\n{fence}")}
            )
        else:
            solution = mo.md("The reference solution becomes available after your first test run.")

        submitted_answer = ui["answer"].value
        reference_parts = []
        conclusion_parts = []
        if not submitted_answer:
            feedback = mo.callout(
                "Submit your explanation to receive feedback here in the notebook.",
                kind="info",
                title="No explanation submitted",
            )
        elif not get_openrouter_api_key():
            feedback = mo.callout(
                "Add an OpenRouter key in Settings → AI → OpenRouter, or put OPENROUTER_API_KEY in .env and restart.",
                kind="warn",
                title="Model feedback needs a key",
            )
            reference_parts = [mo.accordion({"Reference explanation": mo.md(spec["answer"])})]
        else:
            result = grade_answer(index, submitted_answer.strip())
            if result["ok"]:
                feedback = mo.vstack(
                    [
                        mo.md(f"**AI feedback model:** `{result['model']}`"),
                        mo.callout(result["summary"], kind="success", title="Tutor assessment"),
                        mo.callout(result["what_you_got_right"], kind="info", title="What is correct"),
                        mo.callout(result["what_to_improve"], kind="warn", title="What to improve"),
                        mo.callout(result["next_question"], kind="neutral", title="One question to test your model"),
                    ]
                )
            else:
                feedback = mo.callout(
                    f"Model: `{result['model']}`\n\n{result['error']}\n\nYour answer is saved. Select **Check my reasoning** again to retry.",
                    kind="danger",
                    title="Feedback request failed",
                )
            reference_parts = [mo.accordion({"Reference explanation": mo.md(spec["answer"])})]
        if submitted_answer:
            conclusion_parts = [
                mo.callout(mo.md(spec["takeaway"]), kind="success", title="What this lab established"),
                mo.callout(mo.md(guide["next"]), kind="neutral", title="Why the next lab follows"),
            ]

        return mo.vstack(
            [
                mo.md("## 4 · Test your code"),
                ui["test"],
                test_output,
                *code_feedback_parts,
                mo.md("### Progressive hints"),
                *hint_parts,
                solution,
                mo.md(
                    f"## 5 · Explain\n\n**Use the model and experiment to answer this question.**\n\n{guide['quiz']}"
                ),
                ui["answer"],
                feedback,
                *reference_parts,
                *conclusion_parts,
                ui["done"],
                mo.md("[Back to the course map](#course-map)"),
                mo.md("---"),
            ]
        )


    tutor_chat = mo.ui.chat(
        tutor_model,
        prompts=[
            "Help me predict what this experiment will do.",
            "Ask me a question that checks my current mental model.",
            "Help me debug my current exercise without giving the full solution.",
        ],
        max_height=520,
    )

    (
        lab00_ui,
        lab01_ui,
        lab02_ui,
        lab03_ui,
        lab04_ui,
        lab05_ui,
        lab06_ui,
        lab07_ui,
        lab08_ui,
        lab09_ui,
        lab10_ui,
        lab11_ui,
        lab12_ui,
        lab13_ui,
        lab14_ui,
        lab15_ui,
        lab16_ui,
        lab17_ui,
        lab18_ui,
        lab19_ui,
        lab20_ui,
        lab21_ui,
        lab22_ui,
        lab23_ui,
    ) = tuple(make_lab_ui(index) for index in range(24))
    return (
        OPENROUTER_MODEL,
        exercise_ready,
        get_openrouter_api_key,
        lab00_ui,
        lab01_ui,
        lab02_ui,
        lab03_ui,
        lab04_ui,
        lab05_ui,
        lab06_ui,
        lab07_ui,
        lab08_ui,
        lab09_ui,
        lab10_ui,
        lab11_ui,
        lab12_ui,
        lab13_ui,
        lab14_ui,
        lab15_ui,
        lab16_ui,
        lab17_ui,
        lab18_ui,
        lab19_ui,
        lab20_ui,
        lab21_ui,
        lab22_ui,
        lab23_ui,
        render_lab_intro,
        render_lab_wrapup,
        tutor_chat,
    )


@app.cell(hide_code=True)
def _(mo):
    sidebar_width = mo.ui.slider(
        320,
        720,
        step=10,
        value=390,
        label="Sidebar width",
        show_value=True,
    )
    return (sidebar_width,)


@app.cell(hide_code=True)
def _(
    OPENROUTER_MODEL,
    get_openrouter_api_key,
    lab00_ui,
    lab01_ui,
    lab02_ui,
    lab03_ui,
    lab04_ui,
    lab05_ui,
    lab06_ui,
    lab07_ui,
    lab08_ui,
    lab09_ui,
    lab10_ui,
    lab11_ui,
    lab12_ui,
    lab13_ui,
    lab14_ui,
    lab15_ui,
    lab16_ui,
    lab17_ui,
    lab18_ui,
    lab19_ui,
    lab20_ui,
    lab21_ui,
    lab22_ui,
    lab23_ui,
    mo,
    sidebar_width,
    tutor_chat,
):
    all_lab_ui = (
        lab00_ui, lab01_ui, lab02_ui, lab03_ui, lab04_ui, lab05_ui,
        lab06_ui, lab07_ui, lab08_ui, lab09_ui, lab10_ui, lab11_ui,
        lab12_ui, lab13_ui, lab14_ui, lab15_ui, lab16_ui, lab17_ui,
        lab18_ui, lab19_ui, lab20_ui, lab21_ui, lab22_ui, lab23_ui,
    )
    completed_labs = sum(bool(ui["done"].value) for ui in all_lab_ui)
    key_status = (
        "OpenRouter is configured."
        if get_openrouter_api_key()
        else "Add a key in Settings → AI → OpenRouter to enable tutor feedback."
    )
    navigation = " ".join(f"[{index:02d}](#lab-{index:02d})" for index in range(24))
    mo.sidebar(
        mo.vstack(
            [
                mo.md("## Inference Lab"),
                sidebar_width,
                mo.md(f"**Progress:** {completed_labs}/24 labs"),
                mo.Html(f"<progress value='{completed_labs}' max='24' style='width:100%'></progress>"),
                mo.md(f"**Labs**  \n{navigation}"),
                mo.callout(key_status, kind="success" if get_openrouter_api_key() else "info", title="Tutor status"),
                mo.md("## Ask the tutor"),
                tutor_chat,
            ]
        ),
        footer=mo.md(f"Model: `{OPENROUTER_MODEL}` · Seed: `1729`"),
        width=f"{sidebar_width.value}px",
    )
    return


@app.cell(hide_code=True)
def _(get_openrouter_api_key, mo):
    _tutor_is_ready = bool(get_openrouter_api_key())
    _tutor_status = mo.callout(
        (
            "The course tutor is ready. Your API key is not shown in the notebook."
            if _tutor_is_ready
            else "The course tutor is off. This does not block any lesson, simulation, "
            "or local code test. Follow the steps below if you want AI feedback."
        ),
        kind="success" if _tutor_is_ready else "info",
        title="Tutor status",
    )
    mo.vstack(
        [
            mo.Html("<a id='course-map'></a>"),
            mo.md(
                """
                # Inference Lab

                ## Bayesian inference and optimization by experiment

                This is an interactive textbook and a small computational workshop. You will
                build one mental model across optimization, posterior sampling, approximate
                inference, latent-variable models, and hierarchical models. The examples stay
                small enough that you can inspect the complete system.

                **The loop is always:** predict → simulate → inspect → code → explain.

                Work from top to bottom. Each coding task is a real Marimo cell. Thus, `np.`,
                hover help, function signatures, and go-to-definition work on the same page as
                the lesson.

                ## Before Lab 0: start the course once

                1. Check that the notebook name in the editor is **`my_lab.py`**. This is your
                   private working copy. Do not write solutions in the clean
                   **`inference_lab.py`** course file.
                2. Run all cells once before you use the controls. In the top toolbar, select
                   the triangular play control named **Run all stale cells**. Wait until no cell
                   is running. This loads the shared functions, plots, tests, and tutor controls.
                   Do this again after each fresh kernel start.
                3. Marimo runs cells by dependency, not only by their position on the page.
                   After the first full run, a slider or an edited cell reruns only the cells
                   that depend on it.
                4. A `TODO` function can be incomplete during this first run. The course keeps
                   experiments separate from learner code, so one unfinished exercise must not
                   stop later lessons from loading.

                ## Optional tutor setup

                All plots and deterministic code tests work without an API key. The key enables
                written-answer feedback, help for a failed exercise, and the tutor in the course
                sidebar.

                **Recommended setup in Marimo**

                1. Create your own key in the OpenRouter dashboard.
                2. Open the Marimo **Settings** menu.
                3. Go to **AI → AI Providers → OpenRouter**.
                4. Paste the key into the OpenRouter API-key field and save the setting.
                5. Select **Run all stale cells** again. If the tutor status does not change,
                   stop and restart `./scripts/run-local.sh` once.

                **Project-only alternative**

                In the project folder, copy `.env.example` to `.env`, set
                `OPENROUTER_API_KEY=your-key`, and restart the local notebook. The `.env` file is
                ignored by Git. Never paste an API key into a Python or Markdown cell.
                """
            ),
            _tutor_status,
            mo.md(
                """
                ## Course map

                1. **Landscapes and points:** density, likelihood, posterior, MLE, MAP, gradients.
                2. **Sampling geometry:** Metropolis, autocorrelation, HMC, leapfrog, NUTS, diagnostics.
                3. **Approximate and latent inference:** VI, EM, and their failure modes.
                4. **Hierarchies:** partial pooling, coupon models, funnels, and parameterization.
                5. **Neural networks and method choice:** the same gradients, different learned objects.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(lab00_ui, render_lab_intro):
    render_lab_intro(0, lab00_ui)
    return


@app.cell
def _(exercise_ready):
    def learning_loop():
        # TODO: return the five stages in order
        ...

    exercise_ready(0)

    return (learning_loop,)


@app.cell(hide_code=True)
def _(lab00_ui, learning_loop, render_lab_wrapup):
    render_lab_wrapup(0, (learning_loop,), lab00_ui)
    return


@app.cell(hide_code=True)
def _(lab01_ui, render_lab_intro):
    render_lab_intro(1, lab01_ui)
    return


@app.cell
def _(exercise_ready):
    def normal_log_density(x, mean, sd):
        # TODO: use NumPy; do not use scipy.stats
        ...

    exercise_ready(1)

    return (normal_log_density,)


@app.cell(hide_code=True)
def _(lab01_ui, normal_log_density, render_lab_wrapup):
    render_lab_wrapup(1, (normal_log_density,), lab01_ui)
    return


@app.cell(hide_code=True)
def _(lab02_ui, render_lab_intro):
    render_lab_intro(2, lab02_ui)
    return


@app.cell
def _(exercise_ready):
    def log_likelihood(mu, x, sigma):
        # TODO: sum Normal log densities
        ...

    def log_posterior(mu, x, sigma, prior_mu, prior_sigma):
        # TODO: likelihood + log prior
        ...

    exercise_ready(2)

    return log_likelihood, log_posterior


@app.cell(hide_code=True)
def _(
    lab02_ui,
    log_likelihood,
    log_posterior,
    normal_log_density,
    render_lab_wrapup,
):
    render_lab_wrapup(
        2,
        (log_likelihood, log_posterior),
        lab02_ui,
        (normal_log_density,),
    )
    return


@app.cell(hide_code=True)
def _(lab03_ui, render_lab_intro):
    render_lab_intro(3, lab03_ui)
    return


@app.cell
def _(exercise_ready, np):
    def gradient_ascent(fn, start, rate, steps):
        x = float(start)
        path = [x]
        for _ in range(steps):
            # TODO: finite-difference gradient and update x
            ...
        return np.asarray(path)

    exercise_ready(3)

    return (gradient_ascent,)


@app.cell(hide_code=True)
def _(gradient_ascent, lab03_ui, render_lab_wrapup):
    render_lab_wrapup(3, (gradient_ascent,), lab03_ui)
    return


@app.cell(hide_code=True)
def _(lab04_ui, render_lab_intro):
    render_lab_intro(4, lab04_ui)
    return


@app.cell
def _(exercise_ready):
    def laplace_sd(log_posterior, map_value, eps=1e-3):
        # TODO: use local second curvature
        ...

    exercise_ready(4)

    return (laplace_sd,)


@app.cell(hide_code=True)
def _(lab04_ui, laplace_sd, render_lab_wrapup):
    render_lab_wrapup(4, (laplace_sd,), lab04_ui)
    return


@app.cell(hide_code=True)
def _(lab05_ui, render_lab_intro):
    render_lab_intro(5, lab05_ui)
    return


@app.cell
def _(exercise_ready, np):
    def metropolis(log_target, start, proposal_sd, draws, rng):
        current=float(start); samples=[]; accepted=0
        for _ in range(draws):
            # TODO: propose, compare log ratio, and save current
            ...
        return np.asarray(samples), accepted/draws

    exercise_ready(5)

    return (metropolis,)


@app.cell(hide_code=True)
def _(lab05_ui, metropolis, render_lab_wrapup):
    render_lab_wrapup(5, (metropolis,), lab05_ui)
    return


@app.cell(hide_code=True)
def _(lab06_ui, render_lab_intro):
    render_lab_intro(6, lab06_ui)
    return


@app.cell
def _(exercise_ready, np):
    def autocorrelation(x, lag):
        x=np.asarray(x)-np.mean(x)
        # TODO: normalized lagged dot product
        ...

    exercise_ready(6)

    return (autocorrelation,)


@app.cell(hide_code=True)
def _(autocorrelation, lab06_ui, render_lab_wrapup):
    render_lab_wrapup(6, (autocorrelation,), lab06_ui)
    return


@app.cell(hide_code=True)
def _(lab07_ui, render_lab_intro):
    render_lab_intro(7, lab07_ui)
    return


@app.cell
def _(exercise_ready):
    def covariance(rho):
        # TODO: return a 2×2 correlation matrix
        ...

    exercise_ready(7)

    return (covariance,)


@app.cell(hide_code=True)
def _(covariance, lab07_ui, render_lab_wrapup):
    render_lab_wrapup(7, (covariance,), lab07_ui)
    return


@app.cell(hide_code=True)
def _(lab08_ui, render_lab_intro):
    render_lab_intro(8, lab08_ui)
    return


@app.cell
def _(exercise_ready, np):
    def numerical_gradient(fn, point, eps=1e-5):
        point=np.asarray(point,dtype=float)
        # TODO: one centered difference per coordinate
        ...

    exercise_ready(8)

    return (numerical_gradient,)


@app.cell(hide_code=True)
def _(lab08_ui, numerical_gradient, render_lab_wrapup):
    render_lab_wrapup(8, (numerical_gradient,), lab08_ui)
    return


@app.cell(hide_code=True)
def _(lab09_ui, render_lab_intro):
    render_lab_intro(9, lab09_ui)
    return


@app.cell
def _(exercise_ready):
    def energy(log_target, position, momentum):
        # TODO: potential plus kinetic energy
        ...

    exercise_ready(9)

    return (energy,)


@app.cell(hide_code=True)
def _(energy, lab09_ui, render_lab_wrapup):
    render_lab_wrapup(9, (energy,), lab09_ui)
    return


@app.cell(hide_code=True)
def _(lab10_ui, render_lab_intro):
    render_lab_intro(10, lab10_ui)
    return


@app.cell
def _(exercise_ready):
    def leapfrog_step(q,p,step,grad_logp):
        # TODO: half p, full q, half p
        ...

    exercise_ready(10)

    return (leapfrog_step,)


@app.cell(hide_code=True)
def _(lab10_ui, leapfrog_step, render_lab_wrapup):
    render_lab_wrapup(10, (leapfrog_step,), lab10_ui)
    return


@app.cell(hide_code=True)
def _(lab11_ui, render_lab_intro):
    render_lab_intro(11, lab11_ui)
    return


@app.cell
def _(exercise_ready):
    def one_hmc_transition(position, log_target, integrate, rng):
        # TODO: sample momentum, integrate, compare H, accept or repeat
        ...

    exercise_ready(11)

    return (one_hmc_transition,)


@app.cell(hide_code=True)
def _(lab11_ui, one_hmc_transition, render_lab_wrapup):
    render_lab_wrapup(11, (one_hmc_transition,), lab11_ui)
    return


@app.cell(hide_code=True)
def _(lab12_ui, render_lab_intro):
    render_lab_intro(12, lab12_ui)
    return


@app.cell
def _(exercise_ready):
    def is_uturn(start, current, momentum):
        # TODO: use a dot product
        ...

    exercise_ready(12)

    return (is_uturn,)


@app.cell(hide_code=True)
def _(is_uturn, lab12_ui, render_lab_wrapup):
    render_lab_wrapup(12, (is_uturn,), lab12_ui)
    return


@app.cell(hide_code=True)
def _(lab13_ui, render_lab_intro):
    render_lab_intro(13, lab13_ui)
    return


@app.cell
def _(exercise_ready, np):
    def basic_rhat(chains):
        chains=np.asarray(chains,dtype=float)
        # TODO: compare between-chain and within-chain variance
        ...

    exercise_ready(13)

    return (basic_rhat,)


@app.cell(hide_code=True)
def _(basic_rhat, lab13_ui, render_lab_wrapup):
    render_lab_wrapup(13, (basic_rhat,), lab13_ui)
    return


@app.cell(hide_code=True)
def _(lab14_ui, render_lab_intro):
    render_lab_intro(14, lab14_ui)
    return


@app.cell
def _(exercise_ready):
    def mc_elbo(log_joint, mean, log_sd, eps):
        # TODO: sample by reparameterization and average log p - log q
        ...

    exercise_ready(14)

    return (mc_elbo,)


@app.cell(hide_code=True)
def _(lab14_ui, mc_elbo, render_lab_wrapup):
    render_lab_wrapup(14, (mc_elbo,), lab14_ui)
    return


@app.cell(hide_code=True)
def _(lab15_ui, render_lab_intro):
    render_lab_intro(15, lab15_ui)
    return


@app.cell
def _(exercise_ready):
    def diagonal_gaussian_sample(mean, log_sd, eps):
        # TODO: reparameterized samples
        ...

    exercise_ready(15)

    return (diagonal_gaussian_sample,)


@app.cell(hide_code=True)
def _(diagonal_gaussian_sample, lab15_ui, render_lab_wrapup):
    render_lab_wrapup(15, (diagonal_gaussian_sample,), lab15_ui)
    return


@app.cell(hide_code=True)
def _(lab16_ui, render_lab_intro):
    render_lab_intro(16, lab16_ui)
    return


@app.cell
def _(exercise_ready):
    def m_step(x, responsibilities):
        # TODO: update component weights and means
        ...

    exercise_ready(16)

    return (m_step,)


@app.cell(hide_code=True)
def _(lab16_ui, m_step, render_lab_wrapup):
    render_lab_wrapup(16, (m_step,), lab16_ui)
    return


@app.cell(hide_code=True)
def _(lab17_ui, render_lab_intro):
    render_lab_intro(17, lab17_ui)
    return


@app.cell
def _(exercise_ready):
    def output_object(method):
        # TODO: map MAP, EM, VI, and MCMC to their output object
        ...

    exercise_ready(17)

    return (output_object,)


@app.cell(hide_code=True)
def _(lab17_ui, output_object, render_lab_wrapup):
    render_lab_wrapup(17, (output_object,), lab17_ui)
    return


@app.cell(hide_code=True)
def _(lab18_ui, render_lab_intro):
    render_lab_intro(18, lab18_ui)
    return


@app.cell
def _(exercise_ready):
    def shrink(raw, se, population_mean, population_sd):
        # TODO: Normal-Normal shrinkage mean
        ...

    exercise_ready(18)

    return (shrink,)


@app.cell(hide_code=True)
def _(lab18_ui, render_lab_wrapup, shrink):
    render_lab_wrapup(18, (shrink,), lab18_ui)
    return


@app.cell(hide_code=True)
def _(lab19_ui, render_lab_intro):
    render_lab_intro(19, lab19_ui)
    return


@app.cell
def _(exercise_ready):
    def classify_variables():
        # TODO: return observed, latent, and deterministic names
        ...

    exercise_ready(19)

    return (classify_variables,)


@app.cell(hide_code=True)
def _(classify_variables, lab19_ui, render_lab_wrapup):
    render_lab_wrapup(19, (classify_variables,), lab19_ui)
    return


@app.cell(hide_code=True)
def _(lab20_ui, render_lab_intro):
    render_lab_intro(20, lab20_ui)
    return


@app.cell
def _(exercise_ready):
    def funnel_sample(v, z):
        # TODO: centered x with scale exp(v/2)
        ...

    exercise_ready(20)

    return (funnel_sample,)


@app.cell(hide_code=True)
def _(funnel_sample, lab20_ui, render_lab_wrapup):
    render_lab_wrapup(20, (funnel_sample,), lab20_ui)
    return


@app.cell(hide_code=True)
def _(lab21_ui, render_lab_intro):
    render_lab_intro(21, lab21_ui)
    return


@app.cell
def _(exercise_ready):
    def noncenter(mu, sigma, z):
        # TODO: deterministic transform
        ...

    exercise_ready(21)

    return (noncenter,)


@app.cell(hide_code=True)
def _(lab21_ui, noncenter, render_lab_wrapup):
    render_lab_wrapup(21, (noncenter,), lab21_ui)
    return


@app.cell(hide_code=True)
def _(lab22_ui, render_lab_intro):
    render_lab_intro(22, lab22_ui)
    return


@app.cell
def _(exercise_ready):
    def map_loss(mse, weights, l2):
        # TODO: data loss plus Gaussian-prior penalty
        ...

    exercise_ready(22)

    return (map_loss,)


@app.cell(hide_code=True)
def _(lab22_ui, map_loss, render_lab_wrapup):
    render_lab_wrapup(22, (map_loss,), lab22_ui)
    return


@app.cell(hide_code=True)
def _(lab23_ui, render_lab_intro):
    render_lab_intro(23, lab23_ui)
    return


@app.cell
def _(exercise_ready):
    def choose_method(needs_uncertainty, latent_mixture, posterior_hard):
        # TODO: return MAP, EM, VI, or NUTS
        ...

    exercise_ready(23)

    return (choose_method,)


@app.cell(hide_code=True)
def _(choose_method, lab23_ui, render_lab_wrapup):
    render_lab_wrapup(23, (choose_method,), lab23_ui)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final concept map

    ```text
    probabilistic model p(data, parameters)
                 │
                 └── posterior or objective
                           │
                           ├── MAP: optimize one point
                           ├── EM: alternate latent expectations and point updates
                           ├── VI: optimize an approximate distribution
                           └── MCMC: sample the posterior
                                      └── HMC and NUTS use gradients and momentum

    neural-network training uses the same machinery
                 ├── loss = negative log likelihood
                 ├── L2 = Gaussian-prior MAP view
                 ├── SGD = point optimization
                 └── Bayesian network = posterior over weights
    ```

    The gradient is shared machinery. The learned object and the update rule decide
    whether the algorithm finds one point, fits a distribution, or samples a distribution.
    """)
    return


@app.cell(hide_code=True)
def _():
    COURSE_PROSE = ({'overview': '\n'
                  '        This course starts with a practical habit, not with a theorem. '
                  'Each morning, the bakery must make a decision before it knows the next '
                  'day’s sales. A number such as “60 loaves” can be useful, but it hides '
                  'the uncertainty that creates both waste and stockouts. Inference is the '
                  'process of using a model and observed evidence to improve that '
                  'uncertain picture. The course repeatedly asks you to state what you '
                  'expect, change a small system, inspect what happened, implement one '
                  'mechanism, and explain the result. That repetition is deliberate: it '
                  'turns definitions into working intuitions that you can transfer to '
                  'unfamiliar models.\n'
                  '        ',
      'terms': (('Inference',
                 'A procedure that uses observed evidence to learn about an unknown '
                 'quantity.'),
                ('Model',
                 'A simplified data-generating story that states what is known, what is '
                 'unknown, and how they are related.'),
                ('Evidence',
                 'The observations available to the learner, such as recorded daily bread '
                 'sales.'),
                ('Diagnostic',
                 'A check that can reveal when an algorithm or conclusion is not '
                 'trustworthy.')),
      'notation': (('D',
                    'the observed data; later this can be a vector of many sales records'),
                   ('θ',
                    'a generic unknown parameter; its concrete meaning changes from lab to '
                    'lab'),
                   ('p(θ | D)',
                    'the uncertainty about θ after the data D are taken into account')),
      'math_story': '\n'
                    '        Mathematical notation is compressed language. The vertical '
                    'bar in $p(\\theta\\mid D)$ means “given.” Thus, read the expression '
                    'as “our distribution for the unknown quantity $\\theta$, given the '
                    'observed data $D$.” A distribution does not claim that the parameter '
                    'changes randomly from minute to minute. It records which values '
                    'remain plausible under the model and evidence. Later algorithms will '
                    'either choose one value of $\\theta$, approximate the full '
                    'distribution, or generate samples whose frequency represents it. Keep '
                    'asking which object an algorithm returns; this question prevents many '
                    'category errors.\n'
                    '        ',
      'visual_guide': 'The fixed course map shows a loop. Read it as a process with '
                      'memory: your written prediction records the model in your head, and '
                      'the result tells you what part of that model needs revision.',
      'connection': 'The same loop is used when debugging a neural network: predict how a '
                    'change should affect training, run it, inspect diagnostics, change '
                    'code, and explain the mechanism.'},
     {'overview': '\n'
                  '        Tomorrow’s demand is not yet observed, so one exact sales value '
                  'would express false certainty. We instead represent possible sales '
                  'values by a distribution. Imagine spreading one unit of probability '
                  'mass across a number line. Regions where demand is more plausible '
                  'receive more mass. A density curve shows how tightly that mass is '
                  'concentrated at each location, but its height at one point is not '
                  'itself a probability. Operational decisions always concern intervals: '
                  'the chance of selling fewer than 50 loaves, between 50 and 70, or more '
                  'than the oven can produce.\n'
                  '        ',
      'terms': (('Random variable',
                 'A named numerical outcome that is not known yet; here it is tomorrow’s '
                 'demand.'),
                ('Density',
                 'Probability concentration per unit on the horizontal axis, represented '
                 'by curve height.'),
                ('Probability mass',
                 'Area under the density over an interval; this is an actual probability.'),
                ('Standard deviation',
                 'A scale that describes the typical spread of outcomes around the mean.')),
      'notation': (('X', 'tomorrow’s number of loaves sold'),
                   ('μ', 'the center or mean of the demand distribution'),
                   ('σ',
                    'the positive standard deviation; larger values spread mass more '
                    'widely'),
                   ('p(x)',
                    'density height at candidate outcome x, not probability at an exact '
                    'point')),
      'math_story': '\n'
                    '        We write $X\\sim\\mathcal N(\\mu,\\sigma)$ to say that the '
                    'possible value $X$ follows a Normal distribution with center $\\mu$ '
                    'and spread $\\sigma$. Its density is\n'
                    '\n'
                    '        '
                    '$$p(x)=\\frac{1}{\\sigma\\sqrt{2\\pi}}\\exp\\!\\left[-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2\\right].$$\n'
                    '\n'
                    '        The factor $1/\\sigma$ explains why the peak becomes lower '
                    'when $\\sigma$ grows. The curve must keep total area one, so a wider '
                    'curve must be shorter. The probability of an interval $[a,b]$ is '
                    '$P(a\\le X\\le b)=\\int_a^b p(x)\\,dx$. This area depends on both '
                    'density height and interval width.\n'
                    '        ',
      'visual_guide': 'Read the horizontal axis as possible sales totals and the vertical '
                      'axis as concentration per loaf. Compare shaded area, not only peak '
                      'height. The whole curve always has area one.',
      'connection': 'A neural network can also output a distribution instead of one '
                    'prediction. The distinction between a density and interval '
                    'probability remains the same.'},
     {'overview': '\n'
                  '        A distribution becomes an inference problem when part of it is '
                  'unknown. The bakery has observed several sales days and wants to learn '
                  'the underlying average demand $\\mu$. For each candidate value of '
                  '$\\mu$, the likelihood asks how well that value would explain the fixed '
                  'records. The prior describes plausible values before these records are '
                  'used. Bayesian updating multiplies those two landscapes and then '
                  'normalizes the result. The posterior is therefore neither “the data” '
                  'nor “the prior”; it is the model’s revised uncertainty after both '
                  'sources are combined.\n'
                  '        ',
      'terms': (('Parameter',
                 'An unknown but fixed quantity inside the model, such as average demand '
                 'μ.'),
                ('Likelihood',
                 'A score for each candidate parameter value when the observed data are '
                 'held fixed.'),
                ('Prior',
                 'A distribution that represents plausible parameter values before the '
                 'current data are used.'),
                ('Posterior',
                 'The updated parameter distribution after prior information and '
                 'likelihood are combined.')),
      'notation': (('x=(x₁,…,xₙ)', 'the n observed daily sales totals'),
                   ('σ', 'known day-to-day sales noise, not the uncertainty about μ'),
                   ('p(x | μ)',
                    'likelihood: how compatible the fixed records are with candidate μ'),
                   ('p(μ | x)', 'posterior: remaining uncertainty about μ after seeing x')),
      'math_story': '\n'
                    '        Bayes’ rule for this model is\n'
                    '\n'
                    '        $$p(\\mu\\mid x)=\\frac{p(x\\mid\\mu)p(\\mu)}{p(x)}\\propto '
                    'p(x\\mid\\mu)p(\\mu).$$\n'
                    '\n'
                    '        The denominator $p(x)$ is one normalizing number shared by '
                    'every candidate $\\mu$, so it does not change their relative ranking. '
                    'In code, products of many small densities can underflow. We therefore '
                    'use logarithms:\n'
                    '\n'
                    '        $$\\log p(\\mu\\mid x)=\\sum_{i=1}^{n}\\log '
                    'p(x_i\\mid\\mu)+\\log p(\\mu)+C.$$\n'
                    '\n'
                    '        Independent observation scores add. A narrow prior assigns a '
                    'strong penalty far from its center; a wide prior lets the likelihood '
                    'move the posterior more easily.\n'
                    '        ',
      'visual_guide': 'Compare four curves at the same candidate μ: prior height, '
                      'likelihood score, their unnormalized product, and the normalized '
                      'posterior. Watch both the posterior center and width.',
      'connection': 'Negative log likelihood is the data-loss term used in many '
                    'machine-learning models. Adding a log prior creates the Bayesian form '
                    'of regularization.'},
     {'overview': '\n'
                  '        The posterior from the previous lab is a complete landscape, '
                  'but a stocking system may require one number. Optimization searches for '
                  'a point that makes a chosen objective as large or small as possible. '
                  'Maximum likelihood estimation, or MLE, chooses the parameter that best '
                  'explains the data. Maximum a posteriori estimation, or MAP, chooses the '
                  'highest point of the posterior and therefore includes the prior. The '
                  'optimizer and objective are separate ideas: gradient ascent is a search '
                  'method, while likelihood or posterior defines what “best” means.\n'
                  '        ',
      'terms': (('Objective',
                 'The numerical score that an optimizer tries to maximize or minimize.'),
                ('MLE',
                 'The parameter value that maximizes the likelihood and uses no prior '
                 'term.'),
                ('MAP',
                 'The parameter value that maximizes the posterior and includes the '
                 'prior.'),
                ('Gradient',
                 'The local slope; in one dimension it tells which direction increases the '
                 'objective.'),
                ('Learning rate',
                 'The multiplier that converts a gradient into an update step.')),
      'notation': (('θ', 'the parameter being optimized; here θ is the average demand μ'),
                   ('f(θ)', 'the objective, such as log likelihood or log posterior'),
                   ('η', 'the learning rate or step-size multiplier'),
                   ('θₜ', 'the optimizer’s parameter value after t updates')),
      'math_story': '\n'
                    '        Gradient ascent repeats\n'
                    '\n'
                    "        $$\\theta_{t+1}=\\theta_t+\\eta\\,f'(\\theta_t).$$\n"
                    '\n'
                    "        If the slope $f'(\\theta_t)$ is positive, the update moves "
                    'right; if it is negative, the update moves left. A finite difference '
                    'estimates the slope without symbolic calculus:\n'
                    '\n'
                    '        '
                    "$$f'(\\theta)\\approx\\frac{f(\\theta+\\varepsilon)-f(\\theta-\\varepsilon)}{2\\varepsilon}.$$\n"
                    '\n'
                    '        Here $\\varepsilon$ is a tiny probe distance, while $\\eta$ '
                    'is the much larger movement scale. A large $\\eta$ can cross the '
                    'mode, land on the opposite slope, and oscillate or diverge. The final '
                    'point is an estimate, not a distribution.\n'
                    '        ',
      'visual_guide': 'The curve is the fixed objective. The connected markers are '
                      'optimization states. Read each horizontal move together with the '
                      'slope at its starting point, and check whether the steps settle or '
                      'overshoot.',
      'connection': 'Neural-network training uses the same separation: the loss defines '
                    'the goal, while SGD or Adam defines how weights move toward a point '
                    'estimate.'},
     {'overview': '\n'
                  '        A mode answers “where is one highest point?” It does not answer '
                  '“how broadly is probability spread?” Two demand posteriors can peak at '
                  'the same value but imply very different stockout risks. A narrow '
                  'posterior says that nearby values dominate. A broad posterior leaves '
                  'substantial probability far from the mode. This is why MAP is not a '
                  'complete Bayesian answer. A Laplace approximation recovers a local '
                  'uncertainty estimate by fitting a Normal curve near the mode, but it '
                  'can only see the shape close to that one point.\n'
                  '        ',
      'terms': (('Mode', 'A location where a density reaches a local or global maximum.'),
                ('Credible interval',
                 'A posterior interval that contains a stated amount of probability mass.'),
                ('Curvature',
                 'How quickly the slope changes; sharp peaks have large negative '
                 'log-curvature.'),
                ('Laplace approximation',
                 'A Normal approximation centered at a mode and scaled by local '
                 'curvature.')),
      'notation': (('θ_MAP', 'the parameter value at the posterior mode'),
                   ('ℓ(θ)', 'the log posterior evaluated at θ'),
                   ("ℓ''(θ_MAP)",
                    'second derivative, which measures local curvature at the mode')),
      'math_story': '\n'
                    '        Near a smooth mode, a log posterior can be approximated by a '
                    'quadratic curve. This leads to\n'
                    '\n'
                    '        $$p(\\theta\\mid x)\\approx\\mathcal '
                    "N\\!\\left(\\theta_{MAP},\\;-1/\\ell''(\\theta_{MAP})\\right).$$\n"
                    '\n'
                    "        At a maximum, $\\ell''$ is negative. A very negative value "
                    'means a sharp peak and therefore a small approximate variance. A '
                    'value closer to zero means a flatter peak and a larger variance. This '
                    'calculation uses only local information. It misses distant modes, '
                    'skew, hard boundaries, and tails that do not look Normal.\n'
                    '        ',
      'visual_guide': 'Ignore the equal peak locations for a moment and compare horizontal '
                      'spread and shaded credible intervals. Those differences are '
                      'invisible if you retain only the MAP marker.',
      'connection': 'Second-order optimization also uses curvature, but uncertainty '
                    'requires interpreting curvature as local distribution width rather '
                    'than only as a search aid.'},
     {'overview': '\n'
                  '        To preserve the full posterior, we can generate parameter '
                  'values whose long-run occupancy matches the posterior landscape. '
                  'Metropolis–Hastings builds a Markov chain: each new state depends on '
                  'the current state, a random proposal, and an accept-or-reject decision. '
                  'It is not trying to climb permanently toward the mode. It must '
                  'sometimes move to lower-density regions because those regions still '
                  'contain posterior mass. Rejected proposals create repeated states, and '
                  'those repeats are required for the chain to have the correct long-run '
                  'distribution.\n'
                  '        ',
      'terms': (('MCMC',
                 'Markov chain Monte Carlo: methods that estimate a distribution with '
                 'dependent samples.'),
                ('Markov chain',
                 'A sequence whose next state is generated from the current state.'),
                ('Proposal', 'A candidate next state drawn from a chosen movement rule.'),
                ('Acceptance rate',
                 'The fraction of proposals that become the next chain state.'),
                ('Stationary distribution',
                 'A distribution that remains unchanged by one transition of the chain.')),
      'notation': (('θ', 'the current parameter state'),
                   ('θ′', 'the proposed next state'),
                   ('q(θ′ | θ)', 'the proposal distribution used to generate θ′ from θ'),
                   ('α', 'the probability of accepting the proposal')),
      'math_story': '\n'
                    '        For a symmetric random-walk proposal, the acceptance '
                    'probability is\n'
                    '\n'
                    "        $$\\alpha=\\min\\left(1,\\frac{p(\\theta'\\mid "
                    'x)}{p(\\theta\\mid x)}\\right).$$\n'
                    '\n'
                    '        An uphill move has a ratio above one and is always accepted. '
                    'A downhill move is accepted with a smaller probability. In log space, '
                    "compare $\\log u$ with $\\log p(\\theta'\\mid x)-\\log p(\\theta\\mid "
                    'x)$, where $u$ is Uniform between zero and one. Small proposals are '
                    'accepted often but move slowly. Very large proposals often land in '
                    'low-density regions and are rejected. Acceptance alone is therefore '
                    'not a measure of sampling quality.\n'
                    '        ',
      'visual_guide': 'Read the contour or density together with the trace. Green accepted '
                      'moves show travel; rejected moves appear as repeated trace values. '
                      'Compare distance traveled, not only acceptance percentage.',
      'connection': 'Unlike noisy gradient training, Metropolis uses random proposals and '
                    'a correction rule to target a distribution rather than one optimum.'},
     {'overview': '\n'
                  '        Saving 5,000 chain states does not guarantee 5,000 independent '
                  'pieces of information. A random-walk chain often remains near its '
                  'previous value, so adjacent draws resemble one another. Autocorrelation '
                  'measures this serial dependence. Effective sample size, or ESS, '
                  'translates the dependent chain into an approximate number of '
                  'independent draws with similar estimator precision. A chain with fewer '
                  'but less correlated samples can be more informative than a much longer '
                  'sticky chain.\n'
                  '        ',
      'terms': (('Lag',
                 'The number of time steps between two chain values being compared.'),
                ('Autocorrelation',
                 'Correlation between a sequence and a lagged copy of itself.'),
                ('ESS',
                 'An estimate of how many independent draws would carry similar '
                 'information.'),
                ('Mixing',
                 'How effectively a chain moves among all relevant regions of its '
                 'target.')),
      'notation': (('N', 'the raw number of saved MCMC draws'),
                   ('ρₖ', 'autocorrelation at lag k'),
                   ('τ',
                    'integrated autocorrelation time, an approximate dependence penalty')),
      'math_story': '\n'
                    '        For centered draws $y_t=x_t-\\bar x$, a simple lag-$k$ '
                    'estimate compares $y_t$ with $y_{t+k}$. Positive values mean that '
                    'knowing the present helps predict the future. A common intuition is\n'
                    '\n'
                    '        $$ESS\\approx\\frac{N}{1+2\\sum_{k=1}^{K}\\rho_k}.$$\n'
                    '\n'
                    '        The denominator grows when positive autocorrelations persist, '
                    'so ESS falls. Practical estimators use careful truncation and chain '
                    'splitting; this lab uses a simpler form to expose the mechanism. ESS '
                    'concerns Monte Carlo information, not the number of real observations '
                    'used in the statistical model.\n'
                    '        ',
      'visual_guide': 'Use three views together: the trace shows movement over time, lag '
                      'scatter shows pairwise dependence, and the autocorrelation curve '
                      'shows how long memory persists.',
      'connection': 'Correlated mini-batches or optimizer states can also reduce new '
                    'information per step, although MCMC ESS has a specific '
                    'stationary-sampling meaning.'},
     {'overview': '\n'
                  '        One-dimensional plots hide geometry. Suppose baseline demand '
                  'and price sensitivity can compensate for each other: a higher baseline '
                  'with a stronger price penalty can predict similar sales. The posterior '
                  'then forms a narrow diagonal valley in two dimensions. An isotropic '
                  'random walk proposes equally in every direction. A step small enough to '
                  'stay inside the narrow direction moves very slowly along the long '
                  'direction; a larger step frequently crosses the valley walls and is '
                  'rejected. The target is simple, but its geometry makes the sampler '
                  'inefficient.\n'
                  '        ',
      'terms': (('Covariance',
                 'A measure of how two variables change together in their original units.'),
                ('Correlation', 'A unit-free covariance scaled to lie between -1 and 1.'),
                ('Contour',
                 'A line joining points with equal density, like equal-height lines on a '
                 'map.'),
                ('Isotropic', 'Having the same scale in every direction.')),
      'notation': (('θ=(θ₁,θ₂)', 'the two-dimensional parameter vector'),
                   ('Σ', 'the covariance matrix that controls scale and orientation'),
                   ('ρ', 'the correlation; values near ±1 create a thin diagonal shape')),
      'math_story': '\n'
                    '        A standardized correlated Gaussian uses\n'
                    '\n'
                    '        '
                    '$$\\Sigma=\\begin{bmatrix}1&\\rho\\\\\\rho&1\\end{bmatrix}.$$\n'
                    '\n'
                    '        The diagonal entries are variances. The off-diagonal entries '
                    'are covariance, which equals correlation here because both standard '
                    'deviations are one. As $|\\rho|$ approaches one, one eigen-direction '
                    'becomes narrow and the other remains long. You do not need an '
                    'eigendecomposition to see this: the contours visibly flatten into a '
                    'tilted ellipse. Posterior algorithms must move efficiently in the '
                    'target’s useful directions, not only use a globally reasonable scalar '
                    'step size.\n'
                    '        ',
      'visual_guide': 'Each contour is an equal-density line. Compare proposals across the '
                      'short axis with movement along the long axis, and watch how the '
                      'trajectory changes as correlation approaches one.',
      'connection': 'Neural-network losses also have flat and steep directions. Poor '
                    'conditioning slows both optimization and random-walk sampling.'},
     {'overview': '\n'
                  'A gradient generalizes one-dimensional slope to many dimensions. At '
                  'each point, it provides one partial derivative per coordinate. The '
                  'gradient of the log posterior points in the direction of fastest local '
                  'increase. Drawing these vectors over posterior contours turns an '
                  'abstract derivative into a vector field. An optimizer can change '
                  'position directly in the gradient direction and move toward a mode. A '
                  'posterior sampler has a different goal: it must represent the full '
                  'distribution, not only its highest point. It can use gradient '
                  'information as one input without using “always move uphill” as its '
                  'complete rule. The next lab introduces the extra HMC state and movement '
                  'rule that make this possible.\n',
      'terms': (('Partial derivative',
                 'The slope obtained when one coordinate changes and the others stay '
                 'fixed.'),
                ('Gradient',
                 'The vector containing all partial derivatives of a scalar function.'),
                ('Vector field', 'A vector attached to every location in a space.'),
                ('Autodiff',
                 'Software that applies the chain rule to compute exact program '
                 'derivatives.')),
      'notation': (('∇', 'the gradient operator'),
                   ('∇ log p(θ)',
                    'the vector of log-density slopes at parameter position θ'),
                   ('eᵢ',
                    'a unit vector that changes only coordinate i in a finite difference')),
      'math_story': '\n'
                    '        For $d$ parameters, the gradient is\n'
                    '\n'
                    '        $$\\nabla f(\\theta)=\\left(\\frac{\\partial '
                    'f}{\\partial\\theta_1},\\ldots,\\frac{\\partial '
                    'f}{\\partial\\theta_d}\\right).$$\n'
                    '\n'
                    '        A centered numerical estimate for coordinate $i$ is\n'
                    '\n'
                    '        $$\\frac{\\partial '
                    'f}{\\partial\\theta_i}\\approx\\frac{f(\\theta+\\varepsilon '
                    'e_i)-f(\\theta-\\varepsilon e_i)}{2\\varepsilon}.$$\n'
                    '\n'
                    '        This needs two function evaluations per coordinate, so it '
                    'becomes costly in large models. Torch and other autodiff systems '
                    'compute the same mathematical object more efficiently. Near a smooth '
                    'mode, arrows become shorter because the local slope approaches zero.\n'
                    '        ',
      'visual_guide': 'Arrow direction shows local uphill direction; arrow length shows '
                      'slope magnitude. Compare arrows with contour spacing: tightly '
                      'spaced contours usually indicate a steeper direction.',
      'connection': 'Backpropagation computes gradients of a neural-network loss. The '
                    'later HMC labs reuse gradient information but do not train toward one '
                    'final weight vector.'},
     {'overview': 'Lab 8 showed a gradient arrow at each point, but HMC needs a rule for '
                  'how a state changes through time. Its state has two vectors. Position θ '
                  'contains the two model parameters. Momentum r is a temporary movement '
                  'vector that HMC creates for one proposed trajectory. The posterior '
                  'gradient does not directly replace θ. It first changes r; then r '
                  'changes θ. This distinction resolves an important apparent '
                  'contradiction: zero initial momentum does not imply no later motion. If '
                  'the particle starts on a slope, the gradient accelerates it from rest. '
                  'It can then cross the mode because the momentum it gained does not '
                  'disappear when the local slope becomes zero.',
      'terms': (('Position', 'The current model-parameter vector θ on the posterior plot.'),
                ('Momentum',
                 'One temporary vector r that stores the current direction and amount of '
                 'movement.'),
                ('Component',
                 'One signed coordinate of a vector; r₁ is horizontal and r₂ is vertical '
                 'in this plot.'),
                ('Posterior force',
                 'The gradient of log posterior density, which changes momentum at the '
                 'current position.'),
                ('Potential energy',
                 'Negative log target density; high posterior density has low potential '
                 'energy.'),
                ('Kinetic energy',
                 'One half of the squared momentum length when the mass matrix is the '
                 'identity.'),
                ('Hamiltonian',
                 'The sum of potential and kinetic energy that ideal HMC motion '
                 'preserves.')),
      'notation': (('θ = (θ₁, θ₂)',
                    'the particle position: baseline-demand and price-response parameters'),
                   ('r = (r₁, r₂)',
                    'one momentum vector with horizontal and vertical components'),
                   ('∇ log p(θ)', 'the posterior force evaluated at the current position'),
                   ('U(θ), K(r), H', 'potential, kinetic, and total Hamiltonian energy')),
      'math_story': 'For this first HMC model, the mass matrix is the identity, so the '
                    'update rules have a simple form: $$dr/dt = \\nabla \\log p(\\theta)$$ '
                    'and $$d\\theta/dt = r.$$ The first equation takes the local posterior '
                    'slope and changes momentum. The second takes the new momentum and '
                    'changes position. If r starts at zero but the gradient does not, the '
                    'first equation immediately creates motion. Energy gives the same '
                    'story: $$U(\\theta)=-\\log p(\\theta),\\quad K(r)=\\tfrac12 r^T '
                    'r,\\quad H=U+K.$$ Moving toward high density lowers U and raises K. '
                    'At the mode, the slope can be zero while K is large, so position '
                    'continues to change.',
      'visual_guide': 'In the left panel, contours show equal posterior density, the blue '
                      'point is the fixed start, and the star is the mode. The teal curve '
                      'uses posterior force; the dashed gray curve removes that force but '
                      'keeps the same initial momentum. The middle panel shows both signed '
                      'momentum components and their length. The right panel shows '
                      'potential, kinetic, and total energy. Use all three panels to '
                      'identify the cause of movement.',
      'connection': 'A neural-network optimizer can also maintain a velocity-like state, '
                    'but it uses loss reduction and damping to settle near one point. HMC '
                    'uses reversible energy exchange to travel across many plausible '
                    'parameter values instead of converging to the mode.'},
     {'overview': 'Ideal Hamiltonian motion changes position and momentum continuously. '
                  'Computer code sees only a sequence of finite states, so it must decide '
                  'where in each interval to evaluate the posterior force. A one-sided '
                  'update uses only the old force, or only the new force, for the complete '
                  'momentum change. Leapfrog instead places equal half momentum updates '
                  'around the position update. This time-centered structure makes the '
                  'discrete map reversible: a negative step can undo a positive step. '
                  'Reversibility and volume preservation let HMC compare the old and '
                  'proposed joint densities with only their Hamiltonian values. Leapfrog '
                  'still has numerical error, so HMC uses a Metropolis correction. This is '
                  'a precise random accept-or-reject decision, not a second optimization '
                  'step and not a repair of the simulated path.',
      'terms': (('Time-centered update',
                 'An update that uses matching information before and after the central '
                 'position move.'),
                ('Reversible map',
                 'A numerical map that returns to its start when the same operations run '
                 'backward.'),
                ('Volume preservation',
                 'The map does not compress or expand small regions of position-momentum '
                 'space.'),
                ('Proposal',
                 'The endpoint that HMC asks the Markov chain to use as its next state.'),
                ('Hamiltonian error',
                 'The signed change ΔH caused by approximate numerical motion.'),
                ('Metropolis correction',
                 'A random decision that accepts the proposal or repeats the old state '
                 'using ΔH.'),
                ('Acceptance probability',
                 'The number α between zero and one used in the final random decision.')),
      'notation': (('q and p', 'code names for position θ and momentum r'),
                   ('ε', 'the signed leapfrog step size'),
                   ('ΔH = H_new - H_old',
                    'the signed energy error of the complete proposed path'),
                   ('α', 'the probability of accepting the proposed endpoint'),
                   ('u', 'one Uniform(0,1) draw used to make the decision')),
      'math_story': 'Write a momentum update as K and a position update as D. Leapfrog '
                    'composes them as $K_{\\epsilon/2}D_{\\epsilon}K_{\\epsilon/2}$. Its '
                    'inverse is $K_{-\\epsilon/2}D_{-\\epsilon}K_{-\\epsilon/2}$, which is '
                    'the same pattern with a negative step. This is why the two halves '
                    'help: they put the operation in a symmetric order that can retrace '
                    'itself. Each substep is also a shear with unit volume. Exact '
                    'Hamiltonian motion would give $\\Delta H=0$. Leapfrog gives a small '
                    'signed error, so the joint density ratio is '
                    '$\\exp(-H_{new})/\\exp(-H_{old})=\\exp(-\\Delta H)$. HMC therefore '
                    'uses $\\alpha=\\min(1,\\exp(-\\Delta H))$. If $\\Delta H\\leq0$, '
                    'accept. If it is positive, accept only when a uniform draw u is '
                    'smaller than α. A rejection stores the old position again in the '
                    'chain.',
      'visual_guide': 'In the path panel, teal is symmetric leapfrog and orange is the '
                      'one-sided comparison. In the energy panel, zero means exact '
                      'conservation; bounded oscillation is better than systematic drift. '
                      'In the acceptance panel, the curve averages many momentum draws '
                      'because one path can have negative ΔH and acceptance 1. In the '
                      'table, return error measures how closely each method retraces its '
                      'complete position-momentum state.',
      'connection': 'An optimizer can use momentum and a learning rate without needing a '
                    'reversible update, because its goal is to settle at one point. HMC '
                    'needs reversibility and an accept-or-reject correction because its '
                    'goal is to preserve a complete target distribution.'},
     {'overview': 'A leapfrog trajectory is the deterministic middle of HMC. Once its '
                  'initial position, momentum, step size, and step count are fixed, every '
                  'later point is fixed. A position-only view can still trace a wide or '
                  'complicated shape because it hides momentum, and finite leapfrog steps '
                  'can add phase and energy error. The central question of this chapter is '
                  'whether such visible movement is enough to represent a posterior. You '
                  'will compare it with complete HMC transitions. Each complete transition '
                  'draws momentum, runs a trajectory, makes an accept-or-repeat decision, '
                  'and stores one position for the next transition.',
      'terms': (('Trajectory',
                 'The deterministic sequence of position-momentum states produced after '
                 'one initial state is fixed.'),
                ('Hamiltonian at a state',
                 "One scalar H(q,p) that adds the position's potential energy and the "
                 "momentum's kinetic energy."),
                ('Energy shell',
                 'The set of joint states (q,p) with the same total Hamiltonian H. This is '
                 'not a posterior mode.'),
                ('Position projection',
                 'A view that shows q but hides p. Several different joint '
                 'position-momentum states can appear in the same position-only view.'),
                ('Phase error',
                 'A timing error from discrete integration that slowly shifts where an '
                 'approximate orbit is along its cycle.'),
                ('Shadow energy',
                 'A nearby modified energy that a stable symplectic leapfrog path follows '
                 'closely, even when the exact Hamiltonian is not perfectly constant.'),
                ('Transition',
                 'One complete proposal-and-decision operation that returns exactly one '
                 'next chain position.'),
                ('Momentum refresh',
                 'A new auxiliary Normal draw before each HMC trajectory.'),
                ('Markov chain',
                 'The ordered sequence of stored positions, where the next transition '
                 'starts from the current stored position.'),
                ('Repeated state',
                 'The old position stored again after rejection; it is a required sample, '
                 'not missing output.'),
                ('Stationary target',
                 'The distribution that a correct transition preserves and that a '
                 'well-mixed chain represents over time.')),
      'notation': (('q_t', 'the position stored at chain transition t'),
                   ('p₀', 'fresh momentum drawn for the current transition'),
                   ('(q*, p*)',
                    'the deterministic endpoint proposed by the leapfrog trajectory'),
                   ('L', 'the fixed number of leapfrog steps in each trajectory'),
                   ('α and u',
                    'the acceptance probability and the Uniform random draw used for the '
                    'decision'),
                   ('log_target(q)',
                    'the target log density at position q; its negative is potential '
                    'energy'),
                   ('pᵀp or np.dot(p, p)',
                    'the sum of squared momentum components; one half of it is kinetic '
                    'energy for unit mass'),
                   ('old_h and new_h',
                    'the scalar Hamiltonians before and after the proposed trajectory')),
      'math_story': 'Lab 9 wrote total Hamiltonian energy as $H=U+K$. In this notebook, '
                    '`log_target(q)` returns $\\log p(q)$, so potential energy is '
                    '$U(q)=-\\log p(q)$. The negative sign means that a more probable '
                    'position has lower potential energy. We use the identity mass matrix, '
                    'so kinetic energy is $K(p)=p^Tp/2$. In NumPy, `np.dot(p, p)` computes '
                    '$p^Tp$, the sum of squared momentum components. Thus '
                    '`-float(log_target(q)) + 0.5 * float(np.dot(p, p))` is exactly '
                    '$H(q,p)$. The `float` calls turn NumPy scalar results into ordinary '
                    'Python scalars; they do not change the mathematics. Compute this once '
                    'for the current state and once for the proposal. Their difference '
                    'controls the Metropolis correction. The momentum sign flip leaves '
                    'kinetic energy unchanged because squaring removes the sign.',
      'visual_guide': 'The result is hidden until you lock a prediction. After you run it, '
                      'the first panel will show one deterministic proposal path. The '
                      'second will show one stored position from each complete HMC '
                      'transition. The third will color one no-refresh path by time and '
                      'will include an inset for $\\Delta H$ relative to its initial joint '
                      'state. Compare the processes only after all three panels are '
                      'visible.',
      'connection': 'A neural-network optimizer also repeats state updates, but it '
                    'normally keeps one current parameter vector and seeks a low-loss '
                    'point. HMC deliberately adds auxiliary randomness and an invariant '
                    'transition so stored positions represent a distribution instead.'},
     {'overview': '\n'
                  '        Basic HMC requires a trajectory length, usually expressed as a '
                  'step size times a fixed number of leapfrog steps. A short trajectory '
                  'wastes the ability to make distant proposals. A long trajectory can '
                  'turn back toward its starting point and spend computation retracing '
                  'explored space. The No-U-Turn Sampler, or NUTS, grows a trajectory '
                  'adaptively and stops when its geometry indicates that continued '
                  'movement is doubling back. Production NUTS also uses a balanced tree '
                  'and careful sampling rules; this lab focuses on the central geometric '
                  'signal.\n'
                  '        ',
      'terms': (('NUTS', 'No-U-Turn Sampler, an HMC method that adapts trajectory length.'),
                ('Trajectory',
                 'The ordered positions produced by numerical Hamiltonian motion.'),
                ('U-turn',
                 'A state where momentum points back toward an earlier part of the '
                 'trajectory.'),
                ('Warmup',
                 'Early adaptation used to learn step size and often a mass matrix.'),
                ('Tree depth', 'The number of trajectory-doubling levels used by NUTS.')),
      'notation': (('θ₀', 'the trajectory’s starting position'),
                   ('θ', 'a current or endpoint position'),
                   ('r', 'current momentum'),
                   ('(θ-θ₀)·r',
                    'alignment between displacement from the start and momentum')),
      'math_story': '\n'
                    '        A simplified U-turn detector checks\n'
                    '\n'
                    '        $$(\\theta-\\theta_0)^Tr<0.$$\n'
                    '\n'
                    '        The displacement $\\theta-\\theta_0$ points from the start to '
                    'the current location. If its dot product with momentum is negative, '
                    'the angle between them is greater than 90 degrees, so motion has a '
                    'component back toward the start. Production NUTS checks both ends of '
                    'a recursively grown trajectory and must preserve detailed balance. It '
                    'also adapts $\\varepsilon$ during warmup. The one-dot-product '
                    'exercise is a geometric intuition, not a production implementation.\n'
                    '        ',
      'visual_guide': 'Draw the displacement and momentum vectors from the same point. '
                      'Their dot product becomes negative when the arrows point more '
                      'against than with each other.',
      'connection': 'Adaptive computation also appears in neural networks, but NUTS adapts '
                    'path length to posterior geometry while preserving a sampling '
                    'target.'},
     {'overview': '\n'
                  '        MCMC does not converge to one parameter value. It aims to enter '
                  'a stationary sampling regime in which all chains explore the same '
                  'posterior distribution. Multiple chains started in different places '
                  'give evidence about whether this happened. Lab 13 makes that comparison '
                  'concrete. First measure how much draws vary inside each chain. Then '
                  'measure how far the chain means are from one another. A basic R-hat '
                  'estimate compares these two scales. Similar-looking marginal histograms '
                  'are not enough: chains can remain stuck in separate modes, have strong '
                  'autocorrelation, or suffer numerical divergences. Read R-hat with '
                  'traces, effective sample size, cumulative means, and knowledge of the '
                  'model geometry.\n'
                  '        ',
      'terms': (('Chain',
                 'One dependent sequence of MCMC states from one initial position.'),
                ('R-hat',
                 'A diagnostic that compares between-chain and within-chain variation.'),
                ('MCSE',
                 'Monte Carlo standard error, uncertainty caused by using a finite '
                 'sample.'),
                ('Divergence',
                 'A warning that numerical HMC motion could not resolve part of the target '
                 'geometry.'),
                ('Convergence',
                 'Entry into a common stationary sampling regime, not arrival at one '
                 'point.')),
      'notation': (('m', 'number of chains; this is axis 0 of the input array'),
                   ('n', 'draws retained in each chain; this is axis 1'),
                   ('xᵢⱼ', 'draw j from chain i'),
                   ('x̄ᵢ', 'mean of all n draws in chain i'),
                   ('x̄··', 'grand mean of the m chain means'),
                   ('W', 'average sample variance inside the chains'),
                   ('B', 'sample variance of the chain means, multiplied by n')),
      'math_story': '\n'
                    '        Suppose `chains` is a rectangular array with `m` rows and `n` '
                    'columns. Row `i` is one chain, and entry `xᵢⱼ` is draw `j` from that '
                    'chain. R-hat starts with two different questions.\n'
                    '\n'
                    '        First ask: how much does a typical chain move around its own '
                    'center? Compute the mean and sample variance of each row.\n'
                    '\n'
                    '        $$\n'
                    '        \\bar{x}_i=\\frac{1}{n}\\sum_{j=1}^{n}x_{ij}\n'
                    '        $$\n'
                    '\n'
                    '        $$\n'
                    '        s_i^2=\\frac{1}{n-1}\\sum_{j=1}^{n}(x_{ij}-\\bar{x}_i)^2\n'
                    '        $$\n'
                    '\n'
                    '        Average those `m` row variances to get the within-chain '
                    'variance.\n'
                    '\n'
                    '        $$\n'
                    '        W=\\frac{1}{m}\\sum_{i=1}^{m}s_i^2\n'
                    '        $$\n'
                    '\n'
                    '        In NumPy, `chains.var(axis=1, ddof=1)` gives one sample '
                    'variance per row. Taking the mean of that vector gives `W`. The '
                    'argument `ddof=1` makes NumPy divide by one less than the number of '
                    'values, as required by the sample-variance formulas here.\n'
                    '\n'
                    '        Second ask: how far apart are the chain centers? Let the '
                    'grand mean be the average of the chain means.\n'
                    '\n'
                    '        $$\n'
                    '        '
                    '\\bar{x}_{\\cdot\\cdot}=\\frac{1}{m}\\sum_{i=1}^{m}\\bar{x}_i\n'
                    '        $$\n'
                    '\n'
                    '        The between-chain quantity is the sample variance of the '
                    'chain means, multiplied by the number of draws in each chain.\n'
                    '\n'
                    '        $$\n'
                    '        '
                    'B=\\frac{n}{m-1}\\sum_{i=1}^{m}(\\bar{x}_i-\\bar{x}_{\\cdot\\cdot})^2\n'
                    '        $$\n'
                    '\n'
                    '        In NumPy, compute the chain means across `axis=1`, take their '
                    'sample variance with `ddof=1`, and multiply by `n`. The factor `n` '
                    'puts `B` on the same variance scale as `W`. If chains occupy '
                    'different regions, their means disagree and `B` becomes large.\n'
                    '\n'
                    '        A basic pooled variance estimate combines the two '
                    'quantities.\n'
                    '\n'
                    '        $$\n'
                    '        \\widehat{V}=\\frac{n-1}{n}W+\\frac{1}{n}B\n'
                    '        $$\n'
                    '\n'
                    '        Then R-hat compares the pooled estimate with the variation '
                    'observed inside chains.\n'
                    '\n'
                    '        $$\n'
                    '        \\widehat{R}=\\sqrt{\\frac{\\widehat{V}}{W}}\n'
                    '        $$\n'
                    '\n'
                    '        For a small hand check, use two chains: `[0, 2]` and `[2, '
                    '4]`. Their means are `1` and `3`. Each row has sample variance `2`, '
                    'so `W` is `2`. The sample variance of the two means is also `2`; '
                    'because `n` is `2`, `B` is `4`. The pooled estimate is `3`, and basic '
                    'R-hat is approximately `1.225`. The chains are short and their '
                    'centers disagree, so a value above one is expected.\n'
                    '        ',
      'visual_guide': 'Compare traces, chain histograms, and cumulative means. A stable '
                      'line within each chain is not enough if different chains stabilize '
                      'in different regions.',
      'connection': 'Training curves can also look stable while a model is wrong. '
                    'Diagnostics test algorithm behavior; they do not validate the '
                    'statistical assumptions by themselves.'},
     {'overview': '\n'
                  '        MCMC represents a posterior with samples, but it can be costly '
                  'when the parameter space is large. Variational inference, or VI, '
                  'chooses a simpler distribution $q_\\phi(\\theta)$ and adjusts its '
                  'parameters $\\phi$ to make it close to the target. This turns inference '
                  'into optimization. The output is still a distribution, unlike MAP, but '
                  'it is restricted to a chosen approximation family. The practical gain '
                  'is speed and compatibility with stochastic gradients; the cost is '
                  'approximation bias that standard optimization diagnostics may not '
                  'reveal.\n'
                  '        ',
      'terms': (('Variational inference',
                 'Optimization of a tractable distribution that approximates a posterior.'),
                ('Variational family',
                 'The set of distributions the approximation is allowed to use.'),
                ('ELBO', 'Evidence lower bound, an objective that VI maximizes.'),
                ('Entropy', 'A measure of how spread out a distribution is.'),
                ('Reparameterization',
                 'Writing a random sample as a differentiable transform of fixed noise.')),
      'notation': (('qφ(θ)', 'the approximation to the posterior'),
                   ('φ', 'variational parameters, such as mean and log standard deviation'),
                   ('p(D,θ)',
                    'the model’s joint density for observed data and latent parameters'),
                   ('ε', 'fixed standard-Normal noise used for reparameterized samples')),
      'math_story': '\n'
                    '        The evidence lower bound is\n'
                    '\n'
                    '        $$\\mathcal L(\\phi)=\\mathbb E_{q_\\phi}\\left[\\log '
                    'p(D,\\theta)-\\log q_\\phi(\\theta)\\right].$$\n'
                    '\n'
                    '        The first term rewards samples that the model considers '
                    'plausible. The subtraction of $\\log q$ rewards entropy and prevents '
                    'the approximation from collapsing without cost. For a Gaussian, write '
                    '$\\theta=\\mu+\\exp(\\lambda)\\varepsilon$, where $\\lambda$ is log '
                    'standard deviation and $\\varepsilon\\sim\\mathcal N(0,1)$. Holding '
                    'sampled $\\varepsilon$ fixed makes the Monte Carlo objective '
                    'differentiable with respect to $\\mu$ and $\\lambda$.\n'
                    '        ',
      'visual_guide': 'First move the Gaussian approximation by hand. Compare both center '
                      'and tails with the target. Then watch optimization change mean and '
                      'spread, not a model parameter point.',
      'connection': 'Modern deep-learning VI uses autodiff and stochastic optimizers. It '
                    'trains distribution parameters instead of one deterministic weight '
                    'vector.'},
     {'overview': '\n'
                  '        VI can optimize its objective successfully and still give a '
                  'poor scientific answer. The reason can be structural: a diagonal '
                  'Gaussian cannot become skewed, split into two modes, or bend around a '
                  'banana-shaped ridge. Optimization finds the best member of the selected '
                  'family, not the best distribution imaginable. This lab separates '
                  'optimization failure from approximation-family failure. A good ELBO '
                  'trajectory says that the chosen family was optimized; it does not prove '
                  'that the family can represent the posterior features needed by the '
                  'decision.\n'
                  '        ',
      'terms': (('Mean-field',
                 'An approximation that treats coordinates as independent factors.'),
                ('Multimodal',
                 'Having more than one separated region of high probability.'),
                ('Skew', 'Asymmetry in which one tail extends farther than the other.'),
                ('Mode collapse',
                 'An approximation that covers one mode and ignores another.'),
                ('Full covariance',
                 'A Gaussian approximation that can represent linear correlation.')),
      'notation': (('q(θ)=∏ᵢqᵢ(θᵢ)',
                    'the mean-field factorization that removes dependence'),
                   ('Σ',
                    'covariance matrix; diagonal Σ cannot rotate or correlate coordinates'),
                   ('KL',
                    'Kullback–Leibler divergence, a discrepancy used to compare '
                    'distributions')),
      'math_story': '\n'
                    '        Mean-field VI assumes\n'
                    '\n'
                    '        $$q(\\theta)=\\prod_i q_i(\\theta_i).$$\n'
                    '\n'
                    '        This is a modeling restriction on the approximation, not on '
                    'the true posterior. A full-covariance Gaussian replaces independent '
                    'scales with a matrix $\\Sigma$ and can represent tilted ellipses, but '
                    'it still cannot represent two separated peaks or a curved banana. The '
                    'common VI objective corresponds to minimizing $KL(q\\|p)$. Placing '
                    '$q$ in low-density gaps is expensive, so a unimodal $q$ can prefer '
                    'one target mode and underestimate global uncertainty.\n'
                    '        ',
      'visual_guide': 'Ask which target feature the approximation family cannot draw. Then '
                      'distinguish a bad parameter setting from a shape that no setting '
                      'can express.',
      'connection': 'A neural network architecture also limits the functions training can '
                    'find. Better optimization cannot add capacity that the chosen family '
                    'does not contain.'},
     {'overview': '\n'
                  '        Some models contain variables that are conceptually important '
                  'but not directly observed. Customer arrival times might come from a '
                  'morning group and an afternoon group, but the group label for each '
                  'customer is missing. If the labels were known, estimating each group '
                  'mean would be easy. If the group parameters were known, estimating '
                  'label probabilities would be easy. Expectation Maximization, or EM, '
                  'alternates these two conditional tasks. It produces a point estimate of '
                  'model parameters and soft assignments, not a full posterior over all '
                  'uncertainty.\n'
                  '        ',
      'terms': (('Latent variable',
                 'A model variable that is not directly observed, such as a hidden group '
                 'label.'),
                ('Mixture model',
                 'A distribution formed by combining several component distributions.'),
                ('Responsibility',
                 'The current probability that one component generated one observation.'),
                ('E-step', 'The EM step that computes expected hidden assignments.'),
                ('M-step',
                 'The EM step that updates parameters using those expected assignments.')),
      'notation': (('xᵢ', 'observed arrival time for customer i'),
                   ('zᵢ', 'unobserved component label for customer i'),
                   ('rᵢk', 'responsibility of component k for observation i'),
                   ('πₖ, μₖ', 'mixture weight and mean of component k')),
      'math_story': '\n'
                    '        The E-step computes\n'
                    '\n'
                    '        $$r_{ik}=P(z_i=k\\mid x_i,\\pi,\\mu).$$\n'
                    '\n'
                    '        The M-step treats these probabilities as fractional counts:\n'
                    '\n'
                    '        $$N_k=\\sum_i r_{ik},\\qquad \\pi_k=\\frac{N_k}{n},\\qquad\n'
                    '        \\mu_k=\\frac{\\sum_i r_{ik}x_i}{N_k}.$$\n'
                    '\n'
                    '        Each complete E/M cycle does not decrease the observed-data '
                    'likelihood. EM is a form of coordinate ascent on a lower bound, which '
                    'links it to VI. It can still reach a local optimum or preserve a '
                    'symmetric bad start if both components begin identically.\n'
                    '        ',
      'visual_guide': 'Color mixing shows responsibilities, not hard truth. Watch '
                      'ambiguous points contribute fractionally to both component means as '
                      'E and M steps alternate.',
      'connection': 'Soft clustering is useful, but a responsibility is conditional on one '
                    'fitted parameter point. Bayesian mixture inference would also '
                    'preserve parameter uncertainty.'},
     {'overview': '\n'
                  '        MAP, EM, VI, MCMC, and NUTS are often listed as if they were '
                  'interchangeable solvers. They are not. They target different objects '
                  'and preserve different forms of uncertainty. MAP returns one posterior '
                  'mode. EM returns a point fit plus conditional latent responsibilities. '
                  'VI returns a member of an approximate distribution family. MCMC returns '
                  'dependent samples from a target distribution, while NUTS is one '
                  'gradient-based MCMC transition method. Method choice should start with '
                  'the object the decision needs, then consider geometry, cost, and '
                  'diagnostics.\n'
                  '        ',
      'terms': (('Point estimate',
                 'One selected parameter vector, with uncertainty omitted from the output '
                 'object.'),
                ('Approximation',
                 'A simpler object used in place of a target that is harder to compute.'),
                ('Monte Carlo',
                 'Use of random samples to estimate expectations or probabilities.'),
                ('Inference target',
                 'The mathematical object an algorithm is designed to recover or '
                 'optimize.')),
      'notation': (('θ*', 'an optimized point estimate'),
                   ('qφ(θ)', 'a variational approximation with optimized parameters φ'),
                   ('{θ⁽ˢ⁾}', 'a collection of sampled parameter states'),
                   ('p(θ | D)', 'the posterior target that MCMC aims to represent')),
      'math_story': '\n'
                    '        The methods can be organized by their output:\n'
                    '\n'
                    '        $$\\text{MAP: }\\theta^*=\\arg\\max_\\theta p(\\theta\\mid '
                    'D),$$\n'
                    '        $$\\text{VI: }\\phi^*=\\arg\\max_\\phi \\mathcal L(\\phi),$$\n'
                    '        $$\\text{MCMC: }\\theta^{(1)},\\ldots,\\theta^{(S)}\\sim '
                    'p(\\theta\\mid D)\\ \\text{in the stationary sense}.$$\n'
                    '\n'
                    '        EM also optimizes a point objective while alternating '
                    'expected latent assignments and parameter updates. NUTS is not '
                    'another target; it is an adaptive HMC algorithm used to generate MCMC '
                    'transitions. These distinctions determine which summaries and '
                    'diagnostics are meaningful.\n'
                    '        ',
      'visual_guide': 'Read the comparison by columns: target, returned object, preserved '
                      'uncertainty, main diagnostic, and common failure. Do not compare '
                      'speed before checking that outputs answer the same question.',
      'connection': 'Ordinary neural-network training usually belongs in the '
                    'point-estimate column. Bayesian neural methods move toward VI or '
                    'posterior sampling.'},
     {'overview': '\n'
                  '        The bakery now operates several branches. Each branch has its '
                  'own demand or success rate, but the branches are related because they '
                  'share suppliers, recipes, and customer behavior. Estimating every '
                  'branch separately wastes this shared information. Forcing all branches '
                  'to one value ignores real variation. A hierarchical model learns a '
                  'population distribution and branch-specific parameters together. The '
                  'result is partial pooling: uncertain branches move toward the '
                  'population mean more than branches with abundant precise data.\n'
                  '        ',
      'terms': (('Hierarchical model',
                 'A model in which group parameters are drawn from a shared population '
                 'distribution.'),
                ('Partial pooling',
                 'Sharing information across groups without forcing their estimates to be '
                 'equal.'),
                ('Shrinkage',
                 'Movement of a noisy group estimate toward a shared population estimate.'),
                ('Hyperparameter',
                 'A parameter that controls the distribution of lower-level parameters.'),
                ('Exchangeability',
                 'A modeling judgment that group labels do not change prior '
                 'plausibility.')),
      'notation': (('θⱼ', 'the unknown parameter for branch j'),
                   ('μ', 'population mean shared by branches'),
                   ('τ', 'between-branch standard deviation'),
                   ('SEⱼ', 'standard error of branch j’s raw estimate')),
      'math_story': '\n'
                    '        A simple hierarchy is\n'
                    '\n'
                    '        $$\\theta_j\\sim\\mathcal N(\\mu,\\tau),\\qquad '
                    'y_j\\sim\\mathcal N(\\theta_j,SE_j).$$\n'
                    '\n'
                    '        With fixed $\\mu$ and $\\tau$, the posterior mean has the '
                    'weighted form\n'
                    '\n'
                    '        $$E[\\theta_j\\mid y_j]=w_jy_j+(1-w_j)\\mu,\\qquad\n'
                    '        w_j=\\frac{\\tau^2}{\\tau^2+SE_j^2}.$$\n'
                    '\n'
                    '        Large measurement uncertainty $SE_j$ makes $w_j$ small, so '
                    'the population receives more weight. As $\\tau$ approaches zero, the '
                    'model says branches are nearly identical and all estimates strongly '
                    'pool toward $\\mu$. Shrinkage is learned uncertainty weighting, not '
                    'an arbitrary penalty.\n'
                    '        ',
      'visual_guide': 'Read each arrow from a raw branch estimate to its posterior '
                      'estimate. Compare arrow length with data amount and then reduce '
                      'population variation to strengthen pooling.',
      'connection': 'Hierarchical effects resemble regularized embeddings: low-data '
                    'entities borrow strength, while high-data entities can retain '
                    'distinct estimates.'},
     {'overview': '\n'
                  '        We now apply the hierarchy to three coupon rules. For each '
                  'rule, reviewed buyers tell us how often the rule fires among buyers, '
                  'while reviewed nonbuyers tell us how often it fires without a purchase. '
                  'The first rate measures buyer coverage; the second measures nonbuyer '
                  'offer frequency. Their ratio can update purchase odds, but it is '
                  'evidential rather than causal: a rule firing does not prove that the '
                  'coupon caused a purchase. Small review counts create wide uncertainty, '
                  'so posterior samples must be propagated into the odds calculation.\n'
                  '        ',
      'terms': (('Observed variable',
                 'A value recorded in the data, such as a reviewed count.'),
                ('Latent variable',
                 'An unknown model quantity inferred from observed values.'),
                ('Deterministic transform',
                 'A quantity calculated exactly from latent variables.'),
                ('Logit', 'The map log(p/(1-p)) from a probability to the real line.'),
                ('Likelihood ratio',
                 'A ratio that states how much more common evidence is under one '
                 'condition.')),
      'notation': (('sⱼ', 'probability that rule j fires for a reviewed buyer'),
                   ('fⱼ', 'probability that rule j fires for a reviewed nonbuyer'),
                   ('ηⱼ, ξⱼ', 'unbounded logit-scale latent variables for sⱼ and fⱼ'),
                   ('kⱼ⁺, nⱼ⁺', 'buyer fire count and number of reviewed buyers'),
                   ('kⱼ⁻, nⱼ⁻', 'nonbuyer fire count and number of reviewed nonbuyers')),
      'math_story': '\n'
                    '        One side of the hierarchy is\n'
                    '\n'
                    '        $$\\mu_s\\sim\\mathcal N(0,2),\\quad \\sigma_s\\sim '
                    'HalfNormal(1),\\quad\n'
                    '        \\eta_j\\sim\\mathcal N(\\mu_s,\\sigma_s),$$\n'
                    '        $$s_j=\\operatorname{logit}^{-1}(\\eta_j),\\qquad\n'
                    '        k_j^+\\sim Binomial(n_j^+,s_j).$$\n'
                    '\n'
                    '        The false-fire hierarchy uses $\\mu_f,\\sigma_f,\\xi_j,$ and '
                    '$f_j$ in the same way. The transform '
                    '$\\operatorname{logit}^{-1}(a)=1/(1+e^{-a})$ keeps rates between zero '
                    'and one. For prior purchase odds $O_{prior}$, a rule gives '
                    '$O_{post}=O_{prior}(s_j/f_j)$. Applying this to every posterior draw '
                    'preserves uncertainty and dependence.\n'
                    '        ',
      'visual_guide': 'Compare rate distributions before the ratio. A denominator near '
                      'zero can create a long right tail, so medians and intervals are '
                      'more useful than one ratio of posterior means.',
      'connection': 'This is the full probabilistic pipeline: model counts, infer latent '
                    'rates, transform samples, and carry uncertainty into a decision '
                    'quantity.'},
     {'overview': '\n'
                  '        Hierarchical models can be statistically sensible and '
                  'computationally difficult. Neal’s funnel is a minimal example. One '
                  'scale variable controls how widely several lower-level variables can '
                  'spread. When the scale is large, the joint distribution has a wide '
                  'mouth. When it is small, all lower-level variables must pass through a '
                  'narrow neck. HMC must use one numerical step size across both regions. '
                  'A step that is efficient in the mouth can be too coarse for the neck, '
                  'which causes energy errors and divergences.\n'
                  '        ',
      'terms': (('Funnel',
                 'A joint distribution whose conditional width changes greatly with '
                 'another parameter.'),
                ('Conditional distribution',
                 'A distribution for one variable when another variable is fixed.'),
                ('Divergence',
                 'An HMC warning that numerical integration failed to follow the target '
                 'accurately.'),
                ('Geometry',
                 'The local scales, correlations, and curvature of a probability '
                 'distribution.')),
      'notation': (('v', 'a top-level variable that controls log variance'),
                   ('xⱼ', 'a lower-level variable whose scale depends on v'),
                   ('exp(v/2)', 'conditional standard deviation of xⱼ')),
      'math_story': '\n'
                    '        A common funnel is\n'
                    '\n'
                    '        $$v\\sim\\mathcal N(0,3),\\qquad x_j\\mid v\\sim\\mathcal '
                    'N(0,\\exp(v/2)).$$\n'
                    '\n'
                    '        Because variance is $\\exp(v)$, standard deviation is '
                    '$\\exp(v/2)$. Negative $v$ makes this scale tiny, coupling all $x_j$ '
                    'values tightly to zero. Positive $v$ allows a wide range. The density '
                    'is valid everywhere, but the useful numerical scale changes by orders '
                    'of magnitude. Divergences tend to occur where leapfrog cannot resolve '
                    'the narrow geometry. Removing divergent draws without repairing the '
                    'model or parameterization does not solve the bias.\n'
                    '        ',
      'visual_guide': 'Find the narrow neck and wide mouth before reading samples. Then '
                      'locate divergence markers; their position often identifies the '
                      'unresolved region.',
      'connection': 'A valid loss function can also be badly conditioned. Statistical '
                    'correctness does not guarantee that a chosen computational method can '
                    'explore it reliably.'},
     {'overview': '\n'
                  '        A non-centered parameterization rewrites the same hierarchical '
                  'model in different coordinates. Instead of sampling a group effect '
                  'directly from a distribution whose width changes with $\\sigma$, it '
                  'samples a standard-Normal variable $z$ and constructs the effect as '
                  '$\\mu+\\sigma z$. The probability model for the final group effect is '
                  'unchanged. The geometry seen by HMC can change greatly, especially when '
                  'group data are weak and the centered variables remain strongly coupled '
                  'to the population scale.\n'
                  '        ',
      'terms': (('Centered parameterization',
                 'Sampling a group parameter directly around the population mean and '
                 'scale.'),
                ('Non-centered parameterization',
                 'Sampling a standard variable and transforming it into the group '
                 'parameter.'),
                ('Reparameterization',
                 'Changing coordinates without changing the represented probability '
                 'model.'),
                ('Weak data',
                 'Observations that do not strongly identify a group parameter on their '
                 'own.')),
      'notation': (('ηⱼ', 'group-level parameter on its modeled scale'),
                   ('μ', 'population mean'),
                   ('σ', 'population standard deviation'),
                   ('zⱼ', 'standard-Normal auxiliary coordinate')),
      'math_story': '\n'
                    '        The centered form is\n'
                    '\n'
                    '        $$\\eta_j\\sim\\mathcal N(\\mu,\\sigma).$$\n'
                    '\n'
                    '        The non-centered form is\n'
                    '\n'
                    '        $$z_j\\sim\\mathcal N(0,1),\\qquad \\eta_j=\\mu+\\sigma '
                    'z_j.$$\n'
                    '\n'
                    '        Conditional on $\\mu$ and $\\sigma$, both forms give the same '
                    'Normal distribution for $\\eta_j$. They differ in the coordinates '
                    'sampled by HMC. With weak data, $z_j$ can remain close to an '
                    'independent standard scale while $\\sigma$ changes. With very strong '
                    'group data, the centered form can sometimes be better. Non-centering '
                    'is therefore a geometry choice, not a universal rule.\n'
                    '        ',
      'visual_guide': 'Compare the same posterior in two coordinate systems. Look for a '
                      'rounder cloud, fewer divergences, higher ESS, and traces that move '
                      'across the supported range.',
      'connection': 'Reparameterization is common in deep learning too: equivalent '
                    'functions can produce very different gradient geometry and training '
                    'behavior.'},
     {'overview': '\n'
                  '        A neural network can be read through the same inference map. '
                  'Its weights are parameters, the data loss usually comes from a '
                  'likelihood, and regularization can be interpreted as a prior. Standard '
                  'SGD or Adam training finds one weight vector, so it is point inference '
                  'even when the network is large. A Bayesian neural network instead '
                  'places a posterior distribution over weights. Exact posterior sampling '
                  'is usually impractical at modern scale, so variational approximations '
                  'or other specialized methods are used.\n'
                  '        ',
      'terms': (('Weight',
                 'A learned numerical parameter that controls a neural-network '
                 'transformation.'),
                ('Cross-entropy',
                 'A common classification loss equal to negative log likelihood.'),
                ('L2 regularization',
                 'A penalty proportional to the sum of squared weights.'),
                ('Autodiff',
                 'Automatic differentiation through the network’s computation graph.'),
                ('Bayesian neural network',
                 'A neural network with a posterior distribution over weights.')),
      'notation': (('w', 'the full neural-network weight vector'),
                   ('D', 'training inputs and targets'),
                   ('L(w)', 'the training objective or loss'),
                   ('λ', 'regularization strength or prior precision scale')),
      'math_story': '\n'
                    '        Ordinary point training solves\n'
                    '\n'
                    '        $$w^*=\\arg\\min_w L(w).$$\n'
                    '\n'
                    '        If the data term is negative log likelihood and the prior is '
                    '$w\\sim\\mathcal N(0,\\lambda^{-1}I)$, MAP minimizes\n'
                    '\n'
                    '        $$-\\log p(D\\mid w)+\\frac{\\lambda}{2}\\sum_i w_i^2+C.$$\n'
                    '\n'
                    '        Thus L2 regularization has a Gaussian-prior interpretation, '
                    'although practical training choices do not automatically create '
                    'calibrated Bayesian uncertainty. SGD estimates gradients from '
                    'mini-batches. VI can instead optimize parameters of $q_\\phi(w)$, but '
                    'its approximation family and scale assumptions become critical in the '
                    'high-dimensional weight space.\n'
                    '        ',
      'visual_guide': 'Follow the diagram from data through weights to loss. Then identify '
                      'which part corresponds to likelihood, prior penalty, optimizer, and '
                      'returned object.',
      'connection': 'This lab is the bridge back to daily ML work: familiar training is '
                    'one branch of a larger inference map, not a separate subject.'},
     {'overview': '\n'
                  '        The final challenge removes the familiar labels from the '
                  'problem. Begin with the decision and ask what output it needs: one '
                  'action-driving point, soft hidden assignments, a fast approximate '
                  'distribution, or calibrated posterior samples. Then write the '
                  'probabilistic model and identify observed, latent, and deterministic '
                  'quantities. Only after the target is clear should you choose an '
                  'algorithm. Finally, name the diagnostic or failure mode that could '
                  'invalidate the result. This order prevents method choice from becoming '
                  'a list of fashionable tools.\n'
                  '        ',
      'terms': (('Decision requirement',
                 'The information an action needs, including whether uncertainty matters.'),
                ('Failure mode',
                 'A predictable way an algorithm or model can produce a misleading '
                 'result.'),
                ('Calibration',
                 'Agreement between stated probabilities and long-run frequencies under '
                 'relevant conditions.'),
                ('Computational budget', 'Available time, memory, and model evaluations.'),
                ('Sensitivity analysis',
                 'Checking how conclusions change under reasonable modeling choices.')),
      'notation': (('p(D,θ)', 'the joint probabilistic model for data and unknowns'),
                   ('p(θ | D)', 'the posterior target when full uncertainty is required'),
                   ('θ*', 'a point result from optimization'),
                   ('qφ(θ)', 'an approximate posterior returned by VI')),
      'math_story': '\n'
                    '        Use one route for every new problem:\n'
                    '\n'
                    '        $$\\text{decision}\\rightarrow\\text{model '
                    '}p(D,\\theta)\\rightarrow\\text{target}\\rightarrow\\text{algorithm}\\rightarrow\\text{diagnostics}.$$\n'
                    '\n'
                    '        MAP optimizes one posterior point. EM alternates latent '
                    'expectations and point-parameter updates. VI optimizes an approximate '
                    'distribution. MCMC targets the posterior with dependent samples, and '
                    'NUTS is a geometry-aware MCMC algorithm. Gradients can support MAP, '
                    'VI, HMC, NUTS, and neural-network training, but they do different '
                    'work because each method updates different state and targets a '
                    'different object. State the likely failure before trusting the '
                    'output.\n'
                    '        ',
      'visual_guide': 'For each scenario, move left to right through the concept map. If '
                      'you cannot name the returned object or its diagnostic, do not '
                      'choose the method yet.',
      'connection': 'The transferable skill is not reciting method definitions. It is '
                    'predicting behavior from target geometry, output requirements, and '
                    'algorithm mechanics.'})
    return (COURSE_PROSE,)


if __name__ == "__main__":
    app.run()
