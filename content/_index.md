+++
title = "Symmetry-Constrained Causal Partial Identification"
description = "Data symmetries—invariances of the causal effect under data transformations—as constraints that provably sharpen causal partial identification bounds."
[extra]
authors = [
    {name = "Uzair Akbar", url = "https://uzairakbar.com"},
    {name = "Zulfiqar Zaidi", url = "https://sites.google.com/view/zulfiqar-zaidi/home"},
    {name = "Niki Kilbertus", url = "http://nikikilbertus.info/"},
    {name = "Krikamol Muandet", url = "https://www.krikamol.org"},
    {name = "Bo Dai", url = "https://bo-dai.github.io/"},
]
venue = {name = "Preprint", date = 2026-08-30}          # TODO-for-user: venue TBD; add url (and award) keys when known
buttons = [
    {name = "Paper", url = "#"},                         # TODO-for-user: OpenReview/arXiv abs link
    {name = "PDF", url = "#"},                           # TODO-for-user: arXiv pdf link
    # {name = "Reviews", url = "#", no_icon = true},     # TODO-for-user: uncomment when OpenReview exists (theme has no reviews icon)
    {name = "Code", url = "https://github.com/uzairakbar/symmetry4CausalBounds"},
    {name = "Slides", url = "presentation.html"},
    # {name = "Poster", url = "poster.pdf"},             # TODO-for-user: uncomment when poster exists
    # {name = "Video", url = "#"},                       # TODO-for-user: uncomment when talk video exists
]
katex = true
large_card = true
favicon = true
# google analytics (TODO-for-user: paste ID, uncomment; single raw-HTML string, page-level hook)
# head_includes = '<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script><script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag("js",new Date());gtag("config","G-XXXXXXXXXX");</script>'
+++

**TLDR**: We introduce known *data symmetries*—invariance of the causal effect under certain data transformations—as a new, underutilized source of constraints for causal *partial identification (PI)*. Enforced *explicitly* via an invariance-error constraint, or *implicitly* by simple data-augmentation pre-processing, symmetry constraints provably sharpen PI bounds under two canonical PI models—in population and in finite samples.

---

# Partial Identification

Causal effects are generally not identifiable from observational data alone: hidden confounding leaves a whole set of data-generating processes consistent with what we observe. *Partial identification (PI)* responds by bounding the causal effect over the identified set `$\mathcal{H}_{\mathrm{pi}} := \{ h^q_{\star} : q \in \mathcal{Q}_{\mathrm{pi}} \}$`, encoding assumptions `$\mathcal{P}_{\mathrm{pi}}$` as constraints of an optimization problem. In practice, however, PI bounds are often too wide to inform decisions.

<!-- P5: figure slots for Fig 1 / Fig 2 go here, e.g.
{% figure(src=["sem-graph.svg", "intervention-graph.svg"], alt=["SEM with hidden confounding", "do(x) intervention"], dark_invert=[true, true]) %}
**Figure 1:** Hidden confounding (left) and the intervention of interest (right).
{% end %}
-->

# Symmetry Constraints

Many domains come with *known symmetries* `$\mathcal{T}$`: transformations of the data under which the causal effect is invariant by design, `$h_{\star}(\tau \boldsymbol{x}) = h_{\star}(\boldsymbol{x})$` for all `$\tau \in \mathcal{T}$`. We show how to exploit them for PI—*explicitly*, by constraining candidate models to have small invariance error, or *implicitly*, by augmenting the data before running an off-the-shelf PI method. Both routes provably sharpen the resulting bounds.

<!-- P5: figure slots for Fig 3 / Fig 4 go here, e.g.
{% figure(src=["da-graph.svg", "transformation-intervention-graph.svg"], alt=["DA graph", "soft intervention graph"], dark_invert=[true, true]) %}
**Figure 3:** Data augmentation as a soft intervention.
{% end %}
{% figure(src=["inv-pi.svg", "da-pi.svg", "da-iv-pi.svg", "pi-da-pi-intersection.svg"], alt=["invariance-constrained set", "post-DA set", "post-DA IV set", "intersection"], dark_invert=[false, false, false, false]) %}
**Figure 4:** Identified sets under symmetry constraints.
{% end %}
-->

# Experimental Results

Across simulations, an optical-device dataset, and a do-MNIST benchmark, symmetry constraints consistently tighten PI bounds obtained by canonical PI methods.

<!-- P5: figure slots for Figs 5-7 / Table 1 go here, e.g.
{% figure(src=["simulation-query-sweep.svg"], alt=["simulation query sweeps"]) %}
**Figure 5:** Simulation query sweeps.
{% end %}
-->

# Citation

```bibtex
@misc{akbar2026symmetry,
      title={Symmetry-Constrained Causal Partial Identification},
      author={Uzair Akbar and Zulfiqar Zaidi and Niki Kilbertus and Krikamol Muandet and Bo Dai},
      year={2026},
      eprint={TBD},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={TBD},
}
```
