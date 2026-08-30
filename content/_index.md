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

{% figure(src=["sem-graph.svg", "intervention-graph.svg"], alt=["SEM with hidden confounding", "do(x) intervention"], dark_invert=[true, true]) %}
**Figure 1:** The observational SEM with hidden confounding (left) and the intervention of interest (right).
{% end %}

{% figure(src=["pi-x.svg", "pi-metrics.svg"], alt=["PI interval at a query point", "identified set with PI metrics"], dark_invert=[true, true]) %}
**Figure 2:** The PI interval at a query point (left) and the identified-set geometry behind common PI metrics (right).
{% end %}

# Symmetry Constraints

Many domains come with *known symmetries* `$\mathcal{T}$`: transformations of the data under which the causal effect is invariant by design, `$h_{\star}(\tau \boldsymbol{x}) = h_{\star}(\boldsymbol{x})$` for all `$\tau \in \mathcal{T}$`. We show how to exploit them for PI—*explicitly*, by constraining candidate models to have small invariance error, or *implicitly*, by augmenting the data before running an off-the-shelf PI method. Both routes provably sharpen the resulting bounds.

{% figure(src=["da-graph.svg", "transformation-intervention-graph.svg"], alt=["data augmentation graph", "soft intervention graph"], dark_invert=[true, true]) %}
**Figure 3:** Data augmentation (left) acts on the treatment like a soft intervention (right).
{% end %}

{% figure(src=["inv-pi.png", "da-pi.svg", "da-iv-pi.svg", "pi-da-pi-intersection.png"], alt=["invariance-constrained identified set", "post-DA identified set", "post-DA IV identified set", "intersection of identified sets"], dark_invert=[true, true, true, true]) %}
**Figure 4:** Identified sets under symmetry constraints: the explicit invariance constraint, the post-DA set, the post-DA IV set, and the robust intersection with the baseline.
{% end %}

{% figure(src=["table-point-vs-partial.svg"], alt=["point vs partial identification under DA"], dark_invert=[true]) %}
**Table 1:** Point vs. *partial* identification under DA.
{% end %}

# Experimental Results

Across simulations, an optical-device dataset, and a do-MNIST benchmark, symmetry constraints consistently tighten PI bounds obtained by canonical PI methods.

{% figure(src=["simulation-query-sweep.svg"], alt=["simulation query sweeps"], dark_invert=[true]) %}
**Figure 5:** Query sweeps on simulated data: symmetry constraints consistently tighten PI bounds.
{% end %}

{% figure(src=["optical-query-sweep.svg"], alt=["optical device query sweeps"], dark_invert=[true]) %}
**Figure 6:** Query sweeps on the optical-device dataset.
{% end %}

{% figure(src=["do-mnist-sweep.svg"], alt=["do-MNIST digit sweep"], dark_invert=[true]) %}
**Figure 7:** do-MNIST digit sweep: post-DA bounds with invariance constraints are the sharpest valid bounds.
{% end %}

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
