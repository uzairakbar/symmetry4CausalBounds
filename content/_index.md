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

<style>
.figrow { display: flex; flex-wrap: wrap; justify-content: center; align-items: flex-end; gap: 1.75rem; margin: 1.5rem 0; }
.figrow > figure { flex: 1 1 300px; margin: 0; }
.figrow > figure > div { width: auto; max-width: 100%; margin: 0 auto; display: flex; flex-wrap: wrap; justify-content: center; align-items: flex-end; }
.figrow > figure > img { align-self: center; }
img[src$="simulation-query-sweep.svg"], img[src$="optical-query-sweep.svg"] { max-width: 80%; }
</style>

**TLDR**: 
*Partial identification (PI)* aims to bound causal effects under assumptions that may not be sufficient for point-identification. We introduce known *data symmetries*—invariance of the causal effect under certain data transformations—as a new, underutilized source of constraints to sharpen PI bounds.

---

# Partial Identification

Causal effects are generally not identifiable from observational data alone: hidden confounding leaves a whole set of data-generating processes consistent with what we observe. *Partial identification (PI)* responds by bounding the causal effect over the identified set `$\mathcal{H}_{\mathrm{p}\textnormal{\i}} := \{ h^q_{\star} : q \in \mathcal{Q}_{\mathrm{p}\textnormal{\i}} \}$`, encoding assumptions `$\mathcal{P}_{\mathrm{p}\textnormal{\i}}$` as constraints of an optimization problem. In practice, however, PI bounds are often too wide to inform decisions.

<div class="figrow">
<figure>
<div class="hug-top-always" style="column-gap: 2rem;">
<img class="dark-invert" loading="lazy" src="sem-graph.svg" alt="SEM with hidden confounding" style="height:185px">
<img class="dark-invert" loading="lazy" src="intervention-graph.svg" alt="do(x) intervention" style="height:185px">
</div>
<figcaption><b>Figure 1:</b> The observational SEM with hidden confounding (left) and the intervention of interest (right).</figcaption>
</figure>
<figure>
<div class="hug-top-always" style="column-gap: 2rem;">
  <img class="dark-invert" loading="lazy" src="pi-x.svg" alt="PI interval at a query point" style="max-height:200px">
  <img class="dark-invert" loading="lazy" src="pi-metrics.svg" alt="identified set with PI metrics" style="max-height:192.5px">
</div>
<figcaption><b>Figure 2:</b> The PI interval at a query point (left) and the identified-set geometry behind common PI metrics (right).</figcaption>
</figure>
</div>

# Symmetry Constraints

Many domains come with *known symmetries* `$\mathcal{T}$`: transformations of the data under which the causal effect is invariant by design, `$h_{\star}(\tau \boldsymbol{x}) = h_{\star}(\boldsymbol{x})$` for all `$\tau \in \mathcal{T}$`. We show how to exploit them for PI—*explicitly*, by constraining candidate models to have small invariance error, or *implicitly*, by augmenting the data before running an off-the-shelf PI method. Both routes provably sharpen the resulting bounds.

<div class="figrow">
<figure>
<div class="hug-top-always">
<img class="dark-invert" loading="lazy" src="da-graph.svg" alt="data augmentation graph" style="height:185px">
<img class="dark-invert" loading="lazy" src="transformation-intervention-graph.svg" alt="soft intervention graph" style="height:185px">
</div>
<figcaption><b>Figure 3:</b> Data augmentation (left) acts like a soft intervention (right).</figcaption>
</figure>
<figure>
<img class="dark-invert" loading="lazy" src="table-point-vs-partial.svg" alt="point vs partial identification under DA" style="max-height:210px">
<figcaption><b>Table 1:</b> Point vs. <i>partial</i> identification under DA.</figcaption>
</figure>
</div>

<figure>
<div style="display: flex; align-items: flex-start;">
<figure>
<img class="dark-invert" loading="lazy" src="inv-pi.svg" alt="invariance-constrained identified set" style="height:206px">
<figcaption>(a) <code>$\mathcal{H}_{\mathrm{p}\textnormal{\i}+\textnormal{\i}\mathrm{nv}}$</code></figcaption>
</figure>
<figure>
<img class="dark-invert" loading="lazy" src="da-pi.svg" alt="post-DA identified set" style="height:206px">
<figcaption>(b) <code>$\mathcal{H}_{\widetilde{\mathrm{p}\textnormal{\i}}}$</code></figcaption>
</figure>
<figure>
<img class="dark-invert" loading="lazy" src="da-iv-pi.svg" alt="post-DA IV identified set" style="height:206px">
<figcaption>(c) <code>$\mathcal{H}_{\widetilde{\mathrm{p}\textnormal{\i}}+\widetilde{\textnormal{\i}\mathrm{v}}}$</code></figcaption>
</figure>
<figure>
<img class="dark-invert" loading="lazy" src="pi-da-pi-intersection.svg" alt="intersection of identified sets" style="height:215px">
<figcaption>(d) <code>$\mathcal{H}_{\mathrm{p}\textnormal{\i}}(\boldsymbol{x}) \cap (\mathcal{H}_{\widetilde{\mathrm{p}\textnormal{\i}}}(\boldsymbol{x}) \pm \varepsilon)$</code></figcaption>
</figure>
</div>
<figcaption><b>Figure 4:</b> Identified sets under symmetry constraints: <i>(a)</i> explicit invariance constraints prune the baseline set; <i>(b)</i> DA pre-processing yields a sharper, better-centered set; <i>(c)</i> IV constraints enforce symmetry; <i>(d)</i> intersecting with the baseline set stays robust under arbitrary DA.</figcaption>
</figure>

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
@misc{akbar2026symmetry4CausalBounds,
      title={Symmetry-Constrained Causal Partial Identification},
      author={Uzair Akbar and Zulfiqar Zaidi and Niki Kilbertus and Krikamol Muandet and Bo Dai},
      year={2026},
      eprint={TBD},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={TBD},
}
```
