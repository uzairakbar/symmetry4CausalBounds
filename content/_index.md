+++
title = "Symmetry-Informed Causal Partial Identification"
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
    # {name = "Paper", url = "#"},                       # TODO-for-user: venue paper link once published
    {name = "PDF", url = "https://arxiv.org/pdf/2610.09230"},
    {name = "Reviews", url = "https://openreview.net/forum?id=OeK46xQ84V"},
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
img[src$="optical-query-sweep.svg"] { max-width: 80%; }
img[src$="cigarettes.svg"] { max-width: min(100%, 460px); }
</style>

**TLDR**: 
*Partial identification (PI)* aims to bound causal effects under assumptions that may not be sufficient for point-identification. We introduce known *data symmetries*—invariance of the causal effect under certain data transformations—as a new, underutilized source of constraints to sharpen PI bounds.

---

# Partial Identification

Causal effects are generally not identifiable from observational data alone: hidden confounding leaves a whole set of data-generating processes consistent with what we observe. *Partial identification (PI)* responds by bounding the causal effect over the identified set `$\mathcal{H}_{\mathrm{p}\textnormal{\i}} := \{ h^q_{\star} : q \in \mathcal{Q}_{\mathrm{p}\textnormal{\i}} \}$`, encoding assumptions `$\mathcal{P}_{\mathrm{p}\textnormal{\i}}$` as constraints of an optimization problem. In practice, however, PI bounds are often too wide to inform decisions.

<div class="figrow">
<figure>
<div class="hug-top-always" style="column-gap: 1rem;">
<img class="dark-invert" loading="lazy" src="sem-graph.svg" alt="SEM with hidden confounding" style="height:185px">
<img class="dark-invert" loading="lazy" src="intervention-graph.svg" alt="do(x) intervention" style="height:185px">
</div>
<figcaption><b>Figure 1:</b> The observational SEM with hidden confounding (left) and the intervention of interest (right).</figcaption>
</figure>
<figure>
<div class="hug-top-always" style="column-gap: 1rem;">
  <img class="dark-invert" loading="lazy" src="pi-x.svg" alt="PI interval at a query point" style="max-height:200px">
  <img class="dark-invert" loading="lazy" src="pi-metrics.svg" alt="identified set with PI metrics" style="max-height:192.5px">
</div>
<figcaption><b>Figure 2:</b> PI sets <code>$\mathcal{H}_{\mathrm{p}\textnormal{\i}}(\boldsymbol{x})$</code> (left) and <code>$\mathcal{H}_{\mathrm{p}\textnormal{\i}}$</code> (right) for various sensitivity parameters <code>$\boldsymbol{\gamma}$</code> and different metrics of interest: estimation, approximation, and worst-case errors.</figcaption>
</figure>
</div>

# Symmetry-Informed Bounds

Many domains come with *known symmetries* `$\mathcal{T}$`: transformations of the data under which the causal effect is invariant by design, `$h_{\star}(\tau \boldsymbol{x}) = h_{\star}(\boldsymbol{x})$` for all `$\tau \in \mathcal{T}$`. We show how to exploit them for PI—*explicitly*, by constraining candidate models to have small invariance error, or *implicitly*, by augmenting the data before running an off-the-shelf PI method. Both routes provably sharpen the resulting bounds.

<div class="figrow">
<figure>
<div class="hug-top-always" style="column-gap: 1rem;">
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

On an optical-device benchmark, US cigarette demand data, and a do-MNIST benchmark, symmetry-informed PI stays valid while sharpening the bounds obtained by canonical PI methods.

{% figure(src=["optical-query-sweep.svg"], alt=["optical device query sweeps"], dark_invert=[true]) %}
**Figure 5:** Optical device. Across queries along the principal components of the inputs and a radial sweep, symmetries reduce interval width and worst-case error over baseline while maintaining validity.
{% end %}

{% figure(src=["cigarettes.svg"], alt=["US cigarette demand price elasticity bounds"], dark_invert=[true]) %}
**Figure 6:** US cigarette demand. State (top) and neighbor (bottom) price elasticity across confounding (left) and IV (right) budgets: imposing the symmetry makes the neighbor-price effect clearly positive unless the IV is heavily leaky.
{% end %}

{% figure(src=["do-mnist-sweep.svg"], alt=["do-MNIST digit sweep"], dark_invert=[true]) %}
**Figure 7:** do-MNIST. Bounds on 10 exemplar images with alternating colors: ERM follows color, while symmetries sharpen PI.
{% end %}

# Citation

```bibtex
@misc{akbar2026symmetry4CausalBounds,
      title={Symmetry-Informed Causal Partial Identification},
      author={Uzair Akbar and Zulfiqar Zaidi and Niki Kilbertus and Krikamol Muandet and Bo Dai},
      year={2026},
      eprint={2610.09230},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.09230},
}
```
