"""zop: prediction-powered inference when the labelling probability is exactly zero.

Modules
-------
data        CropGlobe loading, domain registry, reproducible splits
maps        the crop map (a classifier trained in one country)
overlap     feature-space positivity diagnostic (domain classifier)
estimators  plug-in, transported PPI variants, label-shift, IPW-PPI, in-target PPI/PPI++
bounds      identified set for target crop shares under a bounded error-profile shift (LP)
"""

__version__ = "0.1.0"
