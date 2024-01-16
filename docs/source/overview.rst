========
Overview
========

**xspec** is a cutting-edge Python package, powered by the PyTorch framework, focused on the automatic estimation of X-ray CT parameters which are vital for determining the X-ray spectral energy response. These parameters include source voltage, anode take-off angle, filter material and thickness, along with the scintillator type and thickness.

Leveraging the robust computational capabilities of PyTorch, **xspec** adeptly models the X-ray system spectral response by breaking it down into three integral components: the source spectrum, the filter response, and the scintillator response. Each of these components is represented through analytical parametric models.

**xspec** can precisely estimate the parameters of each individual component. This is achieved through its capacity
to analyze normalized radiographs captured under a variety of X-ray system spectral conditions. Each spectrum
processed may involve different source voltages or filters, consistently utilizing one anode take-off angle and one
scintillator.

**xspec** handles both discrete and continuous parameters. For discrete parameters, like filter types and scintillator
types, it employs comprehensive exhaustive search techniques, whereas for continuous parameters, it harnesses the power of PyTorch for gradient descent optimization.

Additionally, **xspec** supports the calculation of spectral energy responses using the estimated system parameters , making it a highly versatile tool for X-ray spectral analysis.