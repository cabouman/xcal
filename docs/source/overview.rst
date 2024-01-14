========
Overview
========

We model the X-ray system spectral response by segregating it into three distinct components: the source spectrum, filter response, and scintillator response.
For each of these components, we formulate analytical parametric representations.
Further, we model the process of bright-dark normalization in transmission data as a function of the X-ray system spectral response, with known scanned samples for calibration. Leveraging the parametric models developed for the different components, we design an algorithm to precisely estimate the parameters of each individual component from normalized radiographs, scanned under various X-ray system spectral responses.
Notably, each spectrum might utilize different source voltages or filters, while maintaining a fixed scintillator.
A critical aspect of our algorithm is its capability to adeptly manage both discrete and continuous parameters. For discrete parameters, we employ exhaustive search techniques, while for continuous parameters, we use a gradient descent optimization strategy.