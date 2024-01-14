======
Theory
======

.. _ssec:src_model:


X-ray System Spectral Response Model
------------------------------------

Source Model
^^^^^^^^^^^^

The source response characterizes the distribution of the expected number of electron photons emitted by a polychromatic source over various energy bins. Determining a universal analytical model for X-ray sources presents a significant challenge due to the variety of X-ray tubes available and ongoing research in this area. Consider two different types of X-ray sources:

- Transmission Sources: These vary based on target thickness and diamond used. Low-energy sources typically employ thin anode materials like Copper (Cu), Cobalt (Co), Molybdenum (Mo), and Tungsten (W), combined with a diamond. High-energy sources often use tungsten and copper with specific thickness requirements.

- Reflection Sources: Here, the target thickness is usually considered infinite, with Tungsten (W) being the primary material. Important parameters include tilt and observation angles, influencing the heel effect, and inherent filters like Beryllium (Be) and Steel.

1. Given these intricacies, an interpolation function over simulated source spectra is created using software like MCNP and SpekPy. The interpolation function for the source voltage variation is defined as:

.. math::

   f_0(r v_1 + (1-r) v_0) = -\frac{r}{1-r} f_1(r v_1 + (1-r) v_0) \quad \text{for } r \in [0, 1].

The interpolation function for emitted photons with base anode angle :math:`\psi_0` is:

.. math::

   S^{sr}(E, v, \psi_0; D^{sr}, V) = r \cdot D_{k^{sr}+1}^{sr}(E, \psi_0) + (1-r) \cdot D_{k^{sr}}^{sr}(E, \psi_0),

where :math:`r = \frac{v  -  V_{k^{sr}}}{V_{k^{sr}+1}  -  v}`.

2. The Philibert absorption correction factor for an analytical model with anode angle is:

.. math::

   Ph(E, v, \psi) = \frac{1}{1+\frac{\mu_{MAC}(E)}{\kappa \sin (\psi)}} \cdot \frac{1}{1+\left(\frac{h}{1+h}\right) \frac{\mu_{MAC}(E)}{\kappa \sin (\psi)}},

where :math:`\mu_{MAC}(E)` is the mass absorption coefficient of the X-ray tube anode. The factor :math:`h` is defined as :math:`\frac{1.2 A}{Z^2}`, and :math:`\kappa` is calculated as :math:`\frac{4 \cdot 10^5}{v^{1.65} - E^{1.65}}`.

3. The adjusted source spectrum for varying anode angles is:

.. math::

   S^{sr}(E, v, \psi) = S^{sr}(E, v, \psi_0) \frac{Ph(E, v, \psi)}{Ph\left(E, v, \psi_0\right)}.

.. _ssec:fltr_model:

Filter Model
^^^^^^^^^^^^
The filter response is fundamentally influenced by the filter material composition and thickness. X-ray filters, made of materials like aluminum (Al) or copper (Cu), absorb low-energy photons from the X-ray beam. The filter response is represented as:

.. math::
   :label: equ:fltr_resp

   S^{fl}(E) = \prod_{p=1}^{N^{fl}} s^{fl}\left(E; M_p^{fl}, T_p^{fl}\right) = \mathrm{e}^{-\sum_p \mu(E, M_p^{fl}) T_p^{fl}},

where :math:`\mu(E, M_p^{fl})` is the Linear Attenuation Coefficient (LAC) of the :math:`p^{th}` filter made of material :math:`M_p^{fl}` at energy :math:`E`, and :math:`T_p^{fl}` denotes its thickness.

Scintillator Model
^^^^^^^^^^^^^^^^^^
A scintillator converts absorbed X-ray photon energies into visible light photons. The response of various scintillators, often modeled using MCNP simulations, can be represented as:

.. math::

   S^{sc}\left(E ; M^{sc}, T^{sc}\right) = \frac{\mu^{en}(E;  M^{sc})}{\mu(E;  M^{sc})}\left(1 - e^{-\mu(E;  M^{sc}) T^{sc}}\right) E,

where :math:`\mu^{en}(E;  M^{sc})` is the linear energy-absorption coefficient of the scintillator made of :math:`M^{sc}` and :math:`\mu` represents the LAC of the scintillator made of :math:`M^{sc}`.

Total X-ray Spectral Response
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The total X-ray spectral response, :math:`S(E)`, is the energy-wise product of the source spectrum :math:`S^{sr}(E)`, the filter response :math:`S^{fl}(E)`, and the scintillator response :math:`S^{sc}(E)`:

.. math::

   S(E) = S^{sr}(E) \cdot S^{fl}(E) \cdot S^{sc}(E).

Forward Model of X-ray Measurement
----------------------------------
The forward model begins with an X-ray source, passes through filters, and ends at a scintillator:

.. math::

   \bar{I}_{\text{obj}} = \int_0^{E_{\max}} S^{sr}(E) \cdot S^{fl}(E) \cdot e^{-\int_L \mu(E, r) \, d r} \cdot S^{sc}(E) \, d E.

In CT systems, X-ray measurements are treated as Poisson random variables, transforming to:

.. math::

   I_{\text{obj}} \sim \operatorname{Poisson}(\bar{I}_{\text{obj}})

The Beer–Lambert law informs the ideal transmission function:

.. math::

   T = \frac{\bar{I}_{\text{obj}}}{I_{\text{blank}}} = \frac{\int_0^{E_{\max}} S(E) \cdot e^{-\int_L \mu(E, r) \, d r} \, d E}{\int_0^{E_{\max}} S(E) \, d E},

where :math:`I_{\text{blank}}` is an average of multiple X-ray measurements without an object.

To correct for dark current in detectors, bright-dark normalization is used:

.. math::

   y = \frac{I_{\text{obj}}}{I_{\text{blank}}} = \frac{I_{\text{scan}} - I_{\text{dark}}}{I_{\text{bright}} - I_{\text{dark}}} = T + \tau

with :math:`\tau \sim N(0, \frac{\bar{I}_{\text{obj}}}{I^2_{\text{blank}}})` representing additive Gaussian noise.

MAP Cost Function
^^^^^^^^^^^^^^^^^
1. The MAP cost function for a single-polychromatic dataset is:

.. math::

   l(\theta^{sr}_{a_k}, \{\theta^{fl}_{p} \mid p \in B_k\}, \theta^{sc}) = \frac{1}{2}\|\boldsymbol{y}^{(k)} - \boldsymbol{A} \boldsymbol{x}^{(k)} \|_{\Lambda^{(k)}}^2,
where :math:`l` is a function of the :math:`a_k`-th source parameter :math:`\theta^{sr}_{a_k}`, a set of filter parameters :math:`\left\{\theta^{fl}_{p} \right\}` determined by the index set :math:`B_k`, and the scintillator parameter :math:`\theta^{sc}`; :math:`\Lambda^{(k)}` can be an identity matrix or a diagonal matrix with :math:`\Lambda^{(k)}_{i, i} = \frac{I_{\text{blank},i}}{y_i}`.

2. For multi-polychromatic datasets, it extends to:

.. math::

   L(\Theta) = \sum_{k=1}^{K} l(\theta^{sr}_{a_k}, \{\theta^{fl}_{p} \mid p \in B_k\}, \theta^{sc}),
where :math:`\Theta` denotes the aggregate set of parameters across all datasets, with :math:`K` representing the total number of single-polychromatic datasets. The parameter set :math:`\Theta` is composed of the source parameters :math:`\left\{\theta^{sr}_{a} \mid a = 1, \ldots, N_a\right\}`, the filter parameters :math:`\left\{\theta^{fl}_{b} \mid b = 1, \ldots, N_b\right\}`, and the scintillator parameter :math:`\theta^{sc}`.
