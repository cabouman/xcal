Overview and Introduction
=========================

Xspec is a Python package designed to accurately determine the spectral response of X-ray systems by solving the inverse problem using measurements from homogeneous samples with known composition and dimensions. Xspec stands out by leveraging advanced modeling techniques and providing a flexible, user-friendly approach to analyzing X-ray systems across various applications.

User Inputs
-----------

- **Normalized Radiograph**: Users are required to provide a normalized radiograph for analysis.
- **Component Information**: Basic knowledge of the source, filter, and scintillator components used in the setup is necessary. This includes:

  - Source Types: Options include Transmission, Reflection, or Synchrotron, among others.
  - Possible Filter Materials
  - Possible Scintillator Materials
  - The specific component used in each measurement.

- **Homogeneous Samples** (Optional): Providing homogeneous samples, with their composition and dimensions known, is essential. If the user does not know the dimension of the sample, functions are provided to calibrate the dimension information from a 3D reconstruction.

Key Features and Strengths
--------------------------

Based on a Parametric Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Robust to Ill-Conditioned Problems**: Solves problems effectively with a limited number of parameters.
- **Physically Realistic**: Ensures that solutions are not only mathematically sound but also align with physical reality.

Separable Model: Source, Filter, and Scintillator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Versatile Data Use**: Eliminates the need to acquire new data for every configuration change. Users can easily substitute components in the model.

Enables Joint Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the ill-conditioned nature of the inverse problem, accurately estimating scintillator parameters can be challenging. Xspec enhances the precision of spectral estimation through joint parameter estimation, leveraging:

- **Multi-Voltage Datasets**
- **Multi-Filter Datasets**

Modular Design
~~~~~~~~~~~~~~

- **Customizable Components**: Users can integrate their own models of sources, filters, and scintillators based on specific application needs. This modularity ensures that Xspec can be tailored to a wide range of X-ray analysis scenarios, accommodating the complexity and diversity of real-world applications.
