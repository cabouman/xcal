from setuptools import setup, find_packages

setup(name='xspec',
      version='0.1.0',
      description='Algorithms for X-Ray Spectrum Calibration',
      author='Wenrui Li',
      author_email='li3120@purdue.edu',
      license='BSD-3',
      packages=find_packages(where="./"),
      package_dir={"": "./"},
      package_data={"xspec": ["./chem_consts/*.h5"]},
      setup_requires=[],
      install_requires=[],
      zip_safe=False)
