# xspec

Python code for X-ray spectral estimation.

## Download source code from GitHub

```
git clone --recursive https://github.com/cabouman/xspec.git
cd xspec
touch xspec/opt/_pytorch_lbfgs/__init__.py
touch xspec/opt/_pytorch_lbfgs/functions/__init__.py
```

## Installing xspec

- (recommended) Create a clean virtual environment, such as

```
conda env create -f environment.yml
conda activate xspec
```

- To install with source, go to the root folder and run

```
pip install .
```



## Build documentation
After the package is installed, you can build the documentation.
In your terminal window, 
- Go to folder docs/
```
cd docs/
```
- Install required dependencies
```
pip install -r requirements.txt
```
- Build documentation
```
make html
```
- Open documentation in build/html/index.html. You will see API references on that webpage.


