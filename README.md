# xspec

xspec is a Python package for automatically estimating the X-ray CT parameters that determine the X-ray energy spectrum including the source voltage, filter material and thickness, and scintillator and thickness. The package takes as input views of a known material target at different energies.

Full documentation is available at [xspec_docs](https://xspec.readthedocs.io/en/latest/index.html).

## Step 1: Clone repository

```bash
git clone git@github.com:cabouman/xspec.git
cd xspec
```

## Step 2: Install xspec

Two options are listed below for installing xspec. 
Option 1 only requires that a bash script be run, but it is less flexible. 
Option 2 explains how to perform manual installation.

### Option 1: Clean install from dev_scripts

To do a clean install, use the command:

```bash
cd dev_scripts
source ./clean_install_all.sh
cd ..
```

### Option 2: Manual install

1. **Create conda environment:**
   Create a new conda environment named `xspec` using the following commands:

   ```bash
   conda remove env --name xspec --all
   conda create --name xspec python=3.10
   conda activate xspec
   conda install ipykernel
   python -m ipykernel install --user --name xspec --display-name xspec
   ```

2. **Install package:**

   ```bash
   pip install -r requirements.txt
   pip install .
   ```

3. **Install demo requirement**

   ```bash
   pip install -r demo/requirements.txt
   ```

## Step 3: Run Demo

a. Go to folder demo/

```bash
cd demo/
```

b. Run demo 1: Simulated Multi-Voltage Datasets scanned with three different source voltages and the same filter and scintillator.

```bash
python demo_spec_est_3_voltages.py
```

c. Run demo 2: Measured ALS Datasets scanned with two different filtrations.

```bash
python demo_als.py
```
