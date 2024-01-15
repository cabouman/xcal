# xspec

xspec is a Python package for automatically estimating the X-ray CT parameters that determine the X-ray energy spectrum including the source voltage, filter material and thickness, and scintillator and thickness. The package takes as input views of a known material target at different energies.

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
source ./install_all.sh
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

3. **Build documentation**
   After the package is installed, you can build the documentation.
   In your terminal window:

   a. Install required dependencies

   ```bash
   cd docs/
   conda install pandoc
   pip install -r requirements.txt
   ```

   b. Build documentation

   ```bash
   make clean html
   cd ..
   ```

   c. Open documentation in docs/build/html/index.html. You will see API references on that webpage.

4. **Install demo requirement**

   ```bash
   pip install -r demo/requirements.txt
   ```

## Step 3: Run Demo

a. Go to folder demo/

```bash
cd demo/
```

b. Run demo 1: 3 Datasets scanned with 3 different source voltages and the same filter and scintillator.

```bash
python demo_spec_est_3_voltages.py
```