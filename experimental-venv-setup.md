# Setting Up an Experimental Virtual Environment

## Step 1: Create a New Virtual Environment

```bash
# From your CPSC_393 directory
cd /Users/kavinravi/Documents/CPSC_393

# Create a new venv called 'venv-experimental'
python3 -m venv venv-experimental
```

## Step 2: Activate the New Environment

```bash
source venv-experimental/bin/activate
```

You should see `(venv-experimental)` in your terminal prompt.

## Step 3: Install Required Packages

```bash
# Upgrade pip first
pip install --upgrade pip

# Install Jupyter and ipykernel to use this venv in Jupyter
pip install jupyter ipykernel

# Install your data science packages
pip install pandas numpy matplotlib plotnine scikit-learn

# Install TensorFlow/Keras (with legacy support if needed)
pip install tensorflow keras-opt

# Add any other packages you need
# pip install tensorflow-probability  # if needed
```

## Step 4: Register the Environment as a Jupyter Kernel

```bash
# This makes your experimental venv available in Jupyter
python -m ipykernel install --user --name=cpsc393-experimental --display-name "Python (CPSC 393 Experimental)"
```

## Step 5: Use the New Kernel in Jupyter

1. Open Jupyter Notebook/Lab
2. When opening a notebook (like optimizers.ipynb), click on the kernel name in the top right
3. Select "Python (CPSC 393 Experimental)" from the dropdown
4. Now that notebook will use your experimental environment!

## Managing Multiple Environments

**Main venv**: Use for stable work (assignments, established notebooks)
- Kernel name: Your existing setup

**Experimental venv**: Use for testing new packages, legacy Keras, experimental code
- Kernel name: "Python (CPSC 393 Experimental)"

## Switching Between Environments

In Jupyter:
- Kernel → Change Kernel → Select the environment you want

In Terminal:
```bash
# Deactivate current venv
deactivate

# Activate main venv
source venv/bin/activate

# Or activate experimental venv
source venv-experimental/bin/activate
```

## Verifying Your Current Environment

In a notebook cell, run:
```python
import sys
print(sys.executable)
print("TF_USE_LEGACY_KERAS:", os.environ.get("TF_USE_LEGACY_KERAS", "Not set"))
```

## Removing the Experimental Environment (if needed later)

```bash
# Remove the Jupyter kernel
jupyter kernelspec uninstall cpsc393-experimental

# Delete the venv folder
rm -rf venv-experimental
```








