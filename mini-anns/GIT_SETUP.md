# 🔧 Git Setup Guide for Mini-ANNs

Complete guide for setting up Git version control for your Mini-ANNs project.

## 📋 What's Included in .gitignore

### ✅ **Tracked Files (Important Project Files)**
- `README.md` - Main project documentation
- `requirements.txt` - Python dependencies
- `*.py` - Python scripts and modules
- `*.md` - Documentation files
- `*.ipynb` - Jupyter notebooks
- `*.json` - Configuration files
- `*.yml` / `*.yaml` - YAML configuration
- `*.txt` - Text files
- Directory structure (`.gitkeep` files)

### ❌ **Ignored Files (Excluded from Git)**
- `__pycache__/` - Python cache files
- `*.pyc` - Compiled Python files
- `*.pth` / `*.pt` - PyTorch model files
- `*.pkl` - Pickle files
- `data/*/` - Dataset files (auto-downloaded)
- `results/plots/*.png` - Generated plots
- `results/logs/*.csv` - Training logs
- `.ipynb_checkpoints/` - Jupyter checkpoints
- `.vscode/` - VSCode settings
- `.idea/` - PyCharm settings
- `*.log` - Log files
- `venv/` / `env/` - Virtual environments

## 🚀 Git Commands

### **Initial Setup (Already Done)**
```bash
# Initialize Git repository
git init

# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

### **First Commit**
```bash
# Create initial commit
git commit -m "Initial commit: Mini-ANNs project setup

- Complete project structure with 15 experiments
- Working experiment runner (95.58% MNIST accuracy)
- Flask API server ready
- Comprehensive documentation
- All dependencies installed and tested"
```

### **Daily Git Workflow**
```bash
# Check status
git status

# Add specific files
git add notebooks/run_experiment_02.py
git add README.md

# Commit changes
git commit -m "Add experiment 02: Mini Autoencoder"

# View commit history
git log --oneline

# View changes
git diff
```

### **Branching (Recommended)**
```bash
# Create feature branch
git checkout -b feature/new-experiment

# Work on your changes
# ... make changes ...

# Commit changes
git add .
git commit -m "Add new experiment implementation"

# Switch back to main
git checkout master

# Merge feature branch
git merge feature/new-experiment
```

## 📁 Repository Structure

### **What Gets Committed:**
```
mini-anns/
├── .gitignore                 ✅ Tracked
├── README.md                  ✅ Tracked
├── requirements.txt           ✅ Tracked
├── PROJECT_STATUS.md          ✅ Tracked
├── QUICK_REFERENCE.md         ✅ Tracked
├── RUNNING_GUIDE.md           ✅ Tracked
├── GIT_SETUP.md               ✅ Tracked
├── notebooks/
│   ├── *.ipynb               ✅ Tracked
│   └── run_experiment_*.py   ✅ Tracked
├── app/
│   └── *.py                  ✅ Tracked
├── api/
│   └── *.py                  ✅ Tracked
├── scripts/
│   └── *.py                  ✅ Tracked
├── docs/
│   └── *.md                  ✅ Tracked
├── data/
│   ├── .gitkeep              ✅ Tracked
│   ├── mnist/.gitkeep        ✅ Tracked
│   ├── fashion-mnist/.gitkeep ✅ Tracked
│   └── cifar10/.gitkeep      ✅ Tracked
└── results/
    ├── plots/.gitkeep        ✅ Tracked
    └── logs/.gitkeep         ✅ Tracked
```

### **What Gets Ignored:**
```
mini-anns/
├── __pycache__/              ❌ Ignored
├── *.pyc                     ❌ Ignored
├── *.pth                     ❌ Ignored
├── data/mnist/MNIST/raw/*    ❌ Ignored (large files)
├── results/plots/*.png       ❌ Ignored (generated)
├── results/logs/*.csv        ❌ Ignored (generated)
├── .ipynb_checkpoints/       ❌ Ignored
├── .vscode/                  ❌ Ignored
├── .idea/                    ❌ Ignored
├── venv/                     ❌ Ignored
└── *.log                     ❌ Ignored
```

## 🔄 Git Workflow Examples

### **Adding New Experiment**
```bash
# Create new experiment script
# ... create notebooks/run_experiment_02.py ...

# Add to Git
git add notebooks/run_experiment_02.py

# Commit
git commit -m "Add experiment 02: Mini Autoencoder

- Implements autoencoder for MNIST compression
- Achieves 90% reconstruction accuracy
- Includes visualization and metrics"
```

### **Updating Documentation**
```bash
# Update README
# ... edit README.md ...

# Add changes
git add README.md

# Commit
git commit -m "Update README with new experiment results

- Add experiment 02 results
- Update performance metrics
- Fix documentation links"
```

### **Fixing Issues**
```bash
# Fix bug in experiment
# ... edit notebooks/run_experiment_01.py ...

# Add fix
git add notebooks/run_experiment_01.py

# Commit
git commit -m "Fix: Resolve plotting issue in experiment 01

- Fix matplotlib backend issue
- Improve error handling
- Update visualization code"
```

## 🌐 Remote Repository Setup

### **Connect to GitHub/GitLab**
```bash
# Add remote repository
git remote add origin https://github.com/yourusername/mini-anns.git

# Push to remote
git push -u origin master

# Future pushes
git push origin master
```

### **Clone Repository**
```bash
# Clone the repository
git clone https://github.com/yourusername/mini-anns.git
cd mini-anns

# Install dependencies
pip install -r requirements.txt

# Run first experiment
cd notebooks
python run_experiment_01.py
```

## 📊 Git Best Practices

### **Commit Messages**
```bash
# Good commit messages
git commit -m "Add experiment 02: Mini Autoencoder"
git commit -m "Fix: Resolve plotting issue in experiment 01"
git commit -m "Update README with new performance metrics"
git commit -m "Add Streamlit dashboard for model demos"

# Bad commit messages
git commit -m "fix"
git commit -m "update"
git commit -m "changes"
```

### **Branch Naming**
```bash
# Good branch names
feature/experiment-02-autoencoder
bugfix/plotting-issue
docs/update-readme
enhancement/streamlit-dashboard

# Bad branch names
new-stuff
fix
test
```

### **File Organization**
```bash
# Keep related changes together
git add notebooks/run_experiment_02.py
git add docs/experiments.md
git commit -m "Add experiment 02 with documentation"

# Don't mix unrelated changes
git add notebooks/run_experiment_02.py
git add app/streamlit_app.py  # Different feature
git commit -m "Add experiment and fix app"  # Confusing
```

## 🔍 Useful Git Commands

### **Viewing History**
```bash
# View commit history
git log --oneline

# View specific file history
git log --oneline notebooks/run_experiment_01.py

# View changes in last commit
git show HEAD

# View changes between commits
git diff HEAD~1 HEAD
```

### **Undoing Changes**
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo changes to specific file
git checkout -- notebooks/run_experiment_01.py

# Undo all changes
git checkout -- .
```

### **Branching**
```bash
# List branches
git branch

# Create and switch to new branch
git checkout -b feature/new-experiment

# Switch to existing branch
git checkout master

# Merge branch
git merge feature/new-experiment

# Delete branch
git branch -d feature/new-experiment
```

## 🚀 Quick Start Commands

### **First Time Setup**
```bash
# Initialize repository
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Mini-ANNs project setup"

# Add remote (replace with your URL)
git remote add origin https://github.com/yourusername/mini-anns.git

# Push to remote
git push -u origin master
```

### **Daily Workflow**
```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Descriptive commit message"

# Push to remote
git push origin master
```

---

## 🎯 Summary

Your Mini-ANNs project is now properly set up with Git version control:

- ✅ **Comprehensive .gitignore** - Excludes unnecessary files
- ✅ **Directory structure preserved** - Using .gitkeep files
- ✅ **Ready for collaboration** - Clean repository structure
- ✅ **Best practices** - Proper file organization

**Next steps:**
1. Make your first commit
2. Set up remote repository
3. Start tracking your experiment progress
4. Collaborate with others

**Happy coding! 🚀✨**
