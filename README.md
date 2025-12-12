# 277B_Final_Project
Final project for 277B

Create environment. If you have all the dependencies downloaded you can skip this. CUDA related libraries are very large...

If GPU/CUDA available:

`conda env create -f environment_gpu.yaml`

If running on CPU:

`conda env create -f environment_cpu.yaml`

Activate the environment:

`conda activate lstm_model`

Open launcher:

`bash launcher.sh`