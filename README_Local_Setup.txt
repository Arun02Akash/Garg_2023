
====================================================================
    SETUP GUIDE FOR LOCAL MACHINE RUN (with VSCODE and MINICONDA)
====================================================================

1. Install Visual Studio Code:
   - Go to: https://code.visualstudio.com/
   - Download and install VS Code for your system (Windows/macOS/Linux).
   - Open VS Code after installation.

2. Install Miniconda:
   - Go to: https://www.anaconda.com/download/success
   - Download the Miniconda installer for your OS (64-bit, Python 3.x).
   - Run the installer and follow the setup instructions.

3. Add Miniconda to Your PATH (Windows):
   - Open the Start menu and search for “Edit the system environment variables.”
   - In the System Properties window, click Environment Variables (under the Advanced tab).
   - In the User variables section (for your username), select Path and click Edit.
   - Click New and add the following two paths:
           -  C:\Users\<username>\miniconda3
           -  C:\Users\<username>\miniconda3\Scripts

4. Open a new terminal on VSCode and confirm installation:
     conda --version
     conda info --envs
     You should see:
     base*  C:\Users\<username>\miniconda3

5. Upload the `in-context-learning-main` project folder into the folder directory of your choice:
    - C:\Users\<username>\<project-directory>

6.  Open the project folder in VS Code, then launch a terminal and navigate to the same directory using:
   cd C:\Users\<username>\<project-directory>

7. Modify `environment.yml`:
   - Change the pip version from `21.2.4` to `21.1.3`.

8. Create and activate the Conda environment:
    conda env create -f environment.yml
    conda activate in-context-learning

9. Install PyTorch with CUDA:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

10. Verify CUDA GPU availability:
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
    Output should resemble:
    2.0.1 True

11. Install any missing packages manually if required:
    pip install <package-name>

12. Set up your Weights & Biases (wandb) account:
    - Go to: https://wandb.ai/site
    - Sign up and verify your email
    - Choose a username
    - Create a team/organization (e.g., <net-id>-texas-a-m-university)
    - Update the `entity` field in `src/conf/wandb.yaml`:
      entity: <net-id>-texas-a-m-university
    - Log in via terminal:
      wandb login
      Paste your API key when prompted.

13. Execute the training job:
    python src/train.py --config src/conf/toy.yaml

14. To change training tasks or model settings:
    - Modify:
      - `src/conf/toy.yaml` for training configs
      - `src/conf/models/standard.yaml` for model architecture

15. Training output will be saved in the directory specified by `out_dir` in toy.yaml.

Notes:

1. When training the sparse_linear_regression task, always specify sparsity under task_kwargs in toy.yaml.

2. task_kwargs in toy.yaml can also be used to pass various other parameters specific to each task such as depth for decision trees, scale and noise_std for noisy regressions, and hidden_layer_size for two-layer neural networks.

3. Ensure that model.n_positions is greater than or equal to training.points.end in the config.

4. Similarly, model.n_dims must be greater than or equal to training.dims.end.

5. To compare additional baseline models on each task, add them to the task_to_baselines dictionary inside the get_relevant_baselines function in models.py.

====================================================================
