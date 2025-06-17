
====================================================================
             SETUP GUIDE FOR TAMU GRACE HPRC (with MobaXterm)
====================================================================

1. Download and Install MobaXterm:
   - Visit: https://mobaxterm.mobatek.net/download.html
   - Download the *Home Edition Installer* and install it on your system.

2. Launch MobaXterm and Start a New SSH Session:
   - Click the **Session** button in the top-left corner.
   - Choose **SSH** from the options.

3. Fill in the SSH Configuration:
   - Remote host: grace.hprc.tamu.edu
   - Check the box for "Specify username" and enter your **NetID**.
   - Port: 22 (default)
   - Ensure **X11 forwarding** is checked (for remote GUI apps).

4. Click **OK** to connect and enter your TAMU NetID password when prompted.

5. Navigate to your scratch directory:
   cd /scratch/user/<net-id>/

6. Setup Miniconda in your environment:
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
   bash ~/miniconda.sh -u -p ~/miniconda3
   /scratch/user/<net-id>/miniconda3/bin/conda init
   source ~/.bashrc

   - Confirm installation:
     conda --version
     conda info --envs
     You should see:
     base  *  /scratch/user/<net-id>/miniconda3

   - If `conda` is still not recognized:
     - Add this to `~/.bashrc`:
       export PATH="/scratch/user/<net-id>/miniconda3/bin:$PATH"
     - Reload config:
       source ~/.bashrc

7. Upload the `in-context-learning-main` project folder into:
   /scratch/user/<net-id>/

8. Navigate to the project directory:
   cd /scratch/user/<net-id>/in-context-learning-main/

9. Modify `environment.yml`:
   - Change the pip version from `21.2.4` to `21.1.3`.

10. Create and activate the Conda environment:
    conda env create -f environment.yml
    conda activate in-context-learning

11. Install PyTorch with CUDA:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

12. Verify CUDA GPU availability:
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
    Output should resemble:
    2.0.1 True

13. Install any missing packages manually if required:
    pip install <package-name>

14. Set up your Weights & Biases (wandb) account:
    - Go to: https://wandb.ai/site
    - Sign up and verify your email
    - Choose a username
    - Create a team/organization (e.g., <net-id>-texas-a-m-university)
    - Update the `entity` field in `src/conf/wandb.yaml`:
      entity: <net-id>-texas-a-m-university
    - Log in via terminal:
      wandb login
      Paste your API key when prompted.

15. Place your `run.sh` script in the project directory:
    /scratch/user/<net-id>/in-context-learning-main/

16. Execute the training job:
    sbatch run.sh

17. To change training tasks or model settings:
    - Modify:
      - `src/conf/toy.yaml` for training configs
      - `src/conf/models/standard.yaml` for model architecture

18. Training output will be saved in the directory specified by `out_dir` in toy.yaml.

19. Sync offline wandb logs (after training):
    wandb sync wandb/offline-run-*

20. The output generated from program execution can be found in output.txt and error.txt

Notes:

1. When training the sparse_linear_regression task, always specify sparsity under task_kwargs in toy.yaml.

2. task_kwargs in toy.yaml can also be used to pass various other parameters specific to each task such as depth for decision trees, scale and noise_std for noisy regressions, and hidden_layer_size for two-layer neural networks.

3. Ensure that model.n_positions is greater than or equal to training.points.end in the config.

4. Similarly, model.n_dims must be greater than or equal to training.dims.end.

5. To compare additional baseline models on each task, add them to the task_to_baselines dictionary inside the get_relevant_baselines function in models.py.

====================================================================
