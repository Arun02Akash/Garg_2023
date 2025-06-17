
====================================================================
             SETUP GUIDE FOR EVALUATION 
====================================================================

1. Download the Pretrained Models:
    - Download the `models` folder from the following Google Drive link:
https://drive.google.com/drive/folders/1GjQeag4usn3Qomp_0wk0qy-jsp7Y9z8c?usp=drive_link
Place this folder inside your main project directory.

2. Project Folder Structure:
    - Ensure that your main project directory follows this structure:

Main Project Folder/
├── models/
├── src/
├── .gitignore
├── environment.yml
├── License
├── README.md
├── README_Evaluation_Setup.txt
├── README_HPRC_Setup.txt
├── README_Local_Setup.txt
├── run.sh
└── setting.jpg

3. Open the Project in VS Code:
    - Launch Visual Studio Code and open the main project folder.

4. Navigate to the Evaluation Notebook:
    - Open the `eval.ipynb` notebook located in the `src` folder.

5. Run the Evaluation:
    - Execute the notebook cells in order to evaluate the pretrained models.

6. Customize Tasks and Runs:
    - You can modify the evaluation by selecting different tasks or `run_id`s from the available models. These are listed during notebook execution.

7. Compare Baseline Model Performance:
    - To visualize additional baselines:
          - Edit the `relevant_model_names` dictionary in `plot_utils.py`.
          - Ensure that the corresponding baseline model was evaluated during training alongside the Transformer, as           defined in `models.py`.
          - Refer to Note 5 in `README_HPRC_Setup.txt` for further details.

====================================================================