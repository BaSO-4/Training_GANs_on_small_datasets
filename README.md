# Training_GANs_on_small_datasets
To run the project, install requirements and then:
prepare dataset: 
`python ./data/data.py`
and run training:
`python main.py --mode train --save_dir models --data ./data/flowers-102`
or generating:
`python main.py --mode generate --models_path models --output_dir generations`

To run the project on Google Colab, use the file `run_on_colab.ipynb`.