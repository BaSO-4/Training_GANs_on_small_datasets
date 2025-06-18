# Training_GANs_on_small_datasets
To run the project, install requirements and then:
prepare dataset: 
`python ./data/data.py`
and run training:
`python main.py --mode train --data ./data/flowers-102 --outdir ./outputs --batch 4 --resolution 128`
use `--mode generate` to generate instances from learned or pretrained generators. For further help use `-h` flag.


To run the project on Google Colab, use the file `run_on_colab.ipynb`.