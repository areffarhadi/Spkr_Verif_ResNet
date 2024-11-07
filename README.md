# Spkr_Verif_ResNet
In this repository, we used pre-trained Resnet on the VoxBlink2 dataset for speaker verification.

## How to do
-------------------
To run the code:

1. The required packages are listed in the "requirements.txt" file and you can easily install all of them using: 
`pip install -r requirements.txt`

It would be better to make a new Python environment using `python3 -m venv myenv` , after that, activate the venv using `source myenv/bin/activate` and then install the packages.

2. download the [pretrained model](https://drive.google.com/file/d/1oaO9ZUWjYCKaWExTYTNoNdQow4wENj5V/view?usp=sharing) and copy it in the `./conf/resnet293/model_ft.pt`
3. put your dataset in `cv-corpus-17.0-2024-03-15` folder.
4. open the `cv_spkr_clean.py` file and change the `lang` and location of 'tsv files' for that specific lang.
   
