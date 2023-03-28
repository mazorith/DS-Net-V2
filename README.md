# Dynamic Slimmable Network-V2 (DS-Net)

This repo is a copied version of [DS-Net](https://github.com/changlin31/DS-Net) for further study

### For developement team:

- Here is our current [SuperNet](https://github.com/mazorith/DS-Net-V2/releases/download/Pref_Files/Slim-Sys-Super-1.pth.tar)

- Here are the extra gate [train](https://github.com/mazorith/DS-Net-V2/releases/download/Pref_Files/DS-Netgate_train_dict.p) and [validation](https://github.com/mazorith/DS-Net-V2/releases/download/Pref_Files/DS-Netgate_val_dict.p) labels

- Here are the edited timm [dataset](https://github.com/mazorith/DS-Net-V2/releases/download/Pref_Files/dataset.py) and [loader](https://github.com/mazorith/DS-Net-V2/releases/download/Pref_Files/loader.py) files. For these, you will need to find your timm directory, typically under 
``` anaconda/envs/[YOUR_ENV]/lib/[python_version]/site-packages/timm/data/ ```
and replace the two files there with the ones above
