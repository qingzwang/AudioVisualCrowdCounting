# AudioVisualCrowdCounting
This is the code for AudioVisual crowd counting. To use the code you need to install PyTorch-1.0 and Python 3.7.

## Dataset
We propose a new dataset for crowd counting, which is composed of around 2000 annotaed images token in different locations in China and each image corresponds to a 1 second audio clip and a density map. The images are in different illuminations. More details can be found in our paper [Ambient Sound Helps: Audiovisual Crowd Counting in Extreme Conditions](https://arxiv.org/abs/2005.07097) and you can download the dataset [here](https://doi.org/10.5281/zenodo.3828467). We also provide the original dot annotations [here](https://drive.google.com/file/d/1QA_Dn6WNsXDzVexnEC0fx_QgaUzAT9V8/view?usp=sharing), please feel free to use it.

## Train
1. Download the dataset including [images](https://zenodo.org/record/3828468/files/images.zip?download=1), [audios](https://zenodo.org/record/3828468/files/audio.zip?download=1) and [density maps](https://zenodo.org/record/3828468/files/density_maps.zip?download=1). Unzip the files and put them into the same folder, for example, *./audio_visual_data* and then switch **DATA_PATH** in *datasets/AC/setting.py* to *audio_visual_data*.
2. Download the pretrained [VGGish](https://zenodo.org/record/3839226/files/pytorch_vggish.pth?download=1) and put it into *./models/SCC_Model/* folder.
3. To train a model using raw images, setting **IS_NOISE** to **False** and **BLACK_AREA_RATIO** to **0**.
4. To train a model using low-quality images (low illumination and noisy)， setting **IS_NOISE** to **True**, **BLACK_AREA_RATIO** to **0** and **BRIGHTNESS** to **[0,1]**. The parameter **IS_RANDOM** indicates whether **BRIGHTNESS** is a fixed value or a random number during traning. Details can be found in our paper.
5. You can also change the settings in *config.py*, such as the name of the model.

## Test
After training, you can run *my_test.py* to test the trained model. Note that in *my_tester.py* we also save the predicted density map, you should switch the path **self.save_path** to your own setting.

## Acknowledgement
The repository is derived from [C-3-Framework](https://github.com/gjy3035/C-3-Framework).

## Citation
```
@article{hu2020,
  title={Ambient Sound Helps: Audiovisual Crowd Counting in Extreme Conditions},
  author={Di Hu and Lichao Mou and Qingzhong Wang and Junyu Gao and Yuansheng Hua and Dejing Dou and Xiao Xiang Zhu},
  journal={arXiv preprint},
  year={2020}
}
```
