Code for **Mitigating Object Dependencies: Improving Point Cloud Self-Supervised Learning through Object Exchange** (OESSL) CVPR2024

*-----------------------------------------------------------------------------------*


Todo list:

1、Now the grapcut based unsupervised segmentation is not included in this project, since it is writen using C++. I will update it into this project as soon as possiable. 

2、The fine-tune code will be updated to this project

3、I will continuously optimize this project.



**[Paper](https://arxiv.org/pdf/2404.07504)** **|** **[Project page](https://yanhaowu.github.io/OESSL/)**

![](pics/poster.png)


Our project is built based on **[STSSL](https://github.com/YanhaoWu/STSSL/)**

Installing pre-requisites:

`sudo apt install build-essential python3-dev libopenblas-dev`

`pip3 install -r requirements.txt`

`pip3 install torch ninja`

Installing MinkowskiEngine with CUDA support:

`pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`


# Data Preparation

1、Download ScanNet 
2、Next, preprocess all scannet raw point cloud follwing  "https://github.com/chrischoy/SpatioTemporalSegmentation"
3、Segment the pointclouds and generate box follwing "https://github.com/chrischoy/SpatioTemporalSegmentation/blob/master/README.md" or you can download the segments and boxes **[here](https://drive.google.com/drive/folders/10xqUBK7gLtjK9fFTGddGCGWFi1fuyj26?usp=drive_link)**.

# Reproducing the results

for pre-training. (We use 8 RXT3090 GPUs for pre-training)

you can just run train.py remember to modify the paramters of path : ) 

Then for fine-tuning:

You can refer to **[SpatioTemporalSegmentation](https://github.com/chrischoy/SpatioTemporalSegmentation)**

Any questions, touch me at wuyanhao@stu.xjtu.edu.cn

You can download the pre-trained model on ScanNet here:


# Citation

If you use this repo, please cite as :

```
@inproceedings{wu2024mitigating,
  title={Mitigating Object Dependencies: Improving Point Cloud Self-Supervised Learning through Object Exchange},
  author={Wu, Yanhao and Zhang, Tong and Ke, Wei and Qiu, Congpei and S{\"u}sstrunk, Sabine and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23052--23061},
  year={2024}
}
```
