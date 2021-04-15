# Unsupervised Single Image Super-Resolution Network (USISResNet) for Real-World Data Using Generative Adversarial Network

This is repository of code for NTIRE-2020 (CVPRW-2020) paper titled ["Unsupervised Single Image Super-Resolution Network (USISResNet) for Real-World Data Using Generative Adversarial Network"](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Prajapati_Unsupervised_Single_Image_Super-Resolution_Network_USISResNet_for_Real-World_Data_Using_CVPRW_2020_paper.pdf)

**- Framework description**

<img src="Images/5.png" width="500">

**- Results**

|Method |PSNR(dB) |SSIM | LPIPS|
|----- |----- |----- |----- |
|MsDNN |25.08 |0.7079 |0.482 |
|RCAN |25.31 |0.6402 |0.576 |
|ESRGAN |19.04 |0.2422 |0.755 |
|Proposed |21.71 |0.5895 |0.375 | 

<img src="Images/1.png" width="500">
<img src="Images/2.png" width="800">
<img src="Images/3.png" width="800">
<img src="Images/4.png" width="800">


**- Test the model**
To test/reproduce results, change `option/test/test_ntire1.json` file in which you need to change path for dataset and pre-trained model of G network.
Then you need run following command.
```javascript
python test.py -opt PATH-to-json-file
```

**- Pre-trained model**
- The pre-train model is shared in main folder named `11600_G.pth` for USISResNet.
- The pre-trained model for QA assessment network trained on KADID dataset as mentioned in the manuscript has also be included as `latest_G.pth`.

**- Required Packages**
The list of all required packages are included in `usisresnet.yml` file. You can simply import the .yml file using conda environment.

We are thankful to Xinntao for their [ESRGAN](https://github.com/xinntao/ESRGAN) code on which we have made this work.

For any problem or query, you may contact to Kalpesh Prajapati at <kalpesh.jp89@gmail.com>
