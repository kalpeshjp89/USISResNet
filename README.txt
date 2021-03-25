This is repository of code for NTIRE-2020 (CVPR-2020) paper titled "Unsupervised Single Image Super-Resolution Network (USISResNet) for Real-World Data Using Generative Adversarial Network"
Paper Link: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Prajapati_Unsupervised_Single_Image_Super-Resolution_Network_USISResNet_for_Real-World_Data_Using_CVPRW_2020_paper.pdf


To test/reproduce results, change "option/test/test_ntire1.json" file in which you need to change path for dataset and pre-trained model of G network.
Then you need run following command.
python test.py -opt option/test/test_ntire1.json

(pre-train model is shared in main folder named "11600_G.pth")

You can find "latest_G.pth" model which is pre-trained network for QA assessment trained on KADID dataset as mentioned in the manuscript.

Required Packages.
pytorch 1.4
opencv 3.4.2
python-lmdb 0.96

We are thankful to Xinntao for their ESRGAN code on which we have made this work.
(https://github.com/xinntao/ESRGAN)
