# CMMST
## Datasets

Chenglei Wu, Zhihao Tan, Zhi Wang, and Shiqiang Yang. [2017]. *A Dataset for Exploring User Behaviors in VR Spherical Video Streaming.* In Proceedings of the ACM MMSys 2017. 
It has nine popular videos watched by 48 users with an average view duration of 164 seconds. The dataset is available from [https://wuchlei-thu.github.io/](https://wuchlei-thu.github.io/)

## Pipeline
1. Generate Saliency. [https://github.com/phananh1010/PanoSalNet](https://github.com/phananh1010/PanoSalNet)
2. Generate viewports from heade tracking logs. (PanoSaliency/get_viewport.py)
3. Preprocess viewports to matrix. (tools/get_act_tile_base_frame.py)
4. Run CMMST and calculate the accuracy. (tools/main.py)
