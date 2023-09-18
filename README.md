<div align="center">
  
# ContextualDeblur
Decoupling Image Deblurring into Twofold: A Hierarchical Model for Defocus Deblurring

[Pengwei Liang](https://scholar.google.com/citations?user=54Ci0_0AAAAJ&hl=en), [Junjun Jiang](https://scholar.google.com/citations?user=WNH2_rgAAAAJ), [Xianming Liu](http://homepage.hit.edu.cn/xmliu), and [Jiayi Ma](https://scholar.google.com/citations?user=73trMQkAAAAJ)

Harbin Institute of Technology, Harbin 150001, China. Electronic Information School, Wuhan University, Wuhan 430072, China.

</div>

> Defocus deblurring, especially when facing spatially varying blur due to scene depth, remains a challenging problem. A more comprehensive understanding of the scene can lead to spatially adaptive deblurring techniques. While recent advancements in network architectures have predominantly addressed high-frequency details, the importance of scene understanding remains paramount. A crucial aspect of this understanding is *contextual information*. Contextual information captures vital high-level semantic cues essential for grasping the context and meaning of the input image. Beyond just providing cues, contextual information relates to object outlines and helps identify blurred regions in the image. Recognizing and effectively capitalizing on these cues can lead to substantial improvements in image recovery. With this foundation, we propose a novel method that integrates spatial details and contextual information, offering significant advancements in defocus deblurring. Consequently, we introduce a novel hierarchical model, built upon the capabilities of the Vision Transformer (ViT). This model seamlessly encodes both spatial details and contextual information, yielding a robust solution. In particular, our approach decouples the complex deblurring task into two distinct subtasks. The first is handled by a primary feature encoder that transforms blurred images into detailed representations. The second involves a contextual encoder that produces abstract and sharp representations from the primary ones. The combined outputs from these encoders are then merged by a decoder to reproduce the sharp target image. Our evaluation across multiple defocus deblurring datasets demonstrates that the proposed method achieves compelling performance. 

### 📚The framework the proposed method
<img width="968" alt="image" src="https://github.com/erfect2020/ContextualDeblur/assets/94505384/c188950f-d71d-4d55-83ad-5ebb348652e7">



## Evaluation
```
python disttest.py -opt options/test/Unify_DDPD_Test.yaml
```
