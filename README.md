<div align="center">
  
# Decoupling Image Deblurring into Twofold: A Hierarchical Model for Defocus Deblurring

[Pengwei Liang](https://scholar.google.com/citations?user=54Ci0_0AAAAJ&hl=en), [Junjun Jiang](https://scholar.google.com/citations?user=WNH2_rgAAAAJ), [Xianming Liu](http://homepage.hit.edu.cn/xmliu), and [Jiayi Ma](https://scholar.google.com/citations?user=73trMQkAAAAJ)

Harbin Institute of Technology, Harbin 150001, China. Electronic Information School, Wuhan University, Wuhan 430072, China.

</div>

> Defocus deblurring, especially when facing spatially varying blur due to scene depth, remains a challenging problem. A more comprehensive understanding of the scene can lead to spatially adaptive deblurring techniques. While recent advancements in network architectures have predominantly addressed high-frequency details, the importance of scene understanding remains paramount. A crucial aspect of this understanding is *contextual information*. Contextual information captures vital high-level semantic cues essential for grasping the context and meaning of the input image. Beyond just providing cues, contextual information relates to object outlines and helps identify blurred regions in the image. Recognizing and effectively capitalizing on these cues can lead to substantial improvements in image recovery. With this foundation, we propose a novel method that integrates spatial details and contextual information, offering significant advancements in defocus deblurring. Consequently, we introduce a novel hierarchical model, built upon the capabilities of the Vision Transformer (ViT). This model seamlessly encodes both spatial details and contextual information, yielding a robust solution. In particular, our approach decouples the complex deblurring task into two distinct subtasks. The first is handled by a primary feature encoder that transforms blurred images into detailed representations. The second involves a contextual encoder that produces abstract and sharp representations from the primary ones. The combined outputs from these encoders are then merged by a decoder to reproduce the sharp target image. Our evaluation across multiple defocus deblurring datasets demonstrates that the proposed method achieves compelling performance. 

## 📚Framework
<p align="center">
  <img width="640" alt="image" src="https://github.com/erfect2020/ContextualDeblur/assets/94505384/c188950f-d71d-4d55-83ad-5ebb348652e7">
</p>



## 📊Results
### RealDOF
<p align="center">
<img width="1039" alt="image" src="https://github.com/erfect2020/ContextualDeblur/assets/94505384/f8685ca7-8210-4b4e-8508-625ecb35f545">
</p>

### LFDOF
<p align="center">
<img width="1039" alt="image" src="https://s3.bmp.ovh/imgs/2023/09/18/cc1ed375d87b8410.png">
</p>

### Quantitative Results

<table border="1" cellspacing="0" cellpadding="2">
    <caption>Defocus deblurring comparisons on the DPDD test dataset, RealDOF dataset, and LFDOF dataset. Best and second best results are <strong>highlighted</strong> and <em>italics</em>.</caption>
    <thead>
        <tr>
            <th rowspan="2">Method</th>
            <th colspan="6">DPDD</th>
            <th colspan="6">RealDOF</th>
            <th colspan="6">LFDOF</th>
        </tr>
        <tr>
            <th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>MUSIQ</th><th>FSIM</th><th>CKDN</th>
            <th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>MUSIQ</th><th>FSIM</th><th>CKDN</th>
            <th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>MUSIQ</th><th>FSIM</th><th>CKDN</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Input</td>
            <td>23.890</td><td>0.725</td><td>0.349</td><td>55.353</td><td>0.872</td><td>0.492</td>
            <td>22.333</td><td>0.633</td><td>0.524</td><td>23.642</td><td>0.843</td><td>0.413</td>
            <td>25.874</td><td>0.777</td><td>0.316</td><td>49.669</td><td>0.915</td><td>0.524</td>
        </tr>
        <tr>
            <td>DPDNet</td>
            <td>24.348</td><td>0.747</td><td>0.291</td><td>56.546</td><td>0.901</td><td>0.524</td>
            <td>22.870</td><td>0.670</td><td>0.433</td><td>26.096</td><td>0.881</td><td>0.501</td>
            <td>30.218</td><td>0.883</td><td><strong>0.144</strong></td><td>61.284</td><td>0.975</td><td>0.617</td>
        </tr>
                <tr>
            <td>AIFNet</td>
            <td>24.213</td><td>0.742</td><td>0.447</td><td>46.764</td><td>0.864</td><td>0.502</td>
            <td>23.093</td><td>0.680</td><td>0.557</td><td>22.502</td><td>0.860</td><td>0.461</td>
            <td>29.677</td><td><em>0.884</em></td><td>0.202</td><td>61.481</td><td>0.976</td><td><em>0.624</em></td>
        </tr>
        <tr>
            <td>MDP</td>
            <td>25.347</td><td>0.763</td><td>0.275</td><td>57.322</td><td>0.908</td><td>0.528</td>
            <td>23.500</td><td>0.681</td><td>0.407</td><td>29.023</td><td>0.892</td><td>0.527</td>
            <td>28.069</td><td>0.834</td><td>0.185</td><td>61.388</td><td>0.975</td><td>0.618</td>
        </tr>
        <tr>
            <td>KPAC</td>
            <td>25.221</td><td>0.774</td><td>0.225</td><td>58.508</td><td>0.914</td><td>0.528</td>
            <td>23.975</td><td>0.762</td><td>0.355</td><td>29.611</td><td>0.903</td><td>0.533</td>
            <td>28.942</td><td>0.857</td><td>0.174</td><td>60.435</td><td>0.973</td><td>0.613</td>
        </tr>
        <tr>
            <td>IFANet</td>
            <td>25.366</td><td>0.789</td><td>0.331</td><td>52.208</td><td>0.892</td><td>0.515</td>
            <td>24.712</td><td>0.748</td><td>0.464</td><td>20.887</td><td>0.878</td><td>0.472</td>
            <td>29.787</td><td>0.872</td><td>0.156</td><td>58.892</td><td>0.969</td><td>0.610</td>
        </tr>
        <tr>
            <td>GKMNet</td>
            <td>25.468</td><td>0.789</td><td>0.306</td><td>55.845</td><td>0.910</td><td>0.531</td>
            <td>24.257</td><td>0.729</td><td>0.464</td><td>26.938</td><td>0.904</td><td>0.508</td>
            <td>29.081</td><td>0.867</td><td>0.171</td><td>59.038</td><td>0.969</td><td>0.605</td>
        </tr>
        <tr>
            <td>DRBNet</td>
            <td>25.725</td><td>0.791</td><td>0.240</td><td>58.851</td><td>0.918</td><td>0.546</td>
            <td>24.463</td><td>0.751</td><td>0.349</td><td><em>32.483</em></td><td>0.911</td><td>0.559</td>
            <td>30.253</td><td>0.883</td><td><em>0.147</em></td><td><strong>62.648</strong></td><td><strong>0.978</strong></td><td>0.622</td>
        </tr>
        <tr>
            <td>NRKNet </td>
            <td><em>26.109</em></td><td><em>0.810</em></td><td><em>0.236</em></td><td><em>59.118</em></td><td><em>0.925</em></td><td><em>0.546</em></td>
            <td><strong>25.148</strong></td><td><em>0.768</em></td><td>0.361</td><td>30.237</td><td><em>0.921</em></td><td><em>0.561</em></td>
            <td><em>30.481</em></td><td><em>0.884</em></td><td><em>0.147</em></td><td>61.738</td><td>0.976</td><td>0.620</td>
        </tr>
        <tr>
            <td>Ours</td>
            <td><strong>26.114</strong></td><td><strong>0.814</strong></td><td><strong>0.201</strong></td><td><strong>60.768</strong></td>
            <td><strong>0.934</strong></td><td><strong>0.557</strong></td>
            <td><em>25.141</em></td><td><strong>0.769</strong></td><td><strong>0.295</strong></td><td><strong>34.866</strong></td>
            <td><strong>0.932</strong></td><td><strong>0.577</strong></td>
            <td><strong>30.508</strong></td><td><strong>0.892</strong></td><td><strong>0.144</strong></td><td><em>62.164</em></td>
            <td><em>0.977</em></td><td><strong>0.625</strong></td>
        </tr>
    </tbody>
</table>


## Evaluation
```
python disttest.py -opt options/test/Unify_DDPD_Test.yaml
```
