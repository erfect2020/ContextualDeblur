<div align="center">
  
# ViTDeblur: A Hierarchical Model for Defocus Deblurring

[Pengwei Liang](https://scholar.google.com/citations?user=54Ci0_0AAAAJ&hl=en), [Junjun Jiang](https://scholar.google.com/citations?user=WNH2_rgAAAAJ), [Xianming Liu](http://homepage.hit.edu.cn/xmliu), and [Jiayi Ma](https://scholar.google.com/citations?user=73trMQkAAAAJ)

Harbin Institute of Technology, Harbin 150001, China. Electronic Information School, Wuhan University, Wuhan 430072, China.
</div>

## [Paper](https://ieeexplore.ieee.org/document/10637737)

> Defocus deblurring, especially when facing spatially varying blur due to scene depth, remains a challenging problem. A more comprehensive understanding of the scene can lead to spatially adaptive deblurring techniques. While recent advancements in network architectures have predominantly addressed high-frequency details, the importance of scene understanding remains paramount. A crucial aspect of this understanding is *contextual information*. Contextual information captures vital high-level semantic cues essential for grasping the context and meaning of the input image. Beyond just providing cues, contextual information relates to object outlines and helps identify blurred regions in the image. Recognizing and effectively capitalizing on these cues can lead to substantial improvements in image recovery. With this foundation, we propose a novel method that integrates spatial details and contextual information, offering significant advancements in defocus deblurring. Consequently, we introduce a novel hierarchical model, built upon the capabilities of the Vision Transformer (ViT). This model seamlessly encodes both spatial details and contextual information, yielding a robust solution. In particular, our approach decouples the complex deblurring task into two distinct subtasks. The first is handled by a primary feature encoder that transforms blurred images into detailed representations. The second involves a contextual encoder that produces abstract and sharp representations from the primary ones. The combined outputs from these encoders are then merged by a decoder to reproduce the sharp target image. Our evaluation across multiple defocus deblurring datasets demonstrates that the proposed method achieves compelling performance. 

## 📚Framework
<p align="center">
  <img width="640" alt="image" src="https://github.com/erfect2020/ContextualDeblur/assets/94505384/c188950f-d71d-4d55-83ad-5ebb348652e7">
</p>

<p>
Given a blurred image $x \in \mathbb{R}^{C\times H \times W}$, we first tokenize the image into $h \in \mathbb{R}^{N\times D}$. The primary feature encoder takes the tokenized ${h}$ as input and learns primary representations ${h_{b0}} \in \mathbb{R}^{N\times D}$ that preserve as much detail as possible from ${h}$. Subsequently, the contextual encoder employs the ${h_{b0}}$ to learn the sharp and abstract representations ${h_{s}}$, which eliminate the irrelevant blurry features. The decoder then combines the primary and abstract representations ${h_{b0}}$ and ${h_{s}}$ as inputs to reconstruct the deblurred image ${\hat{x}} \in \mathbb{R}^{C\times H \times W}$.
</p>




## 📊Results
### RealDOF
<p align="center">
<img width="1039" alt="image" src="https://s3.bmp.ovh/imgs/2023/09/18/add3390ad5235d57.png">
</p>

### LFDOF
<p align="center">
<img width="1039" alt="image" src="https://s3.bmp.ovh/imgs/2023/09/18/4e81eaf849a6a6a0.png">
</p>

### Quantitative Results

<table border="1" cellspacing="0" cellpadding="2">
    <caption>Defocus deblurring comparisons on the DPDD test dataset, RealDOF dataset, and LFDOF dataset. Best and second best results are <strong>highlighted</strong> and <em>italics</em>.</caption>
    <colgroup>
        <col span="1" style="width: 10%;">
        <col span="1" style="width: 30%;">
        <col span="1" style="width: 30%;">
        <col span="1" style="width: 30%;">
    </colgroup> 
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
            <td>DPDNet[1] </td>
            <td>24.348</td><td>0.747</td><td>0.291</td><td>56.546</td><td>0.901</td><td>0.524</td>
            <td>22.870</td><td>0.670</td><td>0.433</td><td>26.096</td><td>0.881</td><td>0.501</td>
            <td>30.218</td><td>0.883</td><td><strong>0.144</strong></td><td>61.284</td><td>0.975</td><td>0.617</td>
        </tr>
                <tr>
            <td>AIFNet[2] </td>
            <td>24.213</td><td>0.742</td><td>0.447</td><td>46.764</td><td>0.864</td><td>0.502</td>
            <td>23.093</td><td>0.680</td><td>0.557</td><td>22.502</td><td>0.860</td><td>0.461</td>
            <td>29.677</td><td><em>0.884</em></td><td>0.202</td><td>61.481</td><td>0.976</td><td><em>0.624</em></td>
        </tr>
        <tr>
            <td>MDP[3]</td>
            <td>25.347</td><td>0.763</td><td>0.275</td><td>57.322</td><td>0.908</td><td>0.528</td>
            <td>23.500</td><td>0.681</td><td>0.407</td><td>29.023</td><td>0.892</td><td>0.527</td>
            <td>28.069</td><td>0.834</td><td>0.185</td><td>61.388</td><td>0.975</td><td>0.618</td>
        </tr>
        <tr>
            <td>KPAC[4]</td>
            <td>25.221</td><td>0.774</td><td>0.225</td><td>58.508</td><td>0.914</td><td>0.528</td>
            <td>23.975</td><td>0.762</td><td>0.355</td><td>29.611</td><td>0.903</td><td>0.533</td>
            <td>28.942</td><td>0.857</td><td>0.174</td><td>60.435</td><td>0.973</td><td>0.613</td>
        </tr>
        <tr>
            <td>IFANet[5]</td>
            <td>25.366</td><td>0.789</td><td>0.331</td><td>52.208</td><td>0.892</td><td>0.515</td>
            <td>24.712</td><td>0.748</td><td>0.464</td><td>20.887</td><td>0.878</td><td>0.472</td>
            <td>29.787</td><td>0.872</td><td>0.156</td><td>58.892</td><td>0.969</td><td>0.610</td>
        </tr>
        <tr>
            <td>GKMNet[6] </td>
            <td>25.468</td><td>0.789</td><td>0.306</td><td>55.845</td><td>0.910</td><td>0.531</td>
            <td>24.257</td><td>0.729</td><td>0.464</td><td>26.938</td><td>0.904</td><td>0.508</td>
            <td>29.081</td><td>0.867</td><td>0.171</td><td>59.038</td><td>0.969</td><td>0.605</td>
        </tr>
        <tr>
            <td>DRBNet[7] </td>
            <td>25.725</td><td>0.791</td><td>0.240</td><td>58.851</td><td>0.918</td><td>0.546</td>
            <td>24.463</td><td>0.751</td><td>0.349</td><td><em>32.483</em></td><td>0.911</td><td>0.559</td>
            <td>30.253</td><td>0.883</td><td>0.147</td><td><strong>62.648</strong></td><td><strong>0.978</strong></td><td>0.622</td>
        </tr>
        <tr>
            <td>Restormer[9] </td>
            <td>25.980</td><td><em>0.811</em></td><td><em>0.236</em></td><td>58.826</td><td>0.922</td><td>0.552</td>
            <td>24.284</td><td>0.732</td><td>0.346</td><td>31.059</td><td><em>0.921</em></td><td>0.528</td>
            <td>30.026</td><td>0.883</td><td><em>0.145</em></td><td>62.029</td><td>0.973</td><td>0.615</td>
        </tr>
        <tr>
            <td>NRKNet[8] </td>
            <td><em>26.109</em></td><td>0.810</td><td><em>0.236</em></td><td><em>59.118</em></td><td><em>0.925</em></td><td><em>0.546</em></td>
            <td><strong>25.148</strong></td><td><em>0.768</em></td><td>0.361</td><td>30.237</td><td><em>0.921</em></td><td><em>0.561</em></td>
            <td><em>30.481</em></td><td><em>0.884</em></td><td>0.147</td><td>61.738</td><td>0.976</td><td>0.620</td>
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



If this repo help you, please cite us:
```
@article{liang2024vitdeblur,
  title={Decoupling Image Deblurring Into Twofold: A Hierarchical Model for Defocus Deblurring},
  author={Liang, Pengwei and Jiang, Junjun and Liu, Xianming and Ma, Jiayi},
  journal={IEEE Transactions on Computational Imaging},
  year={2024},
  pages={1207-1220},
  volume={10},
  publisher={IEEE}
}
```


## Reference
[1] A. Abuolaim and M. S. Brown, “Defocus deblurring using dual-pixel data,” in Proceedings of the European Conference on Computer Vision, 2020. [\[code](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel),[paper](https://www.eecs.yorku.ca/~abuolaim/eccv_2020_dp_defocus_deblurring/)\]  
[2] L. Ruan, B. Chen, J. Li, and M.-L. Lam, “Aifnet: All-in-focus image restoration network using a light field-based dataset,” IEEE Transactions on Computational Imaging, vol. 7, pp. 675–688, 2021.[\[code](https://github.com/binorchen/AIFNET),[paper](https://ieeexplore.ieee.org/document/9466450)\]  
[3] A. Abuolaim, M. Afifi, and M. S. Brown, “Improving single-image defocus deblurring: How dual-pixel images help through multi-task learning,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2022, pp. 1231–1239.[\[code](https://github.com/Abdullah-Abuolaim/multi-task-defocus-deblurring-dual-pixel-nimat),[paper](https://arxiv.org/pdf/2108.05251.pdf)\]  
[4] H. Son, J. Lee, S. Cho, and S. Lee, “Single image defocus deblurring using kernel-sharing parallel atrous convolutions,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp.2642–2650.[\[code](https://github.com/HyeongseokSon1/KPAC),[paper](https://arxiv.org/abs/2108.09108)\]  
[5] J. Lee, H. Son, J. Rim, S. Cho, and S. Lee, “Iterative filter adaptive network for single image defocus deblurring,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 2034–2042.[\[code](https://github.com/codeslake/IFAN),[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Iterative_Filter_Adaptive_Network_for_Single_Image_Defocus_Deblurring_CVPR_2021_paper.pdf)\]  
[6] Y. Quan, Z. Wu, and H. Ji, “Gaussian kernel mixture network for single image defocus deblurring,” Advances in Neural Information Processing Systems, vol. 34, pp. 20 812–20 824, 2021.[\[code](https://github.com/csZcWu/GKMNet),[paper](https://proceedings.neurips.cc/paper/2021/file/ae1eaa32d10b6c886981755d579fb4d8-Paper.pdf)\]  
[7] L. Ruan, B. Chen, J. Li, and M. Lam, “Learning to deblur using light field generated and real defocus images,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 304–16 313.[\[code](https://github.com/lingyanruan/DRBNet),[paper](https://arxiv.org/pdf/2204.00367.pdf)\]  
[8] Y. Quan, Z. Wu, and H. Ji, “Neumann network with recursive kernels for single image defocus deblurring,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 5754–5763.[\[code](https://github.com/csZcWu/NRKNet),[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Quan_Neumann_Network_With_Recursive_Kernels_for_Single_Image_Defocus_Deblurring_CVPR_2023_paper.html)\]  
[9] Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., & Yang, M. H., “Restormer: Efficient Transformer for High-Resolution Image Restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5728-5739.[\[code](https://github.com/swz30/Restormer),[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)\]
