# Composed Multi-modal Retrieval: A Survey of Approaches and Applications


This repo is used for recording and tracking recent Composed Multi-modal Retrieval (CMR) works, including Composed Image Retrieval (CIR), Composed Video Retrieval (CVR), Composed Person Retrieval (CPR), etc.  

The survey can be found [here](https://arxiv.org/abs/2503.01334).

This repository is still a work in progress. If you find any work missing or have any suggestions, feel free
to [pull requests](https://github.com/kkzhang95/Awesome-Composed-Multi-modal-Retrieval/pulls). 
We will add the missing papers to this repo ASAP. 

<!--In the following, * indicates that the code is not open source yet -->


# What is Composed-Multi-modal-Retrieval (CMR)?

![figure1](https://github.com/kkzhang95/Awesome-Composed-Multi-modal-Retrieval/blob/main/images/figure1.png)
Figure1 (a) The evolution of content-based retrieval technology. (b) In the current research on composed multimodal retrieval (CMR), three main paradigms have been developed.  (c) The CMR applications, broadly categorized based on application scenarios and image domains. 

Generally, the evolution of content-based retrieval technology has witnessed the transformation from Unimodal Retrieval (UR) to Cross-modal Retrieval (CR), and then to Composed Multi-modal Retrieval (CMR). Compared with early-stage unimodal retrieval, which was limited to querying information within the same modality, as shown in Fig.1(a1), cross-modal retrieval has achieved remarkable accuracy and widespread application in the present era. This enables the search for semantically relevant content in one modality based on the instance query from another modality, e.g., using text search on images in Fig.1(a2), allowing users to make full use of these heterogeneous data. In recent years, composed multi-modal retrieval has emerged as a thriving content-based retrieval technology. Within this technical framework, as depicted in Fig.1(a3), the system aims to discover images/videos that not only bear resemblance to the given reference image/video but also allow for specific modifications based on the provided textual feedback from the user. In this sense, CMR pioneers an advanced level of interactive and conditional retrieval mechanisms, leveraging deep integration of visual and linguistic information. This integration greatly enhances the flexibility and precision of user-expressed search intents, injecting new vitality into domains such as internet search and e-commerce. Consequently, CMR exhibits vast potential and far-reaching impact as the next-generation content-based retrieval engine in real-world application scenarios.


# What are the challenges and existing lines of research in CMR?

A core of CMR is that it requires a synergistic understanding and composition of both input vision and language information as the multi-modal query. The earliest closely related studies of CMR are in the field of attribute-based fashion image retrieval, where the key difference is that the textual feedback in attribute-based fashion image retrieval is limited to the predefined attribute value (e.g., 'mini', 'white', 'red'), while CMR is the natural language with multiple words (e.g., 'showing this animal of the input image facing the camera under sunlight'), which is more flexible yet challenging. The pioneering CMR works introduced a framework where input queries combine an image with natural language instructions for desired modifications, sparking numerous subsequent approaches. Current research in CMR is primarily focused on three paradigms: (1) supervised learning-based CMR (SL-CMR), which focuses on how to design a better combination mechanism of vision and language through supervised training of annotated data, (2) zero-shot learning-based CMR (ZSL-CMR), which focuses on how to simulate and build a visual-linguistic multi-modal information combination framework without annotated data, and (3) semi-supervised learning-based CMR (SSL-CMR), which focuses on how to enhance the learning of visual-linguistic combination via generated pseudo-labeling data.
<!-- generating triplet training data through generation tools -->


## Supervised Learning-based CMR (SL-CMR)

For the SL-CMR pipeline, a notable characteristic is the requirement of annotated triplet data $(I_{r}, T_{m}, I_{t})$, which denotes the reference query image, the modified text, and the ground-truth target image, respectively. As illustrated in Fig.1(b1), for the given inputs $I_{r}$ and $T_{m}$, SL-CMR involves mining the content that should be modified in the reference image $I_{r}$ according to the text $T_{m}$, so as to learn a multi-modal compositional embedding to find the interested target image $I_{t}$. Thus, the challenges faced by SL-CMR mainly lies in addressing two issues: "Where to see", which refers to attending to the content in the reference image that needs change, and "How to change", which aims to modify the reference image based on the textual information while preserving the remaining information. In recent years, research on SL-CMR has primarily focused on three aspects: (1) data construction, focusing on labeling triples with accurate semantic difference descriptions;(2) model architecture, focusing on designing a better vision-language combiner via cross-modal feature alignment and fusion strategies, as well as the design of other novel frameworks that can be plugged; (3) loss optimization, focusing on the design of more reasonable feature combination constraints. 
Although supervised training relying on these carefully labeled data often offers high performance, SL-CMR inherently faces two shortcomings: 1) annotating such triplets is both difficult and labor-intensive, and 2) the supervised approaches trained on the collected limited and specific triplets are also hard for generalization.


### 1. Data Construction Approaches

| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2019 | [Composing text and image for image retrieval-an empirical odyssey](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vo_Composing_Text_and_Image_for_Image_Retrieval_-_an_Empirical_CVPR_2019_paper.pdf) |   -|
| ICCV 2021 | [Image retrieval on real-life images with pre-trained vision-and-language models](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/CIRPLANT)|
| ACL 2019 | [A corpus for reasoning about natural language grounded in photographs](https://arxiv.org/abs/1811.00491) |   -|
| ICCV 2023 | [Zero-shot composed image retrieval with textual inversion](https://openaccess.thecvf.com/content/ICCV2023/papers/Baldrati_Zero-Shot_Composed_Image_Retrieval_with_Textual_Inversion_ICCV_2023_paper.pdf) |   [Code](https://github.com/miccunifi/SEARLE)|
| EMNLP 2019 | [Neural Naturalist: Generating Fine-Grained Image Comparisons](https://arxiv.org/abs/1909.04101) |   -|
| IEEE 2015 | [Discovering states and transformations in image collections](https://ieeexplore.ieee.org/document/7298744) |   -|
| CVPR 2021 | [Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback](https://arxiv.org/abs/1905.12794) |   -|
| ICCV 2017 | [Automatic Spatially-aware Fashion Concept Discovery](https://arxiv.org/abs/1708.01311) |   -|
| NeurIPS 2018 | [Dialog-based Interactive Image Retrieval](https://arxiv.org/abs/1805.00145) |   -|
| AAAI 2024 | [CoVR: Learning Composed Video Retrieval from Web Video Captions](https://arxiv.org/abs/2308.14746) |   [Code](https://imagine.enpc.fr/~ventural/covr/)|
| ECCV 2024 | [EgoCVR: An Egocentric Benchmark for Fine-Grained Composed Video Retrieval](https://arxiv.org/abs/2407.16658) |   [Code](https://github.com/ExplainableML/EgoCVR/)|
| Arxiv 2024 | [Localizing Events in Videos with Multimodal Queries](https://arxiv.org/abs/2406.10079) |   -|









### 2. Model Architecture Approaches
#### 2.1 Design of the visual-linguistic combiner


| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2019 | [Composing text and image for image retrieval-an empirical odyssey](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vo_Composing_Text_and_Image_for_Image_Retrieval_-_an_Empirical_CVPR_2019_paper.pdf) |   -|
| Arxiv 2022 | [Training and challenging models for text-guided fashion image retrieval](https://Arxiv.org/pdf/2204.11004) |   [Code](https://github.com/yahoo/maaf)|
| CVPR 2022 | [Effective conditioned and composed image retrieval combining CLIP-based features](https://openaccess.thecvf.com/content/CVPR2022/papers/Baldrati_Effective_Conditioned_and_Composed_Image_Retrieval_Combining_CLIP-Based_Features_CVPR_2022_paper.pdf) |   -|
| CVPR-W 2022 | [Conditioned and composed image retrieval combining and partially fine-tuning clip-based features](https://openaccess.thecvf.com/content/CVPR2022W/ODRUM/papers/Baldrati_Conditioned_and_Composed_Image_Retrieval_Combining_and_Partially_Fine-Tuning_CLIP-Based_CVPRW_2022_paper.pdf) |   -|
| ACM TOMM 2023 | [Composed image retrieval using contrastive learning and task-oriented clip-based features](https://Arxiv.org/pdf/2308.11485) |   [Code](https://github.com/ABaldrati/CLIP4Cir)|
| AI 2023 | [CLIP-based Composed Image Retrieval with Comprehensive Fusion and Data Augmentation](https://dl.acm.org/doi/10.1007/978-981-99-8388-9_16) |   -|
| CVPR 2021 | [Cosmo: Content-style modulation for image retrieval with text feedback](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_CoSMo_Content-Style_Modulation_for_Image_Retrieval_With_Text_Feedback_CVPR_2021_paper.pdf) |   [Code](https://github.com/postBG/CosMo.pytorch)|
| WACV 2021 | [Compositional learning of image-text query for image retrieval](https://openaccess.thecvf.com/content/WACV2021/papers/Anwaar_Compositional_Learning_of_Image-Text_Query_for_Image_Retrieval_WACV_2021_paper.pdf) |   [Code](https://github.com/ecom-research/ComposeAE)
| IEEE-TMM 2024 | [Align and Retrieve: Composition and Decomposition Learning in Image Retrieval with Text Feedback](https://ieeexplore.ieee.org/document/10568424) |  -|
| CVPR 2020 | [Composed Query Image Retrieval Using Locally Bounded Features](https://ieeexplore.ieee.org/document/9157125) |   -|
| ECCV 2022 | [Fashionvil: Fashion-focused vision-and-language representation learning](https://Arxiv.org/pdf/2207.08150) |   [Code](https://github.com/BrandonHanx/mmf)|
| MM 2021 | [Heterogeneous Feature Fusion and Cross-modal Alignment for Composed Image Retrieval](https://dl.acm.org/doi/10.1145/3474085.3475659) |   -|
| IEEE TMM 2022 | [Heterogeneous Feature Alignment and Fusion in Cross-Modal Augmented Space for Composed Image Retrieval](https://ieeexplore.ieee.org/document/9899752) |   -|
| Sci. Rep. 2022 | [Composed query image retrieval based on triangle area triple loss function and combining CNN with transformer](https://www.nature.com/articles/s41598-022-25340-w) |  -|
| WACV 2022 | [SAC: Semantic attention composition for text-conditioned image retrieval](https://openaccess.thecvf.com/content/WACV2022/papers/Jandial_SAC_Semantic_Attention_Composition_for_Text-Conditioned_Image_Retrieval_WACV_2022_paper.pdf) |   -|
| UniReps 2023 | [NEUCORE: Neural Concept Reasoning for Composed Image Retrieval](https://proceedings.mlr.press/v243/zhao24a/zhao24a.pdf) |   [Code](https://github.com/VisionLanguageLab/NEUCORE)|
| IEEE TMM 2024 | [Negative-Sensitive Framework With Semantic Enhancement for Composed Image Retrieval](https://ieeexplore.ieee.org/document/10493853) |   -|
| CVPR 2020 | [Image search with text feedback by visiolinguistic attention learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Image_Search_With_Text_Feedback_by_Visiolinguistic_Attention_Learning_CVPR_2020_paper.pdf) |   [Code](https://github.com/yanbeic/VAL)|
| Arxiv 2020 | [Modality-agnostic attention fusion for visual search with text feedback](https://Arxiv.org/pdf/2007.00145) |   [Code](https://github.com/yahoo/maaf)|
| AAAI 2021 | [Trace: Transform aggregate and compose visiolinguistic representations for image search with text feedback](https://www.researchgate.net/profile/Mausoom-Sarkar/publication/344083983_TRACE_Transform_Aggregate_and_Compose_Visiolinguistic_Representations_for_Image_Search_with_Text_Feedback/links/5fea20b2299bf14088562c70/TRACE-Transform-Aggregate-and-Compose-Visiolinguistic-Representations-for-Image-Search-with-Text-Feedback.pdf) |   -|
| IEEE TIP 2023 | [Composed Image Retrieval via Cross Relation Network With Hierarchical Aggregation Transformer](https://ieeexplore.ieee.org/document/10205526) |   [Code](https://github.com/yan9qu/crn)|
| MMAsia 2021 | [Hierarchical Composition Learning for Composed Query Image Retrieval](https://dl.acm.org/doi/10.1145/3469877.3490601) |   -|
| IEEE TCSVT 2024 | [Multi-Grained Attention Network With Mutual Exclusion for Composed Query-Based Image Retrieval](https://ieeexplore.ieee.org/document/10225420) |   [Code](https://github.com/CFM-MSG/Code_MANME)|
| CVPRW 2023 | [Language Guided Local Infiltration for Interactive Image Retrieval](https://openaccess.thecvf.com/content/CVPR2023W/IMW/papers/Huang_Language_Guided_Local_Infiltration_for_Interactive_Image_Retrieval_CVPRW_2023_paper.pdf) |   -|
| SIGIR 2021 | [Comprehensive linguistic-visual composition network for image retrieval](https://haokunwen.github.io/files/acmsigir2021.pdf) |   -|
| CVPR 2022 | [FashionVLP: Vision Language Transformer for Fashion Retrieval With Feedback](https://openaccess.thecvf.com/content/CVPR2022/papers/Goenka_FashionVLP_Vision_Language_Transformer_for_Fashion_Retrieval_With_Feedback_CVPR_2022_paper.pdf) |   -|
| IEEE 2023 | [Self-Training Boosted Multi-Factor Matching Network for Composed Image Retrieval](https://ieeexplore.ieee.org/document/10373096) |   -|
| Arxiv 2021 | [Rtic: Residual learning for text and image composition using graph convolutional network](https://Arxiv.org/pdf/2104.03015) |   [Code](https://github.com/nashory/rtic-gcn-pytorch)|
| MM 2022 | [Comprehensive Relationship Reasoning for Composed Query Based Image Retrieval](https://dl.acm.org/doi/10.1145/3503161.3548126) |   -|
| SIGIR 2024 | [CaLa: Complementary Association Learning for Augmenting Composed Image Retrieval](https://arxiv.org/abs/2405.19149) |   [Code](https://github.com/Chiangsonw/CaLa)|
| ACM TOMM 2024 | [SPIRIT: Style-guided Patch Interaction for Fashion Image Retrieval with Text Feedback](https://dl.acm.org/doi/abs/10.1145/3640345) |   [Code](https://github.com/PKU-ICST-MIPL/SPIRIT_TOMM2024)|
| IEEE TIP 2021 | [Geometry Sensitive Cross-Modal Reasoning for Composed Query Based Image Retrieval](https://ieeexplore.ieee.org/abstract/document/9667308) |   -|
| ICLR 2024 | [Sentence-level prompts benefit composed image retrieval](https://Arxiv.org/pdf/2310.05473) |   [Code](https://github.com/chunmeifeng/SPRC)|
| CVPRW 2022 | [Probabilistic Compositional Embeddings for Multimodal Image Retrieval](https://arxiv.org/abs/2204.05845) |   -|
| IEEE TCSVT 2024 | [Set of Diverse Queries with Uncertainty Regularization for Composed Image Retrieval](https://ieeexplore.ieee.org/document/10530361) |   -|
| IEEE-TMM 2023 | [Enhance Composed Image Retrieval via Multi-Level Collaborative Localization and Semantic Activeness Perception](https://ieeexplore.ieee.org/document/10120671) |  -|
| IEEE 2021 | [Conversational Image Search](https://ieeexplore.ieee.org/document/9528996) |   -|
| ICCV 2021 | [Image retrieval on real-life images with pre-trained vision-and-language models](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/CIRPLANT)|








#### 2.2 Design of plug and play structure
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| MM 2022 | [Comprehensive Relationship Reasoning for Composed Query Based Image Retrieval](https://dl.acm.org/doi/10.1145/3503161.3548126) |   -|
| ECCV 2020 | [Learning joint visual semantic matching embeddings for language-guided retrieval](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670137.pdf) |   -|
| AAAI 2021 | [Dual compositional learning in interactive image retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/16271) |   [Code](https://github.com/ozmig77/dcnet)|
| ACM TOMM 2023 | [Amc: Adaptive multi-expert collaborative network for text-guided image retrieval](https://dl.acm.org/doi/abs/10.1145/3584703) |   [Code](https://github.com/KevinLight831/AMC)|
| IEEE TMM 2022 | [Heterogeneous Feature Alignment and Fusion in Cross-Modal Augmented Space for Composed Image Retrieval](https://ieeexplore.ieee.org/document/9899752) |   -|
| WACV 2024 | [Bi-directional Training for Composed Image Retrieval via Text Prompt Learning](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Bi-Directional_Training_for_Composed_Image_Retrieval_via_Text_Prompt_Learning_WACV_2024_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/Bi-Blip4CIR)|
| Arxiv 2024 \$\dagger\$ | [Pseudo Triplet Guided Few-shot Composed Image Retrieval](https://arxiv.org/abs/2407.06001) |  -|
| MM 2021 | [Cross-Modal Joint Prediction and Alignment for Composed Query Image Retrieval](https://dl.acm.org/doi/10.1145/3474085.3475483) |   -|





#### 2.3 Design of non-explicit combiner architecture
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ACM TOMM 2022 | [Tell, Imagine, and Search: End-to-end Learning for Composing Text and Image to Image Retrieval](https://dl.acm.org/doi/10.1145/3478642) |   -|
| ICLR 2022 | [Artemis: Attention-based retrieval with text-explicit matching and implicit similarity](https://Arxiv.org/pdf/2203.08101) |   [Code](https://github.com/naver/artemis)|
| Arxiv 2023 | [Decompose Semantic Shifts for Composed Image Retrieval](https://arxiv.org/abs/2309.09531) |   -|
| SIGIR 2024 | [Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval](https://arxiv.org/abs/2404.15875) |   -|





### 3. Loss Optimization Approaches
#### 3.1 Based on contrastive learning
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2019 | [Composing text and image for image retrieval-an empirical odyssey](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vo_Composing_Text_and_Image_for_Image_Retrieval_-_an_Empirical_CVPR_2019_paper.pdf) |   -|
| BMVC 2018 | [VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612) |   -|
| IEEE 2020 | [Universal Weighting Metric Learning for Cross-Modal Matching](https://ieeexplore.ieee.org/document/9156549) |   -|
| AAAI 2020 | [Ladder Loss for Coherent Visual-Semantic Embedding](https://arxiv.org/abs/1911.07528) |   -|
| ECCV 2020 | [Adaptive Offline Quintuplet Loss for Image-Text Matching](https://arxiv.org/abs/2003.03669) |   -|
| ACM TOMM 2023 | [Amc: Adaptive multi-expert collaborative network for text-guided image retrieval](https://dl.acm.org/doi/abs/10.1145/3584703) |   [Code](https://github.com/KevinLight831/AMC)|
| Arxiv 2023 | [Decompose Semantic Shifts for Composed Image Retrieval](https://arxiv.org/abs/2309.09531) |   -|
| ICCV 2021 | [Image retrieval on real-life images with pre-trained vision-and-language models](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/CIRPLANT)|
| CVPR 2020 | [Composed Query Image Retrieval Using Locally Bounded Features](https://ieeexplore.ieee.org/document/9157125) |   -|
| MM 2021 | [Cross-Modal Joint Prediction and Alignment for Composed Query Image Retrieval](https://dl.acm.org/doi/10.1145/3474085.3475483) |   -|
| ECCV 2020 | [Learning joint visual semantic matching embeddings for language-guided retrieval](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670137.pdf) |   -|
| CVPR 2020 | [Image search with text feedback by visiolinguistic attention learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Image_Search_With_Text_Feedback_by_Visiolinguistic_Attention_Learning_CVPR_2020_paper.pdf) |   [Code](https://github.com/yanbeic/VAL)|
| IEEE TIP 2024 | [Multimodal Composition Example Mining for Composed Query Image Retrieval](https://ieeexplore.ieee.org/document/10418785) |   -|
| ACM MM 2023 | [Target-Guided Composed Image Retrieval](https://Arxiv.org/pdf/2309.01366) |   -|
| ACL 2024 | [VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval](https://aclanthology.org/2024.acl-long.175/) |  [Code](https://github.com/JUNJIE99/VISTA_Evaluation_FineTuning)|
| Sci. Rep. 2022 | [Composed query image retrieval based on triangle area triple loss function and combining CNN with transformer](https://www.nature.com/articles/s41598-022-25340-w) |  -|
| ACM MM 2024 | [Improving Composed Image Retrieval via Contrastive Learning with Scaling Positives and Negatives](https://arxiv.org/abs/2404.11317) |  -|



#### 3.2 Consistency constraint
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| WACV 2021 | [Compositional learning of image-text query for image retrieval](https://openaccess.thecvf.com/content/WACV2021/papers/Anwaar_Compositional_Learning_of_Image-Text_Query_for_Image_Retrieval_WACV_2021_paper.pdf) |   [Code](https://github.com/ecom-research/ComposeAE)
| WACV 2022 | [SAC: Semantic attention composition for text-conditioned image retrieval](https://openaccess.thecvf.com/content/WACV2022/papers/Jandial_SAC_Semantic_Attention_Composition_for_Text-Conditioned_Image_Retrieval_WACV_2022_paper.pdf) |   -|
| AAAI 2021 | [Trace: Transform aggregate and compose visiolinguistic representations for image search with text feedback](https://www.researchgate.net/profile/Mausoom-Sarkar/publication/344083983_TRACE_Transform_Aggregate_and_Compose_Visiolinguistic_Representations_for_Image_Search_with_Text_Feedback/links/5fea20b2299bf14088562c70/TRACE-Transform-Aggregate-and-Compose-Visiolinguistic-Representations-for-Image-Search-with-Text-Feedback.pdf) |   -|
| IEEE TIP 2023 | [Composed Image Retrieval via Cross Relation Network With Hierarchical Aggregation Transformer](https://ieeexplore.ieee.org/document/10205526) |   [Code](https://github.com/yan9qu/crn)|
| SIGIR 2021 | [Comprehensive linguistic-visual composition network for image retrieval](https://haokunwen.github.io/files/acmsigir2021.pdf) |   -|
| MM 2021 | [Heterogeneous Feature Fusion and Cross-modal Alignment for Composed Image Retrieval](https://dl.acm.org/doi/10.1145/3474085.3475659) |   -|
| IEEE TMM 2022 | [Heterogeneous Feature Alignment and Fusion in Cross-Modal Augmented Space for Composed Image Retrieval](https://ieeexplore.ieee.org/document/9899752) |   -|
| ACM TOMM 2023 | [Amc: Adaptive multi-expert collaborative network for text-guided image retrieval](https://dl.acm.org/doi/abs/10.1145/3584703) |   [Code](https://github.com/KevinLight831/AMC)|
| Arxiv 2023 | [Decompose Semantic Shifts for Composed Image Retrieval](https://arxiv.org/abs/2309.09531) |   -|
| ACM MM 2023 | [Target-Guided Composed Image Retrieval](https://Arxiv.org/pdf/2309.01366) |   -|
| KBS 2024 | [Collaborative group: Composed image retrieval via consensus learning from noisy annotations](https://arxiv.org/abs/2306.02092) |   -|




#### 3.3 Multi-task joint loss
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ECCV 2022 | [Fashionvil: Fashion-focused vision-and-language representation learning](https://Arxiv.org/pdf/2207.08150) |   [Code](https://github.com/BrandonHanx/mmf)|
| CVPR 2023 | [FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_FAME-ViL_Multi-Tasking_Vision-Language_Model_for_Heterogeneous_Fashion_Tasks_CVPR_2023_paper.pdf) |   [Code](https://github.com/BrandonHanx/FAME-ViL)|
| ICASSP 2021 | [Multi-Order Adversarial Representation Learning for Composed Query Image Retrieval](https://ieeexplore.ieee.org/document/9414436) |   -|
| ACL 2024 | [VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval](https://aclanthology.org/2024.acl-long.175/) |  [Code](https://github.com/JUNJIE99/VISTA_Evaluation_FineTuning)|





## Zero-Shot Learning-based CMR (ZSL-CMR)


Recently, composed multi-modal retrieval based on zero-shot learning has been proposed to address the above limitations. During the training process, the model is trained solely on easily obtainable image-text pairs, without the need for annotated triplets. Its training process usually revolves around learning the modality transformers, which simulate the combination of visual and linguistic information in testing. As a result, the training and testing phases typically involve different network structures. Thus, the main challenge of ZSL-CMR lies in designing transformation frameworks that are equivalent to supervised learning in the absence of supervision signals, aiming to maximize the zero-shot generalization ability as much as possible. To address this challenge, the academic community has developed strategies across three key aspects: (1) Image-side transformation: this approach focuses on learning the implicit or explicit visual-to-linguistic transformation using images as input. During testing, it converts the reference image into a query that can be integrated with the relative textual information; (2) Text-side transformation: in this approach, text is used as input to simulate image features, constructing a training framework that relies solely on language. During testing, the model directly takes image inputs; (3) External knowledge assistance: this approach explores the utilization of additional knowledge to enhance details, such as attributes and colors, thereby improving retrieval performance. 



### 1. Image-side Transformation Approaches
#### 1.1 Explicit visual Transformation
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ICLR 2024 | [Vision-by-language for training-free compositional image retrieval](https://Arxiv.org/pdf/2310.09291) |   [Code](https://github.com/ExplainableML/Vision_by_Language)|
| Arxiv 2024 | [Training-free zero-shot composed image retrieval with local concept reranking](https://Arxiv.org/pdf/2312.08924) |   -|
| SIGIR 2024 | [LDRE: LLM-based Divergent Reasoning and Ensemble for Zero-Shot Composed Image Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657740) |   [Code](https://github.com/yzy-bupt/LDRE)|




#### 1.2 Implicit visual Transformation
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2023 | [Pic2word: Mapping pictures to words for zero-shot composed image retrieval](https://openaccess.thecvf.com/content/CVPR2023/papers/Saito_Pic2Word_Mapping_Pictures_to_Words_for_Zero-Shot_Composed_Image_Retrieval_CVPR_2023_paper.pdf) |   [Code](https://github.com/google-research/composed_image_retrieval)|
| ICCV 2023 | [Zero-shot composed image retrieval with textual inversion](https://openaccess.thecvf.com/content/ICCV2023/papers/Baldrati_Zero-Shot_Composed_Image_Retrieval_with_Textual_Inversion_ICCV_2023_paper.pdf) |   [Code](https://github.com/miccunifi/SEARLE)|
| Arxiv 2024 | [iSEARLE: Improving Textual Inversion for Zero-Shot Composed Image Retrieval](https://arxiv.org/abs/2405.02951) |   [Code](https://github.com/miccunifi/SEARLE)|
| AAAI 2024 | [Context-I2W: Mapping Images to Context-dependent Words for Accurate Zero-Shot Composed Image Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/28324) |   -|
| SIGIR 2024 | [Fine-grained Textual Inversion Network for Zero-Shot Composed Image Retrieval](https://dl.acm.org/doi/10.1145/3626772.3657831) |   [Code](https://github.com/ZiChao111/FTI4CIR)|
| ICLR 2024 | [Image2Sentence based Asymmetrical Zero-shot Composed Image Retrieval](https://Arxiv.org/pdf/2403.01431) |   -|
| CVPR 2024 | [Knowledge-enhanced dual-stream zero-shot composed image retrieval](https://openaccess.thecvf.com/content/CVPR2024/papers/Suo_Knowledge-Enhanced_Dual-stream_Zero-shot_Composed_Image_Retrieval_CVPR_2024_paper.pdf) |   [Code](https://github.com/suoych/KEDs.)|
| CVPR 2025 | [Missing Target-Relevant Information Prediction with World Model for Accurate Zero-Shot Composed Image Retrieval](https://arxiv.org/abs/2503.17109#:~:text=In%20this%20paper%2C%20we%20propose%20a%20novel%20prediction-based,the%20latent%20space%20before%20mapping%20for%20accurate%20ZS-CIR.) |   -|
| CVPR 2025 | [From Mapping to Composing: A Two-Stage Framework for Zero-shot Composed Image Retrieval](https://arxiv.org/abs/2504.17990) |   -|
| CVPR 2025 | [Data-Efficient Generalization for Zero-shot Composed Image Retrieval](https://arxiv.org/abs/2503.05204) |   -|
| CVPR 2024 | [Denoise-I2W: Mapping Images to Denoising Words for Accurate Zero-Shot Composed Image Retrieval](https://arxiv.org/abs/2410.17393#:~:text=In%20this%20paper%2C%20we%20propose%20a%20novel%20denoising,that%2C%20without%20intention-irrelevant%20visual%20information%2C%20enhance%20accurate%20ZS-CIR.) |   -|






 ### 2. Text-side Transformation Approaches
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2024 | [Language-only Efficient Training of Zero-shot Composed Image Retrieval](https://Arxiv.org/abs/2312.01998) |   [Code](https://github.com/navervision/lincir)|
| Arxiv 2024 | [Reducing Task Discrepancy of Text Encoders for Zero-Shot Composed Image Retrieval](https://arxiv.org/abs/2406.09188) |   -|
| CVPR 2024 | [Composed Video Retrieval via Enriched Context and Discriminative Embeddings](https://arxiv.org/abs/2403.16997) |   [Code](https://github.com/OmkarThawakar/composed-video-retrieval)|
| ICML 2024 | [Improving Context Understanding in Multimodal Large Language Models via Multimodal Composition Learning](https://proceedings.mlr.press/v235/li24s.html) |   [Code](https://github.com/dhg-wei/MCL)|
| CVPR 2024 | [MoTaDual: Modality-Task Dual Alignment for Enhanced Zero-shot Composed Image Retrieval](https://arxiv.org/abs/2410.23736) |   -|




 ### 3. External Knowledge Assistance Approaches
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| Arxiv 2023 | [Pretrain like you inference: Masked tuning improves zero-shot composed image retrieval](https://Arxiv.org/pdf/2311.07622) |   -|
| BMVC 2023 | [Zero-shot composed text-image retrieval](https://Arxiv.org/pdf/2306.07272) |   [Code](https://github.com/Code-kunkun/ZS-CIR)|
| Arxiv 2024 | [Pseudo Triplet Guided Few-shot Composed Image Retrieval](https://arxiv.org/abs/2407.06001) |  -|
| WACV 2024 | [Bi-directional Training for Composed Image Retrieval via Text Prompt Learning](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Bi-Directional_Training_for_Composed_Image_Retrieval_via_Text_Prompt_Learning_WACV_2024_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/Bi-Blip4CIR)|



## Semi-Supervised Learning-based CMR (SSL-CMR)
Although zero-shot combined multimodal retrieval does not rely on labeled data, its performance is often lower than supervised training, which brings obstacles to the application of the model. To alleviate this problem, some works have proposed a semi-supervised combined multimodal learning paradigm based on automatically generated triple data. In this setting, relying on the relatively easy-to-obtain image-text data, existing SSL-CMR work mainly generates triplet data from two aspects: (1) generating images, such as editing the input reference image according to the conditional text to create the target image; (2) generating text, such as describing the difference caption between the two input images. By generating these triplet data, the model can not only capture the combination of vision and language more accurately during learning, but also avoid the limitations of cumbersome annotation. Although the generated data may be noisy, it also combines the advantages of supervision and zero-shot learning, which is a promising direction.


### 1. Automatic Data Construction
#### 1.1 Generating Language Content
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| BMVC 2023 | [Zero-shot composed text-image retrieval](https://Arxiv.org/pdf/2306.07272) |   [Code](https://github.com/Code-kunkun/ZS-CIR)|
| Arxiv 2024 | [Pseudo Triplet Guided Few-shot Composed Image Retrieval](https://arxiv.org/abs/2407.06001) |  -|
| IEEE TIP 2024 | [Multimodal Composition Example Mining for Composed Query Image Retrieval](https://ieeexplore.ieee.org/document/10418785) |   -|
| WACV 2024 | [Bi-directional Training for Composed Image Retrieval via Text Prompt Learning](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Bi-Directional_Training_for_Composed_Image_Retrieval_via_Text_Prompt_Learning_WACV_2024_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/Bi-Blip4CIR)|
| Arxiv 2024 | [HyCIR: Boosting Zero-Shot Composed Image Retrieval with Synthetic Labels](https://arxiv.org/abs/2407.05795) |   -|
| CVPR 2024 | [Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval](https://openaccess.thecvf.com/content/CVPR2024/papers/Jang_Visual_Delta_Generator_with_Large_Multi-modal_Models_for_Semi-supervised_Composed_CVPR_2024_paper.pdf) |   -|
| AAAI 2024 | [Data Roaming and Quality Assessment for Composed Image Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/28081) |   [Code](https://github.com/levymsn/LaSCo)|
| CVPR 2025 | [CoLLM: A Large Language Model for Composed Image Retrieval](https://arxiv.org/abs/2503.19910) |   -|
| CVPR 2025 | [Scaling Prompt Instructed Zero Shot Composed Image Retrieval with Image-Only Data](https://arxiv.org/abs/2504.00812) |   -|
| CVPR 2025 | [SCOT: Self-Supervised Contrastive Pretraining For Zero-Shot Compositional Retrieval](https://arxiv.org/abs/2501.08347#:~:text=In%20this%20work%2C%20we%20propose%20SCOT%20%28Self-supervised%20COmpositional,models%20to%20contrastively%20train%20an%20embedding%20composition%20network.) |   -|
| CVPR 2025 | [ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](https://arxiv.org/abs/2505.20764) |   -|
| CVPR 2025 | [Scale Up Composed Image Retrieval Learning via Modification Text Generation](https://arxiv.org/abs/2504.05316) |   -|
| CVPR 2025 | [good4cir: Generating Detailed Synthetic Captions for Composed Image Retrieval](https://arxiv.org/abs/2503.17871#:~:text=We%20introduce%20good4cir%2C%20a%20structured%20pipeline%20leveraging,vision-language%20models%20to%20generate%20high-quality%20synthetic%20annotations.) |   -|


#### 1.2 Generating Language and Visual Content
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ACM TOMM 2022 | [Tell, Imagine, and Search: End-to-end Learning for Composing Text and Image to Image Retrieval](https://dl.acm.org/doi/10.1145/3478642) |   -|
| Arxiv 2023 | [Compodiff: Versatile composed image retrieval with latent diffusion](https://Arxiv.org/pdf/2303.11916) |   [Code](https://github.com/navervision/CompoDiff)|
| ACL 2024 | [VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval](https://aclanthology.org/2024.acl-long.175/) |  [Code](https://github.com/JUNJIE99/VISTA_Evaluation_FineTuning)|
| CVPR 2025 | [Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval](https://arxiv.org/abs/2311.16515) |   -|
| CVPR 2025 | [Generative Zero-Shot Composed Image Retrieval](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Generative_Zero-Shot_Composed_Image_Retrieval_CVPR_2025_paper.html) |   -|
| CVPR 2025 | [Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy](https://arxiv.org/abs/2411.16752) |   -|
| CVPR 2025 | [Multimodal Reasoning Agent for Zero-Shot Composed Image Retrieval](https://arxiv.org/abs/2505.19952) |   -|


### 2. Noise/Uncertainty in Data
#### 2.1 Content Noise
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| BMVC 2023 | [Zero-shot composed text-image retrieval](https://Arxiv.org/pdf/2306.07272) |   [Code](https://github.com/Code-kunkun/ZS-CIR)|
| WACV 2024 | [Bi-directional Training for Composed Image Retrieval via Text Prompt Learning](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Bi-Directional_Training_for_Composed_Image_Retrieval_via_Text_Prompt_Learning_WACV_2024_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/Bi-Blip4CIR)|
| Arxiv 2023 | [Compodiff: Versatile composed image retrieval with latent diffusion](https://Arxiv.org/pdf/2303.11916) |   [Code](https://github.com/navervision/CompoDiff)|
| AI 2023 | [CLIP-based Composed Image Retrieval with Comprehensive Fusion and Data Augmentation](https://dl.acm.org/doi/10.1007/978-981-99-8388-9_16) |   -|
| Arxiv 2024 | [Pseudo Triplet Guided Few-shot Composed Image Retrieval](https://arxiv.org/abs/2407.06001) |  -|
| AAAI 2024 | [Data Roaming and Quality Assessment for Composed Image Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/28081) |   [Code](https://github.com/levymsn/LaSCo)|
| Arxiv 2024 | [Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval](https://arxiv.org/abs/2311.16515) |   [Code](https://github.com/Delong-liu-bupt/Composed_Person_Retrieval)|



#### 2.2 Annotation Uncertainty
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ACM MM 2023 | [Target-Guided Composed Image Retrieval](https://Arxiv.org/pdf/2309.01366) |   -|
| KBS 2024 | [Collaborative group: Composed image retrieval via consensus learning from noisy annotations](https://arxiv.org/abs/2306.02092) |   -|
| CVPR 2023 | [Ranking-aware Uncertainty for Text-guided Image Retrieval](https://arxiv.org/abs/2308.08131) |   -|
| CVPR 2022 | [Composed Image Retrieval with Text Feedback via Multi-grained Uncertainty Regularization](https://arxiv.org/abs/2211.07394) |   -|
| ICCV 2023 | [Zero-shot composed image retrieval with textual inversion](https://openaccess.thecvf.com/content/ICCV2023/papers/Baldrati_Zero-Shot_Composed_Image_Retrieval_with_Textual_Inversion_ICCV_2023_paper.pdf) |   [Code](https://github.com/miccunifi/SEARLE)|
| Arxiv 2022 | [Training and challenging models for text-guided fashion image retrieval](https://Arxiv.org/pdf/2204.11004) |   [Code](https://github.com/yahoo/maaf)|
| CVPR 2025 | [Composed Image Retrieval with Text Feedback via Multi-grained Uncertainty Regularization](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Learning_with_Noisy_Triplet_Correspondence_for_Composed_Image_Retrieval_CVPR_2025_paper.html) |   -|















## Evaluation Metrics
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| NeurIPS 2023 | [Cola: A Benchmark for Compositional Text-to-image Retrieval](https://arxiv.org/abs/2305.03689) |   [Code](https://cs-people.bu.edu/array/research/cola/)|



## Application 

<!--Research in CMR has vast application potential. Despite being a relatively young field, the current focus of this field primarily includes: (1) Composed image retrieval: it has pioneered the combination of image and textual descriptions for retrieval purposes. It enables various tasks based on different textual conditions, such as domain transformation, where images from different stylistic domains can be retrieved; composition of objects/scenes, allowing the addition or modification of objects or scenes during retrieval; and object/attribute manipulation, providing control over the objects or attributes in the retrieval process. These operations have high practical value in domains such as fashion and e-commerce. (2) Composed video retrieval: the user performs such multi-modal search, by querying an image of a particular visual concept and a modification text, to find videos that exhibit similar visual characteristics with the desired modification. This task has many use cases, including but not limited to searching online videos for reviews of a specific product, how-to videos of a tool for specific usages, live events in specific locations, and sports matches of specific players. (3)  Composed person retrieval: person retrieval aims to identify target person images from a large-scale person gallery.  Distinct from existing image-based and text-based person retrieval approaches, in real-world scenarios, both visual and textual information about a specific person is often available. Therefore, the task of jointly utilizing image and text information for target person retrieval facilitates person matching, which has extensive applications in social services and public security.-->

Research in CMR has vast application potential. 
It can be broadly categorized based on application scenarios and image domain differences, including domains as fashion and e-commerce images, natural images, videos, remote sensing images, person images, sketch images, and interactive conversation. The specific application can be personalized product shopping, media search, event discovery, environmental monitoring, law enforcement, customer service bots, and so on. In summary, CMR represents a paradigm shift in search systems by integrating visual and textual modalities. These systems enable fine-grained, context-aware, and user-centric searches across diverse domains, offering significant improvements in both retrieval accuracy and user satisfaction.

![figure2](https://github.com/kkzhang95/Awesome-Composed-Multi-modal-Retrieval/blob/main/images/figure2.png)

 ### (1) Composed Fashion Image Retrieval and Composed Natural Image Retrieval

Traditional fashion image retrieval primarily relies on simple image search or keyword-based search. However, these methods often fail to meet user needs when searching for specific fashion items with complex attributes such as color, style, and material. The main goal of Composed Fashion Image Retrieval (CFIR) is to achieve more accurate and personalized fashion searches by combining both images and texts. 
CFIR has broad applications in e-commerce. By combining image-based search with textual refinement, it improves search efficiency and helps users locate products that better match their preferences. For example, users can input a reference image (e.g., a photo of a clothing item) along with a textual modification (e.g., "long-sleeve version").
This leads to higher customer satisfaction and greater user engagement. For retailers and brands, CFIR supports personalized product recommendations, fashion trend analysis, and more effective inventory management.

| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| FashionIQ | 2021 | [Fashion iq: A new dataset towards retrieving images by natural language feedback](https://arxiv.org/abs/1905.12794) |   [Dataset](https://github.com/XiaoxiaoGuo/fashion-iq)|
| Fashion200k | 2017 | [Automatic attribute discovery and characterization from noisy web data](https://link.springer.com/chapter/10.1007/978-3-642-15549-9_484) |   [Dataset](https://github.com/xthan/fashion-200k)|
| Shoes | 2018 | [Dialog-based Interactive Image Retrieval](https://arxiv.org/abs/1805.00145) |   [Dataset](https://github.com/XiaoxiaoGuo/fashion-retrieval)|



 ### (2) Composed Natural Image Retrieval

Traditional image retrieval systems typically rely on either visual or textual inputs alone. However, single-modality approaches are limited in their ability to represent complex queries involving multiple attributes or concepts. Composed Natural Image Retrieval (CNIR) addresses this limitation by integrating both textual descriptions and image content, enabling systems to better understand abstract and nuanced user requirements. For example, users can upload an image of a favorite landscape along with a description such as “same location but in Autumn,” and the CNIR system will analyze both modalities to retrieve semantically relevant images. By enabling more accurate and personalized search experiences, CNIR significantly improves efficiency and user satisfaction. It also opens new opportunities for innovative services and the broader development of advanced image retrieval technologies.
 
| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CSS | 2018 | [Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vo_Composing_Text_and_Image_for_Image_Retrieval_-_an_Empirical_CVPR_2019_paper.pdf) |   [Dataset](https://github.com/google/tirg)|
| CIRR | 2021 | [Image retrieval on real-life images with pre-trained vision-and-language models](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.pdf) |   [Dataset](https://github.com/Cuberick-Orion/CIRPLANT)|
| CIRCO | 2023 | [Zero-shot composed image retrieval with textual inversion](https://arxiv.org/abs/2303.15247) |   [Dataset](https://github.com/miccunifi/SEARLE)|
| MIT-States | 2015 | [Discovering states and transformations in image collections](https://ieeexplore.ieee.org/document/7298744) |   -|
| Birds-to-Words | 2019 | [Neural Naturalist: Generating Fine-Grained Image Comparisons](https://arxiv.org/abs/1909.04101) |   -|
| LaSCo | 2024 | [Data Roaming and Quality Assessment for Composed Image Retrieval](https://arxiv.org/abs/2303.09429) |   [Dataset](https://github.com/levymsn/LaSCo)|
| Laion-CIR-Combined | 2024 | [Zero-shot Composed Text-Image Retrieval](https://Arxiv.org/pdf/2306.07272) |   [Dataset](https://github.com/Code-kunkun/ZS-CIR)|
| SynthTriplets18M | 2024 | [CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion](https://Arxiv.org/pdf/2303.11916) |   [Dataset](https://github.com/navervision/CompoDiff)|
| Good4cir | 2025 | [good4cir: Generating Detailed Synthetic Captions for Composed Image Retrieval](https://arxiv.org/html/2503.17871) |   -|


 ### (3) Composed Video Retrieval

Composed Video Retrieval (CoVR) enables the retrieval of specific videos from large databases by integrating visual queries with textual modification instructions, allowing for more precise and context-aware searches. This approach overcomes the limitations of traditional content-based video retrieval systems, which rely solely on visual features and often fail to capture user intent or nuanced context. The primary objective of CoVR is to improve search accuracy by leveraging multi-modal inputs. CoVR holds strong application potential across multiple domains, including online video platforms, live event discovery, and sports video retrieval.  On video platforms, it supports advanced content recommendation and management systems by identifying and suggesting videos that better align with user preferences and interests.

 
| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| WebVid-CoVR | 2024 | [CoVR: Learning Composed Video Retrieval from Web Video Captions](https://arxiv.org/abs/2308.14746) |   [Dataset](https://imagine.enpc.fr/~ventural/covr/)|
| CIRR | 2021 | [Image retrieval on real-life images with pre-trained vision-and-language models](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.pdf) |   [Dataset](https://github.com/Cuberick-Orion/CIRPLANT)|
| FashionIQ | 2021 | [Fashion iq: A new dataset towards retrieving images by natural language feedback](https://arxiv.org/abs/1905.12794) |   [Dataset](https://github.com/XiaoxiaoGuo/fashion-iq)|
| EgoCVR | 2024 | [EgoCVR: An Egocentric Benchmark for Fine-Grained Composed Video Retrieva](https://arxiv.org/abs/2407.16658) |   [Dataset](https://github.com/ExplainableML/EgoCVR/)|
| ICQ | 2024 | [Localizing Events in Videos with Multimodal Queries](https://arxiv.org/abs/2406.10079) |   -|




 ### (4) Composed Remote Sensing Image Retrieval

Composed Remote Sensing Image Retrieval (CRSIR) enables users to perform more precise and expressive searches by combining both visual and textual inputs. Instead of relying on a single modality, users can submit a reference image together with a textual description that specifies desired geographic features, environmental conditions, or temporal information. This multi-modal approach enhances the system’s ability to interpret complex queries, leading to more accurate and context-aware retrieval results.
 
| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| PATTERNCOM | 2024 | [Composed Image Retrieval for Remote Sensing](https://arxiv.org/abs/2405.15587) |   [Dataset](https://github.com/billpsomas/rscir)|
| Airplane, Tennis, and WHIRT | 2024 | [Scene Graph-Aware Hierarchical Fusion Network for Remote Sensing Image Retrieval With Text Feedback](https://ieeexplore.ieee.org/document/10537211/) |   -|





 ### (5) Composed Person Retrieval

Composed Person Retrieval (CPR) represents an innovative approach to identifying specific individuals by leveraging both visual and textual information. Traditional methods, such as Image-based Person Retrieval (IPR) and Text-based Person Retrieval (TPR) , often fall short in effectively utilizing both types of data, leading to a loss in accuracy. CPR aims to address this limitation by simultaneously employing image and text queries to enhance the retrieval process. This dual-modality approach not only increases the descriptive power of the query but also refines the relevance of search results, providing more accurate identification of target individuals. CPR is particularly useful in social services and public security, where precise person identification is crucial.
 
| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| SynCPR, ITCPR | 2024 | [Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieva](https://arxiv.org/abs/2311.16515) |   [Dataset](https://github.com/Delong-liu-bupt/Composed_Person_Retrieval)|



 ### (6) Composed Sketch-based Image Retrieval

Composed Sketch-Text Image Retrieval aims to improve the accuracy and relevance of image retrieval by integrating sketch-based and textual inputs. This approach leverages sketches to capture object shapes and structures, while textual descriptions provide complementary details such as color, material, and texture. By combining coarse structural information with fine-grained attributes, it enables more expressive and flexible querying, especially useful when users lack a specific reference image.
 
| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ShoeV2, ChairV2 | 2016 | [Sketch Me That Shoe](https://ieeexplore.ieee.org/document/7780462) |   -|
| Sketchy | 2016 | [The sketchy database: learning to retrieve badly drawn bunnies](https://dl.acm.org/doi/10.1145/2897824.2925954) |   -|
| FS-COCO | 2022 | [FS-COCO: Towards Understanding of Freehand Sketches of Common Objects in Context](https://arxiv.org/abs/2203.02113) |   -|
| SketchyCOCO | 2020 | [SketchyCOCO: Image Generation From Freehand Scene Sketches](https://ieeexplore.ieee.org/document/9157030) |   -|
| ImageNet-R(endition)  | 2021 | [The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization](https://arxiv.org/abs/2006.16241) |   -|



 ### (7) Interactive/Conversational Retrieval

Interactive/Conversational Retrieval (ICR) represents an advanced approach to image retrieval that leverages natural language interactions between users and systems to refine search outcomes progressively. Unlike traditional methods relying solely on images or predefined textual attributes, ICR integrates user feedback through conversational interfaces, enhancing the accuracy and relevance of search results. This method enables users to provide iterative feedback in natural language, refining queries dynamically until they locate the desired image or item. The primary objective of ICR is to facilitate more intuitive, precise, and personalized searches by incorporating both visual and semantic information effectively. 
ICR has significant applications across various domains, including e-commerce, fashion, and social media.
 
| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| Multi-turn FashionIQ | 2021 | [Conversational Fashion Image Retrieval via Multiturn Natural Language Feedback](https://arxiv.org/abs/2106.04128) |   -|


## Reference
If you find this survey helpful, please cite the following paper:

@article{zhang2025composed,
  title={Composed Multi-modal Retrieval: A Survey of Approaches and Applications},
  author={Zhang, Kun and Li, Jingyu and Li, Zhe and Zhang, Jingjing, and et al.},
  journal={arXiv preprint arXiv:2503.01334},
  year={2025}
}


## Copyright Notice
All content in this repository, including but not limited to text, images, and code, is the intellectual property of [Kun Zhang](kkzhang@ustc.edu.cn) and is protected under applicable copyright laws. Since this repository contains content unpublished, any further reproduction, distribution, display, or performance of this content is strictly prohibited without prior written permission from [Kun Zhang](kkzhang@ustc.edu.cn). (Jun 23, 2024)

