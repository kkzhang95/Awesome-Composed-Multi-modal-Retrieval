# Awesome-Composed-Multi-modal-Retrieval

This repo is used for recording and tracking recent Composed Multi-modal Retrieval (CMR) works, including Composed Image Retrieval (CIR), Composed Video Retrieval (CVR), Composed Person Retrieval (CPR), etc.  

This repository is still a work in progress. If you find any work missing or have any suggestions, feel free
to [pull requests](https://github.com/kkzhang95/Awesome-Composed-Multi-modal-Retrieval/pulls). 
We will add the missing papers to this repo ASAP. In the following, * indicates that the code is not open source yet 


# What is Composed-Multi-modal-Retrieval (CMR)?

Generally, the evolution of content-based retrieval technology has witnessed the transformation from Unimodal Retrieval (UR) to Cross-modal Retrieval (CR), and then to Composed Multi-modal Retrieval (CMR). Compared with the early-stage unimodal retrieval, which was limited to querying information within the same modality, e.g., image search for images, cross-modal retrieval has achieved remarkable accuracy and widespread application in the present era. This enables the search for semantically relevant content in one modality based on the instance query from another modality, e.g., using text search images or videos, allowing users to make full use of these heterogeneous data. In recent years, composed multi-modal retrieval has emerged as a thriving content-based retrieval technology. Within this technical framework, the system aims to discover images/videos that not only bear resemblance to the given reference image/video but also allow for specific modifications based on the provided textual feedback from the user. In this sense, CMR pioneers an advanced level of interactive and conditional retrieval mechanism, leveraging deep integration of visual and linguistic information. This integration greatly enhances the flexibility and precision of user-expressed search intents, injecting new vitality into domains such as internet search and e-commerce. Consequently, CMR exhibits vast potential and far-reaching impact as the next-generation content-based retrieval engine in real-world application scenarios.


# What are the challenges and existing lines of research in CMR?

A core of CMR is that it requires the synergistic understanding and composing of both input vision and language information as the multi-modal query. The earliest closely related studies of CMR are in the field of attribute-based fashion image retrieval, where the key difference is that the textual feedback in attribute-based fashion image retrieval is limited to a single attribute (e.g., 'mini', 'white', 'red'), while CMR is the natural language with multiple words (e.g., 'showing this animal of the input image facing the camera under sunlight'), which is more flexible yet challenging compared to the predefined set of attribute values.  The pioneering CMR works are proposed, where the input query is specified in the form of an image plus some natural language that describes desired modifications to the input image, and led a series of subsequent approaches. Current research in CMR is primarily focused on two main patterns: (1) supervised learning-based CMR (SL-CMR), which focuses on how to design a better combination mechanism of vision and language through supervised training of annotated data, and (2) zero-shot learning-based CMR (ZSL-CMR), which focuses on how to simulate and build a visual-linguistic multi-modal information combination framework without annotated data.


## Supervised Learning-based CMR (SL-CMR)

For the combined multi-modal retrieval models based on supervised learning, a notable characteristic is the requirement of annotated triplet data $(I^{r}, T^{r}, I^{t})$, where $I^{r}$ denotes the reference query image, $T^{r}$ denotes the relatively modified text, and $I^{t}$ denotes the ground truth target image. For the given inputs of  $I^{r}$ and $T^{r}$, the typical pipeline of an SL-CMR model involves mining what content should be modified in the reference image $I^{r}$ according to the caption $T^{r}$, so as to learn a vison-language compositional embedding that encodes the information required to find the interested target image $T^{r}$. Thus, the main challenges faced by such approaches lie in addressing two issues: "Where to see?", which refers to attending to the content in the reference image that needs change, and "How to change?", which aims to modify the reference image based on the textual information while preserving the remaining information. In recent years, research on SL-CMR has primarily focused on three aspects: (1) data augmentation, focusing on alleviating the difficulty of triplet labeling and the defect of labeling noise; (2) model architecture, focusing on designing a better vision-language combiner via cross-modal feature alignment and fusion strategies, as well as the design of other novel frameworks that can be plugged; (3) loss optimization, focusing on the design of more reasonable feature combination constraints. Although supervised training relying on these carefully labeled data often offers better performance, SL-CMR inherently faces two shortcomings: 1) annotating such triplets is both difficult and labor-intensive, and 2) the supervised approaches trained on the collected limited and specific triplets are also hard for generalization.



### 1. Data Augmentation Approaches

#### 1.1 Automatic dataset construction
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| WACV 2024 | [Bi-directional Training for Composed Image Retrieval via Text Prompt Learning](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Bi-Directional_Training_for_Composed_Image_Retrieval_via_Text_Prompt_Learning_WACV_2024_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/Bi-Blip4CIR)|
| AAAI 2024 | [Data Roaming and Quality Assessment for Composed Image Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/28081) |   [Code](https://github.com/levymsn/LaSCo)|
| Arxiv 2023 \$\dagger\$ | [Compodiff: Versatile composed image retrieval with latent diffusion](https://Arxiv.org/pdf/2303.11916) |   [Code](https://github.com/navervision/CompoDiff)|
| BMVC 2023 \$\dagger\$ | [Zero-shot composed text-image retrieval](https://Arxiv.org/pdf/2306.07272) |   [Code](https://github.com/Code-kunkun/ZS-CIR)|
| CVPR 2024 \$\dagger\$ | [Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval](https://openaccess.thecvf.com/content/CVPR2024/papers/Jang_Visual_Delta_Generator_with_Large_Multi-modal_Models_for_Semi-supervised_Composed_CVPR_2024_paper.pdf) |   [Code]()*|

(Note that some methods classify the methods based on automatically constructed data training as semi-supervised or zero-shot. For convenience, we introduce them all here. They are marked with \$\dagger\$.)

#### 1.2 Uncertainty/Noise in data annotation
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| WACV 2021 | [Compositional learning of image-text query for image retrieval](https://openaccess.thecvf.com/content/WACV2021/papers/Anwaar_Compositional_Learning_of_Image-Text_Query_for_Image_Retrieval_WACV_2021_paper.pdf) |   [Code](https://github.com/ecom-research/ComposeAE)|
| Arxiv 2022 | [Training and challenging models for text-guided fashion image retrieval](https://Arxiv.org/pdf/2204.11004) |   [Code](https://github.com/yahoo/maaf)|
| Arxiv 2023 | [Ranking-aware uncertainty for text-guided image retrieval](https://Arxiv.org/pdf/2308.08131) |   [Code]()*|
| ACM MM 2023 | [Target-Guided Composed Image Retrieval](https://Arxiv.org/pdf/2309.01366) |   [Code]()*|
| ICLR 2024 | [Composed image retrieval with text feedback via multi-grained uncertainty regularization](https://Arxiv.org/pdf/2211.07394) |   [Code](https://github.com/Monoxide-Chen/uncertainty_retrieval)|

### 2. Model Architecture Approaches
#### 2.1 Design of the visual-linguistic modality combiner
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2019 | [Composing text and image for image retrieval-an empirical odyssey](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vo_Composing_Text_and_Image_for_Image_Retrieval_-_an_Empirical_CVPR_2019_paper.pdf) |   [Code]()*|
| CVPR 2020 | [Image search with text feedback by visiolinguistic attention learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Image_Search_With_Text_Feedback_by_Visiolinguistic_Attention_Learning_CVPR_2020_paper.pdf) |   [Code](https://github.com/yanbeic/VAL)|
| Arxiv 2020 | [Modality-agnostic attention fusion for visual search with text feedback](https://Arxiv.org/pdf/2007.00145) |   [Code](https://github.com/yahoo/maaf)|
| AAAI 2021 | [Trace: Transform aggregate and compose visiolinguistic representations for image search with text feedback](https://www.researchgate.net/profile/Mausoom-Sarkar/publication/344083983_TRACE_Transform_Aggregate_and_Compose_Visiolinguistic_Representations_for_Image_Search_with_Text_Feedback/links/5fea20b2299bf14088562c70/TRACE-Transform-Aggregate-and-Compose-Visiolinguistic-Representations-for-Image-Search-with-Text-Feedback.pdf) |   [Code]()*|
| ICCV 2021 | [Image retrieval on real-life images with pre-trained vision-and-language models](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/CIRPLANT)|
| CVPR 2021 | [Cosmo: Content-style modulation for image retrieval with text feedback](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_CoSMo_Content-Style_Modulation_for_Image_Retrieval_With_Text_Feedback_CVPR_2021_paper.pdf) |   [Code](https://github.com/postBG/CosMo.pytorch)|
| SIGIR 2021 | [Comprehensive linguistic-visual composition network for image retrieval](https://haokunwen.github.io/files/acmsigir2021.pdf) |   [Code]()*|
| WACV 2021 | [Compositional learning of image-text query for image retrieval](https://openaccess.thecvf.com/content/WACV2021/papers/Anwaar_Compositional_Learning_of_Image-Text_Query_for_Image_Retrieval_WACV_2021_paper.pdf) |   [Code](https://github.com/ecom-research/ComposeAE)|
| Arxiv 2021 | [Rtic: Residual learning for text and image composition using graph convolutional network](https://Arxiv.org/pdf/2104.03015) |   [Code](https://github.com/nashory/rtic-gcn-pytorch)|
| ICLR 2022 | [Artemis: Attention-based retrieval with text-explicit matching and implicit similarity](https://Arxiv.org/pdf/2203.08101) |   [Code](https://github.com/naver/artemis)|
| ECCV 2022 | [Fashionvil: Fashion-focused vision-and-language representation learning](https://Arxiv.org/pdf/2207.08150) |   [Code](https://github.com/BrandonHanx/mmf)|
| Arxiv 2022 | [Training and challenging models for text-guided fashion image retrieval](https://Arxiv.org/pdf/2204.11004) |   [Code](https://github.com/yahoo/maaf)|
| WACV 2022 | [SAC: Semantic attention composition for text-conditioned image retrieval](https://openaccess.thecvf.com/content/WACV2022/papers/Jandial_SAC_Semantic_Attention_Composition_for_Text-Conditioned_Image_Retrieval_WACV_2022_paper.pdf) |   [Code]()*|
| CVPR-W 2022 | [Conditioned and composed image retrieval combining and partially fine-tuning clip-based features](https://openaccess.thecvf.com/content/CVPR2022W/ODRUM/papers/Baldrati_Conditioned_and_Composed_Image_Retrieval_Combining_and_Partially_Fine-Tuning_CLIP-Based_CVPRW_2022_paper.pdf) |   [Code]()*|
| ACM MM 2023 | [Target-Guided Composed Image Retrieval](https://Arxiv.org/pdf/2309.01366) |   [Code]()*|
| ICLR 2024 | [Sentence-level prompts benefit composed image retrieval](https://Arxiv.org/pdf/2310.05473) |   [Code](https://github.com/chunmeifeng/SPRC)|
| CVPR-W 2023 | [Language Guided Local Infiltration for Interactive Image Retrieval](https://openaccess.thecvf.com/content/CVPR2023W/IMW/papers/Huang_Language_Guided_Local_Infiltration_for_Interactive_Image_Retrieval_CVPRW_2023_paper.pdf) |   [Code]()*|
| ACM TOMM 2023 | [Composed image retrieval using contrastive learning and task-oriented clip-based features](https://Arxiv.org/pdf/2308.11485) |   [Code](https://github.com/ABaldrati/CLIP4Cir)|
| ACM TOMM 2024 | [SPIRIT: Style-guided Patch Interaction for Fashion Image Retrieval with Text Feedback](https://dl.acm.org/doi/abs/10.1145/3640345) |   [Code](https://github.com/PKU-ICST-MIPL/SPIRIT_TOMM2024)|




#### 2.2 Design of plug and play structure
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ECCV 2020 | [Learning joint visual semantic matching embeddings for language-guided retrieval](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670137.pdf) |   [Code]()*|
| AAAI 2021 | [Dual compositional learning in interactive image retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/16271) |   [Code](https://github.com/ozmig77/dcnet)|
| TMLR 2023 | [Candidate set re-ranking for composed image retrieval with dual multi-modal encoder](https://Arxiv.org/pdf/2305.16304) |   [Code](https://github.com/Cuberick-Orion/Candidate-Reranking-CIR)|
| ACM TOMM 2023 | [Amc: Adaptive multi-expert collaborative network for text-guided image retrieval](https://dl.acm.org/doi/abs/10.1145/3584703) |   [Code](https://github.com/KevinLight831/AMC)|
| WACV 2024 | [Bi-directional Training for Composed Image Retrieval via Text Prompt Learning](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Bi-Directional_Training_for_Composed_Image_Retrieval_via_Text_Prompt_Learning_WACV_2024_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/Bi-Blip4CIR)|

#### 2.3 Design of non-combinator architecture
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ICLR 2022 | [Artemis: Attention-based retrieval with text-explicit matching and implicit similarity](https://Arxiv.org/pdf/2203.08101) |   [Code](https://github.com/naver/artemis)|
| CVPR 2023 | [FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_FAME-ViL_Multi-Tasking_Vision-Language_Model_for_Heterogeneous_Fashion_Tasks_CVPR_2023_paper.pdf) |   [Code](https://github.com/BrandonHanx/FAME-ViL)|



### 3. Loss Optimization Approaches
#### 3.1 Based on contrastive learning
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ECCV 2020 | [Learning joint visual semantic matching embeddings for language-guided retrieval](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670137.pdf) |   [Code]()*|
| AAAI 2021 | [Trace: Transform aggregate and compose visiolinguistic representations for image search with text feedback](https://www.researchgate.net/profile/Mausoom-Sarkar/publication/344083983_TRACE_Transform_Aggregate_and_Compose_Visiolinguistic_Representations_for_Image_Search_with_Text_Feedback/links/5fea20b2299bf14088562c70/TRACE-Transform-Aggregate-and-Compose-Visiolinguistic-Representations-for-Image-Search-with-Text-Feedback.pdf) |   [Code]()*|


#### 3.2 Consistency loss constraint
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| SIGIR 2021 | [Comprehensive linguistic-visual composition network for image retrieval](https://haokunwen.github.io/files/acmsigir2021.pdf) |   [Code]()*|
| WACV 2021 | [Compositional learning of image-text query for image retrieval](https://openaccess.thecvf.com/content/WACV2021/papers/Anwaar_Compositional_Learning_of_Image-Text_Query_for_Image_Retrieval_WACV_2021_paper.pdf) |   [Code](https://github.com/ecom-research/ComposeAE)|
| WACV 2024 | [Bi-directional Training for Composed Image Retrieval via Text Prompt Learning](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Bi-Directional_Training_for_Composed_Image_Retrieval_via_Text_Prompt_Learning_WACV_2024_paper.pdf) |   [Code](https://github.com/Cuberick-Orion/Bi-Blip4CIR)|


#### 3.3 Pre-training multi-task joint loss
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2022 | [FashionVLP: Vision Language Transformer for Fashion Retrieval With Feedback](https://openaccess.thecvf.com/content/CVPR2022/papers/Goenka_FashionVLP_Vision_Language_Transformer_for_Fashion_Retrieval_With_Feedback_CVPR_2022_paper.pdf) |   [Code]()*|
| ECCV 2022 | [Fashionvil: Fashion-focused vision-and-language representation learning](https://Arxiv.org/pdf/2207.08150) |   [Code](https://github.com/BrandonHanx/mmf)|
| CVPR 2023 | [FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks](https://openaccess.thecvf.com/content/CVPR2023/papers/Han_FAME-ViL_Multi-Tasking_Vision-Language_Model_for_Heterogeneous_Fashion_Tasks_CVPR_2023_paper.pdf) |   [Code](https://github.com/BrandonHanx/FAME-ViL)|
| ICLR 2024 | [Sentence-level prompts benefit composed image retrieval](https://Arxiv.org/pdf/2310.05473) |   [Code](https://github.com/chunmeifeng/SPRC)|









## Zero-Shot Learning-based CMR (ZSL-CMR)


Recently, composed multi-modal retrieval based on zero-shot learning has been proposed to address the above limitations. During the training process, the model is trained solely on easily obtainable image-text pairs, without the need for annotated triplets. Its training process usually revolves around learning the modality transformers, which simulate the combination of visual and linguistic information in testing. As a result, the training and testing phases typically involve different network structures. Thus, the main challenge of ZSL-CMR lies in designing transformation frameworks that are equivalent to supervised learning in the absence of supervision signals, aiming to maximize the zero-shot generalization ability as much as possible. To address this challenge, the academic community has developed strategies across four key aspects: (1) Image-side transformation: this approach focuses on learning the implicit or explicit visual-to-linguistic transformation using images as input. During testing, it converts the reference image into a query that can be integrated with the relative textual information; (2) Text-side transformation: in this approach, text is used as input to simulate image features, constructing a training framework that relies solely on language. During testing, the model directly takes image inputs; (3) Data generation assistance: this type of approach focuses on generating triplet training data through generation tools, facilitating zero-shot construction from the data side; (4) External knowledge assistance: this approach explores the utilization of additional knowledge to enhance details, such as attributes and colors, thereby improving retrieval performance. Despite these efforts, there is currently no comprehensive review that systematically organizes the existing research, challenges, and applications of this emerging field. 



### 1. Image-side Transformation Approaches
#### 1.1 Explicit visual Embedding (Training-free)
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| ICLR 2024 | [Vision-by-language for training-free compositional image retrieval](https://Arxiv.org/pdf/2310.09291) |   [Code](https://github.com/ExplainableML/Vision_by_Language)|
| Arxiv 2024 | [Training-free zero-shot composed image retrieval with local concept reranking](https://Arxiv.org/pdf/2312.08924) |   [Code]()*|



#### 1.2 Implicit visual Embedding 
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2023 | [Pic2word: Mapping pictures to words for zero-shot composed image retrieval](https://openaccess.thecvf.com/content/CVPR2023/papers/Saito_Pic2Word_Mapping_Pictures_to_Words_for_Zero-Shot_Composed_Image_Retrieval_CVPR_2023_paper.pdf) |   [Code](https://github.com/google-research/composed_image_retrieval)|
| ICCV 2023 | [Zero-shot composed image retrieval with textual inversion](https://openaccess.thecvf.com/content/ICCV2023/papers/Baldrati_Zero-Shot_Composed_Image_Retrieval_with_Textual_Inversion_ICCV_2023_paper.pdf) |   [Code](https://github.com/miccunifi/SEARLE)|
| Arxiv 2023 | [Pretrain like you inference: Masked tuning improves zero-shot composed image retrieval](https://Arxiv.org/pdf/2311.07622) |   [Code]()*|
| ICLR 2024 | [Image2Sentence based Asymmetrical Zero-shot Composed Image Retrieval](https://Arxiv.org/pdf/2403.01431) |   [Code]()*|
| AAAI 2024 | [Context-I2W: Mapping Images to Context-dependent Words for Accurate Zero-Shot Composed Image Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/28324) |   [Code]()*|
| Arxiv 2024 | [Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval](https://Arxiv.org/abs/2405.00571) |   [Code]()*|




 ### 2. Text-side Transformation Approaches
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2024 | [Language-only Efficient Training of Zero-shot Composed Image Retrieval](https://Arxiv.org/abs/2312.01998) |   [Code](https://github.com/navervision/lincir)|





 ### 3. External Knowledge Assistance Approaches
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CVPR 2024 | [Knowledge-enhanced dual-stream zero-shot composed image retrieval](https://openaccess.thecvf.com/content/CVPR2024/papers/Suo_Knowledge-Enhanced_Dual-stream_Zero-shot_Composed_Image_Retrieval_CVPR_2024_paper.pdf) |   [Code](https://github.com/suoych/KEDs.)|


## Datasets
General domain (manual annotation)
| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| CIRR | 2021 | [Image retrieval on real-life images with pre-trained vision-and-language models](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.pdf) |   [Dataset](https://github.com/Cuberick-Orion/CIRPLANT)|
| CIRCO | 2023 | [Zero-shot composed image retrieval with textual inversion](https://arxiv.org/abs/2303.15247) |   [Dataset](https://github.com/miccunifi/SEARLE)|

General domain (machine generation)
| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| LaSCo | 2024 | [Data Roaming and Quality Assessment for Composed Image Retrieval](https://arxiv.org/abs/2303.09429) |   [Dataset](https://github.com/levymsn/LaSCo)|
| Laion-CIR-Combined | 2024 | [Zero-shot Composed Text-Image Retrieval](https://Arxiv.org/pdf/2306.07272) |   [Dataset](https://github.com/Code-kunkun/ZS-CIR)|
| SynthTriplets18M | 2024 | [CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion](https://Arxiv.org/pdf/2303.11916) |   [Dataset](https://github.com/navervision/CompoDiff)|

Specific domain (fashion, shoes, 3D scenes, etc.)
| Dataset | Year |    Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| FashionIQ | 2021 | [Fashion iq: A new dataset towards retrieving images by natural language feedback](https://arxiv.org/abs/1905.12794) |   [Dataset](https://github.com/XiaoxiaoGuo/fashion-iq)|
| CFQ | 2022 | [Training and challenging models for text-guided fashion image retrieval](https://arxiv.org/abs/2204.11004) |   [Dataset](https://github.com/yahoo/maaf)|
| Shoes | 2018 | [Dialog-based Interactive Image Retrieval](https://arxiv.org/abs/1805.00145) |   [Dataset](https://github.com/XiaoxiaoGuo/fashion-retrieval)|
| CSS | 2018 | [Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vo_Composing_Text_and_Image_for_Image_Retrieval_-_an_Empirical_CVPR_2019_paper.pdf) |   [Dataset](https://github.com/google/tirg)|
| Fashion200k | 2017 | [Automatic attribute discovery and characterization from noisy web data](https://link.springer.com/chapter/10.1007/978-3-642-15549-9_484) |   [Dataset](https://github.com/xthan/fashion-200k)|


## Evaluation Metrics


## Application 

Research in CMR has vast application potential. Despite being a relatively young field, the current focus of this field primarily includes: (1) Composed image retrieval: it has pioneered the combination of image and textual descriptions for retrieval purposes. It enables various tasks based on different textual conditions, such as domain transformation, where images from different stylistic domains can be retrieved; composition of objects/scenes, allowing the addition or modification of objects or scenes during retrieval; and object/attribute manipulation, providing control over the objects or attributes in the retrieval process. These operations have high practical value in domains such as fashion and e-commerce. (2) Composed video retrieval: the user performs such multi-modal search, by querying an image of a particular visual concept and a modification text, to find videos that exhibit similar visual characteristics with the desired modification. This task has many use cases, including but not limited to searching online videos for reviews of a specific product, how-to videos of a tool for specific usages, live events in specific locations, and sports matches of specific players. (3)  Composed person retrieval: person retrieval aims to identify target person images from a large-scale person gallery.  Distinct from existing image-based and text-based person retrieval approaches, in real-world scenarios, both visual and textual information about a specific person is often available. Therefore, the task of jointly utilizing image and text information for target person retrieval facilitates person matching, which has extensive applications in social services and public security.


| Publication |  Application |   Paper Title     |   Code/Project                                                 |
|:----:|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| Arxiv 2023 | Visual Queation Answering  | [VQA4CIR: Boosting Composed Image Retrieval with Visual Question Answering](https://Arxiv.org/pdf/2312.12273) |   [Code](https://github.com/chunmeifeng/VQA4CIR)|
| Arxiv 2024 | Composed Persion-RID | [Word for Person: Zero-shot Composed Person Retrieval](https://Arxiv.org/pdf/2311.16515) |   [Code](https://github.com/Delong-liu-bupt/Word4Per)|
| CVPR 2024 | Composed Video Retrieval  | [Composed Video Retrieval via Enriched Context and Discriminative Embeddings](https://Arxiv.org/pdf/2403.16997) |   [Code](https://github.com/OmkarThawakar/composed-video-retrieval)|
| AAAI 2024 | Composed Video Retrieval  | [CoVR: Learning Composed Video Retrieval from Web Video Captions](https://arxiv.org/abs/2308.14746) |   [Code](https://imagine.enpc.fr/~ventural/covr/)|
| NAACL 2024 | Image-Text Retrieval | [ComCLIP: Training-Free Compositional Image and Text Matching](https://Arxiv.org/abs/2211.13854) |   [Code](https://github.com/eric-ai-lab/ComCLIP)|



## Copyright Notice
All content in this repository, including but not limited to text, images, and code, is the intellectual property of [Kun Zhang](kkzhang@ustc.edu.cn) and is protected under applicable copyright laws. Since this repository contains content unpublished, any further reproduction, distribution, display, or performance of this content is strictly prohibited without prior written permission from [Kun Zhang](kkzhang@ustc.edu.cn).

