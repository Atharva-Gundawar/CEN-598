The main types of workloads where DL models are used in production are recommender systems, computer vision, and NLP.

Recommender systems: 
- Usually have two main components: an embedding layer and a NN, typically an MLP
- The embedding layer maps high-dimensional sparse data to low-dimensional dense data.
- Standard recommender models are Wide & Deep (W&D), Neural collaborative filtering (NCF), Deep & Cross Network (DCN), Deep Interest Evolution Network (DIEN), and Deep Learning Recommender Model (DLRM)

Computer vision models:
- Multiple convolutional layers, often followed by a normalization function, such as batch normalization, and a nonlinear activation function, such as ReLU
- Standard computer vision models are Inception, ResNet, MobileNet, UNet, Mask-RCNN, SSD, YOLO, DCGAN, and StyleGAN

NLP models:
- There are two main types of NLP models: RNN-based (including LSTM-based and GRU-based) and transformer-based models.
- Standard NLP models are sequence-to-sequence, Transformer-LT, BERT, Deep Speech 2, and Tacotron

Reinforcement Learning:
- Q-learning, policy optimization, and model-based

Newer Techniques:
- _AutoML_, which includes neural architecture search (NAS)

## Recommender Systems Topologies

Recommendation systems are a vital component of many online platforms, providing personalized suggestions to users. There are two primary types of recommendation systems: Content-Based Systems and Collaborative Filtering. Here's how they differ:

### Content-Based Systems

1. **Principle**: These systems recommend items similar to those a user has liked in the past. The similarity is determined based on the features of the items.

2. **Data Used**: They rely on the properties of items (like genre, author, or keywords in a book recommendation system). User's previous actions or preferences towards specific item attributes are also considered.

3. **Advantages**:
   - Personalization: Can offer highly personalized recommendations.
   - No Cold Start for Items: Works well with new items, as recommendations are based on item features.
   - Transparency: Easier to explain why an item was recommended based on its attributes.

4. **Disadvantages**:
   - Limited Scope: Tends to recommend items very similar to what the user already knows.
   - Cold Start for Users: Struggles to make recommendations for new users with no history.
   - Item Feature Dependency: Requires a detailed understanding of each item’s features.

### Collaborative Filtering

1. **Principle**: This method makes recommendations based on the preferences of similar users. It assumes that if users agreed in the past, they will agree in the future.

2. **Data Used**: Primarily relies on past user behavior, like ratings or viewing history, without requiring knowledge of the item itself.

3. **Advantages**:
   - Diversity: Can recommend items that are quite different from what the user has seen before.
   - No Need for Item Data: Doesn't require understanding of the content of the items.
   - Community Wisdom: Leverages the preferences of the user community.

4. **Disadvantages**:
   - Cold Start for Users and Items: Struggles with new users and new items that have little interaction data.
   - Scalability: Can be computationally intensive with large numbers of users and items.
   - Popularity Bias: May favor popular items, overshadowing niche choices.

#### Subcategories of Collaborative Filtering

- **User-Based**: Recommendations are based on the preferences of similar users.
- **Item-Based**: Items similar to those a user likes are recommended, based on similarity in user ratings.

### Conclusion

- **Content-Based** systems focus on the attributes of the items, recommending items similar to what a user has liked before, based on content.
- **Collaborative Filtering** relies on user behavior and preferences, recommending items that similar users have liked, regardless of content.

In practice, many modern systems use a hybrid approach, combining aspects of both to leverage the strengths and mitigate the weaknesses of each.

### Large-scale recommender systems
Breaks the process into two stages 
- Recall (candidate generator) stage selects several items that may be of interest to the user
- Ranking stage scores each item and selects those shown to the user

#### Working of Collaborative filtering:
- A rating matrix R (utility matrix or user-interaction matrix) contains the ratings across various users and items. 
- Collaborative filtering learns a user matrix U and an item matrix V composed of user and item feature vectors such that the squared differences between R and the dense matrix R^ = U(V transpose) is minimized.
- We use ALS and SVD for the factorization

### Neural Recommenders :
- Main ones: Wide and Deep (W&D), Neural collaborative filtering (NCF), Deep Interest Evolution Network (DIEN), and Deep Learning Recommender Model (DLRM).
- Others include [autoencoders](https://arxiv.org/abs/1802.05814) to encode implicit ratings or feedback, [GANs](https://arxiv.org/abs/1705.10513), and deep RL to tackle dynamic changes

#### Wide and Deep
- Uses output from a linear model (referred to as wide) and a deep model
- The probability the recommended app is chosen from the play store given the input vector is:
$$P(v_i=1 | \mathbf{x} ) = \sigma \left( \mathbf{w}^T_{\mathit{wide}} \phi(\mathbf{x}) + f_{\mathit{deep}}\left(\mathbf{W}_{\mathit{deep}}, \mathbf{x} \right) + b \right)$$

#### Neural Collaborative Filtering (NCF)

- **What it is:** NCF combines traditional collaborative filtering methods with neural network architectures to enhance recommendation systems.
- **Pros:** 
	- Improves accuracy over traditional models by capturing non-linear user-item interactions.
	- Offers flexibility in modeling complex user behaviors and preferences.
- **Cons:** 
	- Requires significant computational resources for training and inference.
	- Can be prone to overfitting on sparse or small datasets.
- **Where it is used:** 
	- Widely used in e-commerce and content streaming platforms for personalized recommendations.
	- Implemented in scenarios where detailed user-item interaction data is available.
![](https://deeplearningsystems.ai/figures/ch03-02.png)
#### Deep Interest Evolution Network (DIEN) and Behavior Sequence Transformer (BST)

- **What it is:** DIEN focuses on modeling evolving user interests for recommendation, while BST uses Transformer architecture to capture sequential behavior in user-item interactions.
- **Pros:** 
	- DIEN effectively captures dynamic user interests, improving recommendation relevance.
	- BST leverages the powerful Transformer model to understand sequential and contextual nuances in user behavior.
- **Cons:** 
	- DIEN may struggle with rapidly changing interests or sparse data.
	- BST requires substantial computational power and data for training due to its complex architecture.
- **Where it is used:** 
	- DIEN and BST are primarily used in online advertising and e-commerce for dynamic, context-aware recommendations.
	- Suitable for platforms with rich user interaction data and evolving content catalogs.

#### Deep Learning Recommendation Model (DLRM)

- **What it is:** DLRM is a deep learning model designed for large-scale recommendation tasks, combining categorical and numerical features.
- **Pros:** 
	- Efficiently handles a mix of feature types, which is common in real-world recommendation tasks.
	- Scalable to large datasets and capable of leveraging diverse data sources.
- **Cons:** 
	- Can be complex to implement and tune due to its architecture and feature handling.
	- May require extensive computational resources for optimal performance.
- **Where it is used:** 
	- Utilized in large-scale social media and advertising platforms requiring robust, scalable recommendation systems.
	- Ideal for environments with a variety of user data types and high-dimensional feature spaces.
![](https://deeplearningsystems.ai/figures/ch03-03.png)
#### Graph Neural Networks (GNNs)
    
- **What it is**: 
	- GNNs are neural networks designed to work directly on the graph structure, capturing relationships and interconnections between nodes.
- **Pros**: 
	- Excellently captures relational data; highly effective in tasks like node classification and link prediction.
- **Cons**: 
	- Can be computationally expensive for large graphs; performance can depend heavily on graph structure.
- **Where it's used**: 
	- Used in social network analysis, molecular chemistry, and knowledge graphs.

## Computer Vision Topologies

1. **Widespread Adoption of Computer Vision Topologies in Enterprises**: Computer vision topologies, involving technologies like image/video tagging, facial identification, autonomous navigation, video summarization, medical image analysis, and automatic target recognition using various imaging techniques, are extensively used in enterprise businesses.

2. **Evolution from Feature Engineering to Deep Learning (DL)**: Before DL, significant effort was spent on engineering features like Local Binary Pattern (LBP), Histogram of Oriented Gradients (HOG), and Speeded Up Robust Features (SURF) for tasks like image classification. These features, acting like edge detectors, were then used with classifiers like SVMs, proving more effective than using raw pixels.

3. **Impact of AlexNet in 2012**: The introduction of AlexNet in 2012, winning the ImageNet challenge, marked a pivotal moment in computer vision. This event led to the rapid adoption of DL techniques, significantly reducing image classification errors annually. Despite its historical importance, AlexNet is now overshadowed by newer models.

4. **Hierarchical Feature Detection in CNN Models**: Convolutional Neural Network (CNN) models learn to detect increasingly complex features in each layer. The initial layers often learn features similar to those engineered by researchers and akin to those in the mammal primary visual cortex, primarily acting as edge detectors. However, CNNs differ from the mammal visual cortex by relying more on texture than shape features.

5. **Improving CNNs by Texture Perturbation**: Augmenting training datasets by altering textures in images enhances CNNs' dependency on shape features and improves performance.

6. **Diverse Computer Vision Tasks**: Besides classification, object detection, semantic segmentation, verification, and image generation, computer vision encompasses tasks like action recognition, image denoising, super-resolution, and style transfer.

### Image Classification
Key neural image classifiers include AlexNet, VGG, Inception, ResNet, DenseNet, Xception, MobileNet, ResNeXt, and NAS They use inception, residual, group convolution, and depth-wise separable convolutional layers.

Key models:
- Alex Net: Base vanila model 
- **[VGG](https://arxiv.org/abs/1409.1556)** is a family of topologies similar to AlexNet but with more layers and only uses 3×3 convolution filters.
- **[Inception-v1](https://arxiv.org/abs/1409.4842)**, also known as **GoogleNet**, introduced the inception module, which is composed of multiple filters of different sizes that process the same input![](https://deeplearningsystems.ai/figures/ch03-07.png)
- **Inception-v3** introduces the factorization of an n×n convolutional filter into a 1×n followed by an n×1 filter, as shown in Figure [3.8](https://deeplearningsystems.ai/#ch03/#fig:factorization). This factorization maintains the same receptive field and reduces the number of weights from n^2 to 2n.
- **[ResNet](https://arxiv.org/abs/1512.03385)** is a family of models that popularized layers with skip connections, also known as residual layers.![](https://deeplearningsystems.ai/figures/ch03-10.png)
- **[DenseNet](https://arxiv.org/abs/1608.06993)** connects each layer to every other layer. Each layer's inputs are the concatenation of all feature maps from all the previous layers, which have a large memory footprint.
- **Extreme Inception ([Xception](https://arxiv.org/abs/1610.02357))** combines design principles from VGG, Inception, and ResNet, and introduces depthwise separable convolutions.![](https://deeplearningsystems.ai/figures/ch03-11.png)
- **[MobileNet](https://arxiv.org/abs/1704.04861)**, **[MobileNet-v2](https://arxiv.org/abs/1801.04381)**, and **[MobileNet-v3](https://arxiv.org/abs/1905.02244)** target hardware with limited power, compute, and memory, such as mobile phones. These models use depthwise separable convolution blocks with no pooling layers in between.![](https://deeplearningsystems.ai/figures/ch03-12.png)
- **[ResNeXt](https://arxiv.org/abs/1611.05431)** reintroduced group convolutions (initially used by AlexNet to distribute the model into two GPUs) [[XGD+17](https://deeplearningsystems.ai/#biblio/#xie2017)]. In group convolution, the filters separate into groups, and each group operates on specific channels of the input tensor.
	- it incorporates a more advanced architecture based on the concept of Residual Networks (ResNets) with "cardinality" (a set of transformations) as an additional dimension to the depth and width of the network. It uses blocks where each block is a set of parallel convolutional layers (split-transform-merge strategy).
	![](https://deeplearningsystems.ai/figures/ch03-13.png)
- **NAS** is a family of algorithms that learn both the topology and the weights targeting a particular hardware target, such as NASNet and [EfficientNet](https://arxiv.org/abs/1905.11946) [[TL19](https://deeplearningsystems.ai/#biblio/#tan2019-b)]. EfficientNet was initially used on TPUs, but can be used with other hardware.


### Object Detection:
Traditionally a two-step approach: a region proposal step and a classification step.
Object detectors generate several bounding boxes for a given object, and remove most of them using [non-maximum suppression](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c) (NMS).

#### Non Max Suppression:
Non-maximum suppression (NMS) is a crucial technique used in object detection algorithms to ensure that each object is identified only once. When an object detection model scans an image, it often detects multiple bounding boxes around an object, especially overlapping ones for the same object. NMS helps to clean up these results by selecting the most probable bounding box and eliminating the less probable ones. Here's how it works:

1. **Initial Detection**: The object detection model generates potential bounding boxes around objects in an image, each with an associated confidence score indicating the likelihood of an object being present within that box.

2. **Threshold Setting**: A threshold for the confidence scores is set. Bounding boxes with confidence scores below this threshold are discarded. This step removes weak detections.

3. **Sorting**: The remaining bounding boxes are sorted based on their confidence scores, usually in descending order.

4. **Selecting and Suppressing**:
   - The bounding box with the highest confidence score is selected and kept.
   - Then, the algorithm computes the overlap, typically using the Intersection over Union (IoU) metric, between this selected bounding box and all the other boxes.
   - Bounding boxes that have a significant overlap (above a certain IoU threshold) with the selected box are considered as multiple detections of the same object and thus suppressed (i.e., removed). The exact IoU threshold can vary depending on the specific requirements of the task.

5. **Repeating**: Steps 4 and 5 are repeated for each of the remaining bounding boxes in the sorted list. Each time, the box with the highest score is selected, overlapping boxes are suppressed, and the process continues until all boxes have been processed.

The result of applying NMS is that each detected object in the image is represented by only one bounding box, typically the one with the highest confidence score and least overlap with others. This makes the output of the object detection model much cleaner and more accurate, ensuring that objects are not counted multiple times and reducing clutter in the final output.

Key neural object detectors include Faster-RCNN, YOLO, SSD, RetinaNet, and EfficientDet:
- **[Faster-RCNN](https://arxiv.org/abs/1506.01497)** uses a two-step approach with a region proposal network (RPN) and a classification network
	- The base CNN model extracts feature maps from the image, which are passed to the RPN to generate and refine candidate bounding boxes
	- All the bounding boxes are then reshaped to be the same size and passed to the classifier. The [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) (FPN) improved this topology![](https://deeplearningsystems.ai/figures/ch03-14.png)
- **[YOLO](https://arxiv.org/pdf/1506.02640.pdf)** divides the image into a 7×7 grid. Each grid cell is responsible for 2 bounding boxes.
	- YOLOv2 and [YOLOv3](https://arxiv.org/abs/1804.02767) improves by detecting at three scales, using a deeper CNN topology, and having a class score for each bounding box![](https://deeplearningsystems.ai/figures/ch03-15.png)
- **Single-shot detector [(SSD)](https://arxiv.org/abs/1512.02325)** uses an image classification model, such as VGG or MobileNet, as the base network and appends additional layers to the model
	- Bounding boxes start from predefined anchor boxes. 
	- In each of the appended layers, the model refines or predict the bounding box coordinates, each with a respective score.
- **[RetinaNet](https://arxiv.org/abs/1708.02002)** is the first one-stage detector model that outperforms the two-stage detection approach.
	- The primary reason for previous one-stage detectors trailing in accuracy is the extreme class imbalance (many more background class samples). 
	- RetinaNet uses the _focal loss_ function to mitigate this class imbalance
- **[EfficientDet](https://arxiv.org/abs/1911.09070)** is a scalable family of detectors based on EfficientNet. It uses a pyramid network for multiscale detection
### Segmentation
- **[Mask R-CNN](https://arxiv.org/abs/1703.06870)** extends Faster-RCNN by adding a separate output branch to predict the masks for all the classes.
	- This branch is in parallel to the bounding box predictor branch.
- **[DeepLabv3](https://arxiv.org/abs/1706.05587)** uses _atrous convolution_, also known as dilated convolution, hole algorithm, or up-conv to increase the size of the feature maps by upsampling the weight filter, that is, inserting one or more zeros between each weight in the filters
	- Atrous convolution, combined with Spatial Pyramid Pooling (SPP), is known as Atrous SPP (ASPP)![](https://deeplearningsystems.ai/figures/ch03-16.png)
- **[3D U-Net](https://arxiv.org/abs/1606.06650)** and **[V-Net](https://arxiv.org/abs/1606.04797)** are 3D convolutional networks designed for voxel (3D pixels) segmentation from volumetric data. 
	- These models generally required the immense memory only available on server CPUs for training due to the large activations. 
	- Model parallelism techniques (discussed in Section [5.2](https://deeplearningsystems.ai/#ch05/#ch05.sec2)) can be applied to train on GPUs and accelerators.
- **[Detectron](https://github.com/facebookresearch/detectron2)** is a popular open-source platform developed by Facebook

### Verification

**[Siamese networks](https://dl.acm.org/citation.cfm?id=2987282)** learn a similarity function between two input images, the objective is to simultaneously minimize the distance between the anchor and positive image features and maximize the distance between the anchor and negative image features.

### Image Generation

The main types of algorithms used for image generation include auto-regressive and GAN models, specifically, PixelRNN, PixelCNN, DCGAN, 3D GAN, StackedGAN, StarGAN, SyleGAN, and Pix2pix.

1. **PixelRNN and PixelCNN**:
   - Auto-regressive models.
   - Predict pixels along both axes using recurrent (RNN) and convolutional (CNN) layers.
   - Generate conditional distributions for each RGB image channel at each pixel location.

2. **DCGAN and 3D GAN**:
   - Combine Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs).
   - Generate 3D objects.
   - Sample from a low-dimensional space, passing samples to the generator for high-quality object generation.![](https://deeplearningsystems.ai/figures/ch03-17.png)

3. **Stacked GANs**:
   - Train across multiple stacks of GANs.
   - Result in higher quality image generation.

4. **3D-GAN Generator**:
   - Takes a random vector.
   - Generates a 3D image.

5. **StarGAN and StyleGAN**:
   - Generate photorealistic images, including human faces.
   - Adjust latent factors like freckles, hair color, gender, eyeglasses, and facial shape.

6. **Pix2pix**:
   - Adversarial network.
   - Learns mapping from input to output image.
   - Learns cost function to train this mapping.
   - Capable of generating realistic images from various inputs (e.g., labeled maps, colorizing gray images, filling gaps, removing backgrounds, generating images from sketches).

7. **Other Influential Computer Vision Topologies**:
   - **FaceNet**: Face recognition and verification.
   - **SqueezeNet and ShuffleNet**: Image classification on edge devices.
   - **SENet**: High accuracy image classification.
   - **SRGAN**: Image super-resolution.
   - **SqueezeSegV2**: Road-object segmentation from LiDAR point clouds.
   - **OpenPose**: Pose estimation with some commercial adoption.
   - **Wrnch.AI**: Proprietary model for kinematics detection from 2D video.

## Natural Language Processing:
Traditional NLP systems often use a hidden Markov model (HMM)
- An HMM requires language experts to encode grammatical and semantic rules, parse the data, tag words with the appropriate part-of-speech, and iteratively align inputs and outputs.
A popular benchmark to assess the performance of NLP models is the General Language Understanding Evaluation ([GLUE](https://arxiv.org/abs/1804.07461)) benchmark

### Natural Language Understanding:
Neural NLU models can be RNN-based, CNN-based, and transformer-based. They consist of an encoder that takes the source sentence and a decoder that outputs the target sentence.

The inputs to the NN are known as tokens. While earlier NLU topologies used words as tokens, most newer topologies use learned [subwords](https://arxiv.org/abs/1508.07909). An algorithm segments words constrained to a fixed vocabulary size (the maximum number of subwords). These subwords are often interpretable, and the model can generalize to new words not seen during training using these subwords.

Multi-language NMT involves learning a model used across multiple language pairs.

#### RNN-Based
1. **Sequence-to-sequence ([S2S](https://arxiv.org/abs/1409.3215))**
	- The encoder LSTM units take as input (1) the state of the previous LSTM cell, (2) the output of the previous LSTM cell, and (3) the current token
	- The _thought vector_ is the concatenated state vector and output vector of the last LSTM encoder unit. This thought vector is an encoding of the source sentence
	- Variants of the original S2S topology include models with multiple stacked bidirectional LSTM layers and [bidirectional attention](https://arxiv.org/abs/1611.01603)![](https://deeplearningsystems.ai/figures/ch03-19.png)
2. **Google's Neural Machine Translation ([GNMT](https://arxiv.org/abs/1609.08144))**
	- learns a better thought vector by simultaneously training across multiple languages
	- incorporates an [attention](https://arxiv.org/abs/1508.04025) module to cope with long sentences
	- thought vector should be the same regardless of the source and target language since it captures a meaning

#### CNN-Based
More easily parallelizable and have a higher operational intensity than RNN based, another advantage is they extract features hierarchically and may better capture complex relationships in the data.

CNN models have also been [used](https://arxiv.org/abs/1411.4555) as a preprocessing step to image captioning by extracting relevant features

#### Transformer-Based
Transformer-based models use attention modules without any RNN units. These models are more easily parallelizable than RNNs, can learn longer-term dependencies, and have higher arithmetic intensity.
![](https://deeplearningsystems.ai/figures/ch03-20.png)
A transformer primarily consists of a set of encoder and decoder blocks with the same structure but different weight values and with skip connections. Each encoder block consists of two main layers: a self-attention and a feedforward layer:
- self-attention block helps account for context in the input sentence
- Each decoder block consists of three main layers: a self-attention, an encoder-decoder attention, and a feedforward layer.

1. **Bidirectional Encoder Representations from Transformers ([BERT](https://arxiv.org/abs/1810.04805))**
	- Uses context to learn better embeddings
	- BERT is trained using two self-supervised learning tasks. 
		- In one task, the model predicts a randomly masked-out word based on the context of the words before and after it. 
		- In the second task, the model predicts whether the second sentence follows the first sentence in the original paragraph.
![](https://deeplearningsystems.ai/figures/ta03-01.png)

### Speech Recognition
Task of converting acoustic sound waves into written text. ASR systems and other speech-related systems often transform the acoustic sound waves into a spectrogram or Mel-spectrogram representation. A spectrogram is a 2D frequency-time representation.

1. **Deep Speech 2 (DS2)**
   - Developed by: Baidu
   - Type: Neural ASR (Automatic Speech Recognition)
   - Key Features:
     - First major neural ASR model
     - Uses a spectrogram as input
     - Employs a combination of CNN (Convolutional Neural Networks) and RNN (Recurrent Neural Networks) layers
     - Treats the spectrogram input as an image for processing

2. **Listen, Attend, and Spell (LAS)**
   - Developed by: Google
   - Key Features:
     - Incorporates SpecAugment for data augmentation
     - SpecAugment applies image augmentation techniques to the spectrogram
     - Architecture includes an encoder (pyramid RNN) and a decoder (attention-based RNN)
     - The decoder generates each character based on all previous characters and the entire acoustic sequence

3. **RNN-Transducer (RNN-T)**
   - Key Features:
     - Processes input samples and streams alphabetical character outputs
     - Does not rely on attention mechanisms
     - Google developed a quantized version for mobile devices
     - Real-time performance on devices like Google Pixel
     - Utilized in Gboard app with a minimal memory footprint

4. **Wav2letter++**
   - Developed by: Facebook
   - Type: Open-source neural ASR framework
   - Key Features:
     - Uses the fully convolutional model ConvLM
     - Demonstrated the use of transformers for ASR
     - Focused on efficient and scalable speech recognition solutions

###  Text-to-Speech
Text-to-speech (TTS) is the task of synthesizing speech from text.

A TTS system is typically composed of three stages: (1) a text-analysis model, (2) an acoustic model, and (3) an audio synthesis module known as a vocoder. Traditionally, audio synthesis modules combined short-speech fragments collected from a user to form complete utterances. 

Neural TTS systems are now able to generate human-like speech as measured by the MOS (Mean Opinion Score), a human evaluation of the quality of voice.

1. **WaveNet**
   - Developed by: Google
   - Type: Vocoder autoregressive model, based on PixelCNN
   - Key Features:
     - Predicts audio sample distribution conditioned on previous samples and input linguistic features
     - Input features include phoneme, syllable, and word information
     - Utilizes a stack of dilated causal convolutions for handling long-range temporal dependencies
     - Adopts an -bit integer value timestep to reduce latency and simplify softmax output
   - Challenges: High serving latency due to sequential generation of audio samples

2. **Parallel WaveNet**
   - Developed by: Google
   - Key Features:
     - Employs knowledge distillation to train a feedforward network using WaveNet as the teacher model
     - The feedforward neural network (FFNN) is easily parallelizable
     - Capable of real-time speech sample generation with minimal accuracy loss
     - Used in Google Assistant

3. **Tacotron 2**
   - Developed by: Google
   - Type: Generative end-to-end model
   - Key Features:
     - Trained with audio-text pairs to synthesize speech directly from characters
     - Combines methodologies of WaveNet and Tacotron
     - Uses CNN and LSTM layers for encoding character embeddings into Mel-spectrograms
     - Converts Mel-spectrograms to waveforms using WaveNet model as a vocoder
     - Can adapt to generate speech in different voices using a speaker encoder network

4. **WaveRNN**
   - Developed by: Google
   - Key Features:
     - Uses a dual softmax layer to efficiently predict 16-bit audio samples
     - Each softmax layer predicts 8 bits
     - For real-time inference on mobile CPUs, employs model pruning
     - LPCNet, a variant, combines linear prediction with RNN for higher quality

5. **Deep Voice 3 (DV3)**
   - Developed by: Baidu
   - Type: Generative end-to-end model synthesizer
   - Key Features:
     - Similar to Tacotron 2, but uses a fully convolutional topology for mapping character embeddings to Mel-spectrograms
     - Focuses on improved computational efficiency and reduced training time

6. **ClariNet**
   - Developed by: Baidu
   - Key Features:
     - Extends Deep Voice 3 with a text-to-wave topology
     - Uses a WaveNet distillation approach for efficient processing
### Speech-to-Speech Translation
Google [developed](https://arxiv.org/abs/1811.02050) a data augmentation process to improve the performance of a speech-to-translated-text (ST) system [[JJM+18](https://deeplearningsystems.ai/#biblio/#jia2018)]. Google later developed [Translatotron](https://arxiv.org/abs/1904.06037), an end-to-end direct speech-to-speech translation atttention-based sequence-to-sequence model [[JWB+19](https://deeplearningsystems.ai/#biblio/#jia2019)].

## Reinforcement Learning Algorithms
#### Famous Implementations and applications 
- JPMorgan's internal RL system [LOXM](https://medium.com/@ranko.mosic/reinforcement-learning-based-trading-application-at-jp-morgan-chase-f829b8ec54f2) is used [to train](https://arxiv.org/abs/1811.09549) trading agents [[Mos17](https://deeplearningsystems.ai/#biblio/#mosic2017); [BGJ+18](https://deeplearningsystems.ai/#biblio/#bacoyannis2018)]. 
- Facebook [uses](https://arxiv.org/abs/1811.00260) the open-source ReAgent (formerly called RL Horizon) platform for personalized notifications and recommendations [[GCL+19](https://deeplearningsystems.ai/#biblio/#gauci2019)]. 
- Microsoft acquired the [Bonsai](https://www.bons.ai/) platform, designed to build autonomous industrial systems. 
- Intel developed the [Coach](https://github.com/NervanaSystems/coach) platform, which supports multiple RL algorithms and integrated environments, and is [integrated](https://aws.amazon.com/blogs/machine-learning/custom-deep-reinforcement-learning-and-multi-track-training-for-aws-deepracer-with-amazon-sagemaker-rl-notebook/) into Amazon SageMaker RL. 
- DeepMind built the [TRFL](https://github.com/deepmind/trfl/blob/master/docs/index.md) platform and Google built the [Dopamine](https://github.com/google/dopamine) platform (both on top of TensorFlow), and UC Berkeley released [Ray](https://arxiv.org/abs/1712.05889) with the [RLlib](https://ray.readthedocs.io/en/latest/rllib.html) reinforcement library to accelerate RL research [[CMG+18](https://deeplearningsystems.ai/#biblio/#castro2018); [MNW+18](https://deeplearningsystems.ai/#biblio/#moritz2018)].

#### Simulators include :
- [DeepMind Control Suite](https://arxiv.org/abs/1801.00690) environments
- MuJoCo locomotion environments
- OpenAI Gym
- Bullet, Havoc
- ODE
- [FleX](https://developer.nvidia.com/flex)
- PhysX

RL algorithms often run multiple agents on CPUs; [one per core](https://github.com/NervanaSystems/coach/tree/master/benchmarks/a3c) [[CLN+17](https://deeplearningsystems.ai/#biblio/#caspi2017)]. Recent [work](https://arxiv.org/abs/1803.02811), such as the OpenAI Rapid system, shows that leveraging both CPUs and GPUs can improve performance [[SA19](https://deeplearningsystems.ai/#biblio/#stooke2019)].

![](https://deeplearningsystems.ai/figures/ch03-24.png)

### **Q-learning**
Also known as **value-based**, learns the quality of the agent's state and action. Deep Q-network (DQN) algorithm showed superior performance across various Atari games. Using a variety of Q-learning models [achieves](https://arxiv.org/abs/1710.02298) better performance over any single Q-learning model.

### Policy optimization or Policy Iteration:
The space is explored initially through random actions. Actions that lead to a positive reward are more likely to be retaken.

A primary challenge is the sparse delayed rewards, formally known as the credit assignment problem. Trust Region Policy Optimization ([TRPO](https://arxiv.org/abs/1502.05477)) is typically used over vanilla PG as it guarantees monotonic policy improvements. Actor-Critic using Kronecker-Factored Trust Region (ACKTR).

Various algorithms combine Q-learning and policy optimization methodologies. The most popular ones are Asynchronous Actor-Critic Agents [A3C](https://arxiv.org/abs/1602.01783) and Deep Deterministic Policy Gradients [DDPG](https://arxiv.org/abs/1509.02971). 

### Model-based:
1. Model-based algorithms operate on a predefined set of rules within their environment.
2. Agents utilizing these algorithms predict outcomes of various actions to select the one yielding maximum reward.
3. Notable implementations include DeepMind's AlphaGo, AlphaGo Zero, AlphaZero, and MuZero.
4. One challenge in model-based algorithms is the introduction of biases and errors during learning through trial and error.
5. Model-Based Policy Optimization (MBPO) integrates model learning with policy optimization to reduce error compounding.



