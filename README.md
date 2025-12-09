# CNN-ensemble-for-glaucoma-classification
This repository contains the files for the development of the project for classification of glaucoma into healthy and normal cases. It uses public databases for training both kind of images used (Color Fundus Photography and Optical Coherence Tomography) and two external databases which contain pairs of images of the same patient.

### Motivation and justification

This project was developed as an intention to proof if a mixture between different image modalities would be valuable for glaucoma detection. Typically, ophtalmologists does not use a single image to diagnose. They perform diverse clinical examinations, including but not limited to intraocular pressure, cornea angle measurement, retinal thickness, visual field assessment and so on. In some clinical centers, there exist specialized medical equipment that can take the two most used imaging techniques for diagnosis: Color Fundus Photographs (CFP) and Optical Coherence Tomography (OCT). These images can be centered in the same structure (the optic nerve head) but CFP views the problem from a top view, in contrast to OCT that reconstructs the depth in retinal layers through optic interferometry. This allows an enriched view of the eye structure viewed from two complementary sides. Our proposal was to develop a CNN model capable of extracting automatically the relevant features to decide if an eye presents signs of glaucoma disease. Often, the structural changes observed in both image types precede the vision loss caused by glaucoma, that can't be reversed.

### Procedure
To achieve this goal, we selected a convolutional neural network approach (CNN) because it allowed us to automatically extract the relevant features for determining the presence or abscence of the ilness. We first trained a module alone for each modality consisting of three CNN. Those CNN used different architectures in order to have diversity and perform in the best case, different mistakes. Then, the CNN predictions of each module were combined to build an ensemble. Once we had an ensemble on each modality, we combined the results from the different modalities to perform the last decision.

All of our models were trained on Google Collaboratory, on a GPU NVIDIA Tesla T4 (16 GB VRAM) with available system RAM of 51 GB and 240 GB in temporary storage disk.

#### Used Databases and Preprocessing
##### CFP module
For the training of our models in the CFP module, we collected and preprocessed the Databases ACRIMA, ORIGA, RIM-ONE-DL, LAG, Sjchoi86-HRF and BEH, publically available. In total, we collected 7721 images that were labelled as glaucomatous or healthy. Then we cropped the region around the optic disc (previously named as optic nerve head) for ensuring the object of interest was located in the whole image rather than in a small portion, as the next figure illustrates.
<p align="center"> <img width="600" height="220" alt="ROI EXTRACTION" src="https://github.com/user-attachments/assets/4e8f7b90-a591-448f-98b7-3f497bfbbe2f" /> </p>

After having all the centered images, we performed a variant of the histogram equalization technique named CLAHE. This techniques allows a better distinction of structures with mantaining low the blurring or noise addition compared with the classical histogram equalization technique. We applied a wiener filter before applying CLAHE (mitigate artificial borders, still present despite the power of the CLAHE technique), and both techniques were applied to the green channel (because this channel in general is more responsive than the other ones in detecting and distinguish structures). Done that process, our models were trained with 70% of data, and tested with validation and test sets. Data augmentation transformations were applied in range to these images for overcoming overfitting and avoiding memorization during training.

##### OCT module

For our OCT images we took the images from a public database available [here](https://zenodo.org/records/1481223). This database has the volume cases (i.e. the whole reconstruction of the volume centered in the optic nerve head). We extracted just the central images, where the visualization of the optic disc can be seen, and extended the volume label to each one of the extracted images. It is to note that this database has only examples for POAG (Primary Open Angle Glaucoma, the most common variant) and healthy aspect eyes. After the extraction of these images, due to their nature and resampling, they had aggresive speckle noise. We removed it by using a Non-Local Means algorithm approach and a high-pass filter to prevent excessive blurring caused by an aggresive noise removal. The next image shows the original and enhanced images. Those images were not cropped around the optic disc since they are already centered on it. This was the only database we were able to get access for this development, which had glaucoma labels. In total we had aroudn 4900 images for our models training.

<p align="center"><img width="359" height="349" alt="image" src="https://github.com/user-attachments/assets/0b8ff603-1d5c-4b6d-a4d6-398de7a871f5" /></p>

#### Architecture
For the CNN training we selected the pretrained ImageNet architectures ResNet50, Inception and DenseNet169 for our CFP module and MobileNetV3, and ResNet50 and Xception for our OCT module. This selection is not trivial. Many of these CNN have performed outstandingly in medical image classification, and some of them are built or designed to be more efficient, accurate or lighter than their predecesors. We took the pretrained CNN and freezed the base, added several fully connected layers, regularization techniques (including dropout, regularization terms) and trained our models. Later, we unfreezed the entire models for performing fine-tuning and trained for more epochs. With this unfreezing a better performance was achieved but you have to consider that for your particular set of images it may vary the best layer to unfreeze from. Also, a risk of overfitting and memorization increases a lot for the amount of available parameters. We also implemented early stopping and decay of learning rate for escaping local minima. We also retained the best weights during the training phase. In the next image you can se some training curves for each modality.

<p align="center"><img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/4ae63162-ead1-4720-9b7c-b7d2c0c981b8" /> <img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/5c02285f-abee-48dd-b746-f4a514f01706" /></p>

#### Ensembling techniques explored
Once we obtained the trained models, we performed their combination first by module (or modality) and then by combining both modules into a final outcome. This is what we call ensemble: The combination of those various models to have a greater performance. We explored techniques such as stacking (and training a SVM as meta-model), Multi layer perceptron (MLP) trained on top of the last fully connected layers of each module (automatically decide which model select) and fusion techniques (Averaging, Max Voting and Weighted Average). The fusion techniques were performed at prediction probabilities, and the stacking and MLP were performed at feature level (that is before the predicted outcome of each model).

Stacking works as a two stage learner. In the first phase, the base (or weak) learners are trained and produce predictions. Then, the predictions are used to train the meta-model (second stage, strong learner) to decide and learn how to combine the previous predictions in order to achieve the best result. This second stage training is costly to data, because data leakage has to be avoided by training in a different subset than the used to train the base models. The following figure explains visually how this ensembling technique works.

<p align="center"> <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/38034c79-01d2-4842-bbda-aa2ba5406eb8" /></p>

MLP is also costly to data. It has to be trained on enough data to avoid overfitting to a specific set of images. It can be understood like a way to mix CNN features abstractly and use all that information to perform the final classification. One drawback is that the complexity is much greater than using simpler methods like Averaging (AV), Max Voting (MV) or weighted averaging voting (AWV). Even with that, if there are enough quantity of data, it could produce the best outcome.

<p align="center"> <img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/6c292110-bd37-4266-b441-dbd1338f4ada" /></p>

Max Voting (or Majority Voting) simply takes all the models predictions and takes the maximum occurrence of the classes as the predicted one for that particular sample. It is simple but effective and popular. It is defined only for odd cases but if desired, it can be used on even ones, defining a custom rule when a tie ocurrs. The math behind that method is presented in the image.

<p align="center"> <img width="341" height="109" alt="image" src="https://github.com/user-attachments/assets/d079ef39-7b9f-401b-88ad-fd6ba8773a4e" /></p>

### Results
### Conclusions
