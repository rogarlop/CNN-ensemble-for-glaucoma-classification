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



#### Architecture
#### Ensembling techniques explored
#### Results
#### Conclusions
