# Tonge-Image-Classification

AI-powered tongue image analysis for Tri-Dhat classification in traditional Thai medicine

Software: Python 3.6.13 and TensorFlow 2.3.0 with Keras 2.4.0

Dataset is avaiable at DOI: [10.21227/jy12-2c41](https://dx.doi.org/10.21227/56cx-0f96)

Abstract:
Traditional Thai medicine (TTM) is an increasingly popular treatment option. Tongue diagnosis is a highly efficient method for determining overall health, practiced by TTM practitioners. However, the diagnosis naturally varies depending on the practitioner's expertise. In this work, we propose tongue image analysis using raw pixels and artificial intelligence (AI) to support TTM diagnoses. The target classification of Tri-Dhat consists of three classes: Vata, Pitta, and Kapha. We utilize our own organized genuine datasets collected from our university's TTM hospital. Class balance and data augmentation were conducted, and we present analysis approaches and experimental designs. Transfer learning techniques for various pretrained models of Deep Learning were exploited. We used two-tailed paired t-tests and single-factor ANOVA analyzes for performance comparisons. Our work demonstrates that DenseNet121 and Xception models provided the most significant results with cropped image datasets, including DSLR-taken and mobile-taken images. Notably, model ensemble evaluations yielded the highest average predictions, achieving a precision of 0.94, an F1 score of 0.96, accuracy of 0.96, sensitivity of 0.96, and specificity of 0.97, supported by a p-value of 0.0003 from ANOVA analysis. We suggest that our methods could be effectively deployed in real-world scenarios to aid TTM practitioners in their diagnoses.
