## Real-Time Web-Based SAR Image Colorization Using Deep Learning

This online platform translates Synthetic Aperture Radar (SAR) images into optical satellite images using a Generative Adversarial Network (GAN). Users upload SAR images, and a UNet-based generator, guided by a PatchGAN discriminator, produces realistic optical images.

## About

This project is an advanced online platform that transforms Synthetic Aperture Radar (SAR) images into optical satellite images using a Generative Adversarial Network (GAN). SAR images, widely used in remote sensing, provide detailed surface information but lack natural visual clarity. By leveraging deep learning techniques, this system generates realistic optical images from SAR inputs, making it easier for researchers and analysts to interpret satellite data. This approach is particularly useful for regions with persistent cloud cover, where optical satellites struggle to capture clear images. The system enhances remote sensing applications by improving data accessibility and usability.

The core model is based on a UNet-based generator, which synthesizes high-quality optical images, and a PatchGAN discriminator, which ensures their realism. The training process combines adversarial loss to enhance image authenticity and L1 loss for pixel-wise accuracy. The implementation is powered by PyTorch, with a backend developed using Flask or FastAPI for seamless online interaction. Users can upload SAR images via a web interface, process them in real time, and visualize the generated optical images. By automating data preprocessing, augmentation, and normalization, the platform ensures high efficiency, making SAR-to-optical image translation more accessible for remote sensing professionals and researchers.

## Features

<!--List the features of the project as shown below-->

- SAR-to-Optical Image Translation – Converts Synthetic Aperture Radar (SAR) images into realistic optical satellite images using a deep learning model.
- Generative Adversarial Network (GAN) Implementation – Utilizes a UNet-based generator and PatchGAN discriminator for high-quality image synthesis.
- Web-Based Interface – Provides an intuitive online platform where users can upload SAR images and receive translated optical images.
- Automated Preprocessing & Augmentation – Implements data normalization and augmentation techniques to enhance model performance and generalization.
- Real-Time Image Processing & Visualization – Generates and displays optical images instantly, allowing users to compare results with original SAR inputs.

## Requirements

<!--List the requirements of the project as shown below-->

- Operating System: Requires a 64-bit OS (Windows 10 or Ubuntu) for compatibility with deep learning frameworks.
- Programming Language – Python (Recommended version: 3.8 or later)
- Deep Learning Framework – PyTorch for model implementation and training
- Image Processing Libraries – OpenCV, PIL (Pillow), and NumPy for image handling and transformations
- Dataset – Paired SAR and optical satellite images for training and testing
- Additional Dependencies – Matplotlib for visualization, Torchvision for data utilities, and Flask/Django (if deploying as a web-based application)

## System Architecture

<!--Embed the system architecture diagram as shown below-->

![image](https://github.com/user-attachments/assets/0b94c70b-521e-40ed-a4bd-189c6f473b66)

## Output

### Input

![in](https://github.com/user-attachments/assets/8efcf318-4acd-401f-92c5-87f6960f2403)

### Output

![out](https://github.com/user-attachments/assets/5ee66423-c921-4376-9de3-ea843f03d19a)

## Results and Impact

The trained GAN successfully translates SAR images into high-quality optical satellite images, producing visually realistic outputs with enhanced structural details. Quantitative evaluations using metrics like SSIM and PSNR indicate a high similarity between generated and real images, while the model demonstrates strong generalization on unseen SAR data across various terrains. Visualization with Matplotlib confirms the effectiveness of the image translation process. This innovation enhances remote sensing applications by providing clearer optical images in cloud-covered regions, aiding in environmental monitoring, disaster assessment, and land cover classification. It supports better decision-making in defense, agriculture, and urban planning by offering reliable optical data while reducing dependence on costly optical satellite imagery. Additionally, this work opens new possibilities for future research in multi-modal satellite image translation and enhancement.

## Articles published / References

1. A benchmarking protocol for SAR colorization: From regression to deep learning approaches. Kangqing Shen, Gemine Vivone, Xiaoyuan Yang, Simone Lolli, Michael Schmitt 01 Jan, 2024 - Neural Networks (Elsevier BV) - Vol. 169, pp 698-712.
2. A Study on the Improvement of SAR Image Colorization Performance Using CUT and SPatchGAN Discriminator Seung-Min Shin, Han Seo Oh, D. Chung 30 Apr, 2023 Journal of The Korean Society for Aerona Vol. 51, Iss: 4, pp 273-280.
3. Multi-Band and Polarization SAR Images Colorization Fusion . Xinchen Li, Dan Jing, Yachao Li, Liang Guo, Liang Han, Qing Xu, Mengdao Xing, Yihua Hu 18 Aug 2022, - Remote sensing - Vol. 14, Iss: 16, pp 4022-4022.
4. Labeling Dataset Based Colorization of SAR Images Using Cycle GAN Sam Young Lee, D. Chung 01 Oct ,2022 – The Journal of Korean Institute of Elect - Vol. 33, Iss: 10, pp 776-783.
5. A Critical Examination of SAR Colorization Impact on Flood Mapping Accuracy. Nour Aburaed,Mina Al-Saad,M. Sami Zitouni,Mohammed Q. Alkhatib,Saeed Al Mansoori,Hussain Al-Ahmad 07 Jul 2024, pp 8504-8508.
