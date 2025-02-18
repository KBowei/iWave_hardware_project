# iWave_hardware_project
The inference code for iWave has been re-implemented to facilitate subsequent network quantization and hardware deployment.

The encoding and decoding configuration files are in the cfg directory, with the default configuration file being cfg\encode_yuv.cfg.
The model source code is in the source directory.
Prepare the images in YUV420 format and the pre-trained model files in advance. Note that this repository is specifically designed for the encoding of images in YUV420 format.
In the source/inference directory, the code files related to the model structure are:
learn_wavelet_trans_additive.py (Transformation module, consistent with the structure shown in the presentation)
PixelCNN.py (Entropy model module, consistent with the structure shown in the presentation)
Run command:
Encoding: python \source\inference\encodeAPP.py
