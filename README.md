# Hubmap + HPA kaggle competition solution
This code repository contains my training and inference code for the kaggle competition "HuBMAP + HPA - Hacking the Human Body".
# Encoder and Decoder
The pipeline is made up of a Segformer encoder and a Daformer decoder. The Segformer is loaded with mit b2 pretrained weights and trained with competition data
# Dataset
The competition dataset has HPA data which is of 3000 X 3000 pixels. Due to computational constraints, I have used tiled images to train the model. Each image is tiled into images of 512 X 512 pixels with a shift of 512 pixels. These tiles are further resized into 256 X 256 pixels before feeding into the model.
# Training
The Segformer encoder is loaded with mit b2 pretrained weights. The decoder is left out with it's initial random weights. The entire model is trained with the tiled competition dataset. I trained the model for 60 epochs, but it never seemed to overfit till then.
#Inference
The inference script uses a shifted window mechanism with large tile size to reduce the edge effect (a term used in competition discussions to refer to the model mislabelling pixels due to lack of context information. For example, when we infer an image using tiles, the pixels in the corners and edges of the tiles will lack context information from their neighbouring pixels which have been sliced off, consequently the model to wrongly label some of the pixels). So to counter this issue, we take tiles of 1024 pixels and only take the labels of 512 pixels in the centre of the tile.


