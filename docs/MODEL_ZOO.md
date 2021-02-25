## Model Zoo

**Note:** the models and results here are trained by this repo. In order to accomodate computation efficiency, we use small channel dimensions (64*3) and feature sizes of 128/2 X 128/2. We encourage you to change the channel dimensions and feature sizes to increase the performance.

|   Model   |  Trained on  |   Tested on    |  Recall@1    |  Recall@5    |  Recall@10   | Download Link |
| :--------: | :---------: | :-----------: | :----------: | :----------: | :----------: | :----------: |
| small-model (for one V100 GPU) | Pitts30k-train | Pitts30k-test | 57.9% | 74.3% | 80.6% | [Google Drive](https://drive.google.com/drive/folders/1-MxxW9LVvpq4Cf8jvXWiWh-QIW-DLvNZ?usp=sharing) |
| middle-model (for both efficiency and performance) | Pitts30k-train | Pitts30k-test |       |       |       |       |
| large-model (the proposed one in paper) | Pitts30k-train | Pitts30k-test |       |       |       |       |
