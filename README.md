# This is the repository for the "Self-supervised-Sparse-to-Dense-Motion-Segmentation" paper

The paper addresses the motion segmentation problem via the GRU-based multicut model (Siamese-GRU-Part).
Further, the sparse motion segmentation results are densified via a U-Net based model trained on a self-supervised manner (U-Net-Part).

## Stages
* In the first stage the costs on the edges of a produced graph (for the multicut problem) are assigned based on
a Siamese-GRU model (refer to the folder "Siamese-GRU-Part").
* In the second stage the clustered graph produces the sparse motion segmentation results which are used as noisy labels to train the U-Net model (refer to the folder "U-Net-Part").

## References
````
@InProceedings{Kardoost2020,
 author = {A. Kardoost and K. Ho and P. Ochs and M. Keuper},
 title = {Self-supervised Sparse to Dense Motion Segmentation},
 booktitle = {ACCV},
 year = {2020}
}
````
