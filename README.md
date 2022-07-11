# JPEG Compression Detection and Quality Estimation

A simple demonstration of **JPEG compression detection and quality estimation** using **machine learning**.

## Research Process

## Dataset Creation

## Feature Engineering

The aim is to produce a **tensor** of **fixed-length** representing an image of **arbitrary size**. The custom **feature extraction process** is the following.

Let $W$ and $H$ be the image width and height, respectively. Let $B = 8$ represent a **block size**. The process below is described for a single channel. In the case of multiple channels, e.g., for **YCbCr** color mode, the very same algorithm would be adopted separately.

1. **Image padding**. Assure that both image dimensions $W$ and $H$ are divisible by $B$, thus $\tilde{W} = W + \Delta w$ such that $\exists k \in \mathbb{N}$, such that $\tilde{W} = Bk$. The same applied for $\tilde{H}$. If necessary, expand the image by copying the edge values.
2. **Block splitting**. Split the image of size $\tilde{W} \times \tilde{H}$ into equal $B \times B$ blocks. Let $N$ denote the number of produced blocks.
3. **Block reshaping**. Merge all the $N$ extracted blocks into a single tensor of shape $N \times B \times B$.
3. **Reduction**. Apply $R$ different statistical reductions, each time using a different function, e.g., *min*, *max*, *mean*, *standard deviation*, or *median*. Each reduction will produce $R$ distinct $B \times B$ matrices.
4. **Zig-zag selection and Concatenation**. For subsequent visualization sake, a zig-zag selection is utilized as a substitute for a *flatten* operation (the order of indices does not affect ML algorithms). Thus, each of the $R$ matrices with shape $B \times B$ is converted into a single-dimensional vector of length $B^2$. The resulting vectors are concatenated to form the **final feature vector** of length $R \cdot B^2$.

There are several notable observations. Given the feature extraction strategy above, the contribution of the *minimum* and *maximum* statistics is the least significant. It is completely negligible. As a result, the model trained using just *mean*, *standard deviation*, and *median* performs just as well with considerably fewer parameters.

## References

### Datasets

* [TIFF files dataset](https://people.math.sc.edu/Burkardt/data/tif/tif.html).

### Relevant Research Papers

* Robinson, Jonathan, and Vojislav Kecman. "[Combining support vector machine learning with the discrete cosine transform in image compression](https://pubmed.ncbi.nlm.nih.gov/18238074/)." IEEE Transactions on Neural Networks 14.4 (2003): 950-958.
* Retraint, Florent, and Cathel Zitzmann. "[Quality factor estimation of jpeg images using a statistical model](https://www.sciencedirect.com/science/article/pii/S1051200420301044?casa_token=x3S0erEPH7AAAAAA:ko_yVkwG4rUTbHoo_k8GYBfXEnqMeVDfPq6WVGJXfRXecyvPkNbToFiALAVJka8NKBKYEiLnAw)." Digital Signal Processing 103 (2020): 102759.