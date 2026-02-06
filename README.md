# Sim-to-Real LiDAR-Camera Fusion for Robust 3D Object Detection in Adverse Weather

## Abstract

This project investigates **robust 3D object detection** under adverse weather conditions (rain, snow, fog) by fusing LiDAR point clouds with RGB camera images using **Graph Neural Networks (GNNs)**. LiDAR sensors suffer from significant noise and signal degradation in poor weather, leading to missed detections and ghost points. We address this by learning weather-invariant representations through a sim-to-real transfer pipeline: synthetic degraded data is generated in **CARLA**, and the model is validated on real-world winter driving data from the **Canadian Adverse Driving Conditions (CADC)** dataset.

## Methodology

### Multi-Modal Fusion Pipeline

1. **LiDAR Branch:** Raw point clouds are voxelized and processed through a 3D sparse convolutional backbone (OpenPCDet) to extract per-point geometric features.
2. **Camera Branch:** RGB images are passed through a 2D CNN backbone to produce dense image feature maps.
3. **Projection & Fusion:** LiDAR points are projected onto the image plane via calibration matrices. For each point, the corresponding image feature is sampled and concatenated with the geometric feature.
4. **Graph Construction:** A k-nearest-neighbor (k-NN) graph is built over the fused point features in 3D Euclidean space.
5. **GNN Reasoning:** An EdgeConv-based GNN propagates information across the local graph neighborhood, enabling the network to reason about local geometric and appearance context jointly. This makes the model resilient to sporadic noise points introduced by adverse weather.

### Mathematical Formulation — EdgeConv

The core message-passing operation follows the **EdgeConv** formulation from DGCNN (Wang et al., 2019). For each point $i$, the updated feature $x'_i$ is computed as:

$$x'_i = \max_{j \in \mathcal{N}(i)} \Theta \cdot (x_j - x_i \| x_i)$$

where:

- $\mathcal{N}(i)$ is the set of k-nearest neighbors of point $i$ in feature space,
- $x_i, x_j \in \mathbb{R}^F$ are the input point features,
- $(x_j - x_i \| x_i)$ denotes concatenation of the relative difference and the anchor feature,
- $\Theta \in \mathbb{R}^{F' \times 2F}$ is a learnable weight matrix,
- $\max$ denotes channel-wise max aggregation over the neighborhood.

This formulation captures both **local geometric structure** (via $x_j - x_i$) and **global position** (via $x_i$), making it well-suited for distinguishing genuine object surfaces from weather-induced noise.

## Project Structure

```
Robust-Lidar-Camera-Fusion/
├── src/            # Core GCN model and fusion logic
├── data/           # CARLA (synthetic) and CADC (real) datasets
├── scripts/        # Data generation and training scripts
├── configs/        # Hyperparameter and experiment configurations
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x compatible GPU
- [CARLA Simulator](https://carla.org/) 0.9.13+

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/Robust-Lidar-Camera-Fusion.git
cd Robust-Lidar-Camera-Fusion

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install CARLA Python API
pip install carla

# Install remaining dependencies
pip install -r requirements.txt
```

## References

- Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., & Solomon, J. M. (2019). *Dynamic Graph CNN for Learning on Point Clouds.* ACM TOG.
- Team OpenPCDet. *OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds.*
- Pitropov, M. et al. (2021). *Canadian Adverse Driving Conditions Dataset.* IJRR.

## License

This project is released under the MIT License.
