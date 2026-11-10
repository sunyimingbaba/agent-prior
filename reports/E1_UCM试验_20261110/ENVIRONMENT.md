# E1 pilot environment specification
Isolated environment: /public/agse-e1-pilot-20261110/env, system packages inherited from /opt/conda/envs/Pytorch-2.11.0 (Python 3.13, torch 2.10.0 CUDA 13.0).
Purpose: read-only image-text scoring and image-prior inference. Legacy sources, weights, datasets and previous environments are not modified. New outputs remain in this experiment directory.
GPU readiness requires finite seeded CUDA forward/backward and real model inference, with all compatibility adaptations recorded.
Old Python 3.12 torch 2.4 CUDA 12.4 lacks sm_120 kernel on this RTX 5090.
