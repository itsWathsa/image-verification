# TruFor inference source

Copied from https://github.com/grip-unina/TruFor/tree/ae54475df6f41a491d7615100feb19263dec13f7/test_docker/src/models

Original authors: Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, Luisa Verdoliva / GRIP-UNINA, as credited upstream. Preserve LICENSE.txt, LICENSE_CMX.txt, and upstream notices. Informational/nonprofit use only under the upstream license.

Modifications: absolute imports renamed to trufor_vendor.models; timm layer imports replaced by a small inference-only compatibility module. DropPath is an identity in evaluation mode and rejects training. Truncated normal initialization uses PyTorch's own implementation. Inference weights are loaded strictly; model topology and forward calculations are unchanged. No training support is provided.
