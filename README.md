# draw
Reimplementation of the paper "DRAW: A Recurrent Neural Network For Image Generation" in Julia using Knet.jl

* Type "julia mnist-generation.jl --outdir directory_for_image_generation --attention true" to generate MNIST images using attention mechanism.

* Type "julia svhn-generation.jl --outdir directory_for_image_generation --attention true" to generate SVHN images using attention mechanism.

* Type "julia cifar10-generation.jl --outdir directory_for_image_generation --attention true" to generate CIFAR-10 images using attention mechanism.

* Type "julia mnist-generation-with-2-digits.jl --outdir directory_for_image_generation --attention true" to generate MNIST images with 2 digits using attention mechanism.

* Type "julia cluttered-mnist-classification.jl --attention true" to classify MNIST images using attention mechanism.
