"""shoebox shim for the pieces of `basicsr` that CodeFormer's facelib needs.

The real basicsr package imports `torchvision.transforms.functional_tensor`,
removed in torchvision 0.17, so `import basicsr` is broken on any current
torch. Only the functions facelib and the vendored archs actually call live
here; the weight-download directory is pinned to the shoebox project root.
"""
