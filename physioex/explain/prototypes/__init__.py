"""Prototype- and concept-based explainability for PhysioEx models.

Submodules (import directly to avoid pulling heavy optional deps eagerly):

- ``local``      -- per-sample prototype relevance via integrated gradients
                    (:class:`PrototypeRelevance`, :func:`get_prototypes`).
- ``reconstruct``-- data- and model-driven reconstructions of learned concepts.
- ``posthoc.nmf``-- NMF-based prototype discovery from embeddings.
- ``posthoc.vq`` -- vector-quantized codebook prototypes
                    (:class:`~physioex.explain.prototypes.posthoc.vq.VQBottleneck`).

Example::

    from physioex.explain.prototypes.posthoc.nmf import discover_prototypes_nmf
"""
