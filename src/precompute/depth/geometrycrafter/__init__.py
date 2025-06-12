"""GeometryCrafter depth estimation module."""

try:
    from .extractor import GeometryCrafterExtractor
    __all__ = ['GeometryCrafterExtractor']
except ImportError as e:
    # If imports fail (e.g., GeometryCrafter not in submodules), provide a helpful error
    import warnings
    warnings.warn(f"GeometryCrafterExtractor not available: {e}")
    __all__ = []