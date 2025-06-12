"""Memory-efficient GeometryCrafter wrapper - PLACEHOLDER FOR FUTURE IMPLEMENTATION."""

import logging
from pathlib import Path
from typing import Dict, Optional

from .extractor import GeometryCrafterExtractor

logger = logging.getLogger(__name__)


class MemoryEfficientGeometryCrafterExtractor(GeometryCrafterExtractor):
    """Memory-efficient version for processing long videos.
    
    TODO: Implement memory-efficient processing with:
    - Sliding window approach without loading full video
    - Latent space blending (not depth space)
    - Progressive decoding
    - Smart caching strategies
    
    Current status: Uses base GeometryCrafterExtractor implementation.
    """
    
    def __init__(self, config: Dict):
        """Initialize memory-efficient extractor."""
        super().__init__(config)
        
        logger.warning(
            "MemoryEfficientGeometryCrafterExtractor is not yet implemented. "
            "Using standard GeometryCrafterExtractor for now."
        )
    
    def extract_depth(self, 
                     image_dir: Path, 
                     output_path: Optional[Path] = None) -> Dict:
        """Extract depth maps with memory-efficient processing.
        
        TODO: Implement actual memory-efficient logic.
        Currently falls back to base implementation.
        """
        # For now, use the base implementation
        return super().extract_depth(image_dir, output_path)