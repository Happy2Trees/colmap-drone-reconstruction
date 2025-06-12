"""GeometryCrafter depth extraction wrapper for video sequences."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

import cv2
import numpy as np
import torch
from diffusers.training_utils import set_seed
from tqdm import tqdm
import psutil

# Import ported models from local directory
from .models import (
    GeometryCrafterDiffPipeline,
    GeometryCrafterDetermPipeline,  
    PMapAutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModelVid2vid
)

# Import MoGe from pip installation
try:
    from moge.model.moge_model import MoGeModel
except ImportError:
    # Fallback for different moge package structure
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version('v1')  # Try v1 version

from ..base_depth_estimator import BaseDepthEstimator

logger = logging.getLogger(__name__)


class GeometryCrafterExtractor(BaseDepthEstimator):
    """GeometryCrafter video depth estimation wrapper.
    
    This extractor wraps the original GeometryCrafter implementation for
    video-based depth estimation with temporal consistency.
    """
    
    def __init__(self, config: Dict):
        """Initialize GeometryCrafter depth extractor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Video model specific config
        self.is_video_model = True
        self.model_type = config.get('model_type', 'determ')  # 'determ' or 'diff'
        self.cache_dir = config.get('cache_dir', 'workspace/cache')
        
        # Window parameters for long videos
        self.window_size = config.get('window_size', 110)
        self.overlap = config.get('overlap', 25)
        self.decode_chunk_size = config.get('decode_chunk_size', 8)
        
        # Processing parameters
        self.num_inference_steps = config.get('num_inference_steps', 5)
        self.guidance_scale = config.get('guidance_scale', 1.0)
        self.downsample_ratio = config.get('downsample_ratio', 1.0)
        
        # Model options
        self.force_projection = config.get('force_projection', True)
        self.force_fixed_focal = config.get('force_fixed_focal', True)
        self.use_extract_interp = config.get('use_extract_interp', False)
        self.low_memory_usage = config.get('low_memory_usage', False)
        
        # MoGe options
        self.save_moge_prior = config.get('save_moge_prior', False)
        
        # PLY export options
        self.save_ply = config.get('save_ply', False)
        
        # Set random seed
        self.seed = config.get('seed', 42)
        set_seed(self.seed)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Load and initialize GeometryCrafter models."""
        logger.info("Loading GeometryCrafter models...")
        
        try:
            # Load UNet
            self.unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
                'TencentARC/GeometryCrafter',
                subfolder='unet_diff' if self.model_type == 'diff' else 'unet_determ',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir
            ).requires_grad_(False).to(self.device, dtype=torch.float16)
            
            # Load Point Map VAE
            self.point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
                'TencentARC/GeometryCrafter',
                subfolder='point_map_vae',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                cache_dir=self.cache_dir
            ).requires_grad_(False).to(self.device, dtype=torch.float32)
            
            # Load MoGe prior model from pip installation
            self.prior_model = MoGeModel.from_pretrained(
                'Ruicheng/moge-vitl', 
                cache_dir=self.cache_dir
            ).eval().requires_grad_(False).to(self.device, dtype=torch.float32)
            
            # Create pipeline
            if self.model_type == 'diff':
                self.pipe = GeometryCrafterDiffPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    unet=self.unet,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    cache_dir=self.cache_dir
                ).to(self.device)
            else:
                self.pipe = GeometryCrafterDetermPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    unet=self.unet,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    cache_dir=self.cache_dir
                ).to(self.device)
            
            # Enable optimizations
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("Xformers enabled for memory efficient attention")
            except Exception as e:
                logger.warning(f"Xformers not available: {e}")
            
            self.pipe.enable_attention_slicing()
            
            logger.info("GeometryCrafter models loaded successfully")
            
            # Add forward_image wrapper for MoGeModel compatibility
            def forward_image_wrapper(image: torch.Tensor, **kwargs):
                with torch.no_grad():
                    output = self.prior_model.infer(image, resolution_level=9, apply_mask=False, **kwargs)
                    points = output['points']  # B, H, W, 3
                    masks = output['mask']     # B, H, W
                    return points, masks
            
            self.prior_model.forward_image = forward_image_wrapper
            
        except Exception as e:
            logger.error(f"Failed to load GeometryCrafter models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def extract_depth(self, 
                     image_dir: Path, 
                     output_path: Optional[Path] = None,
                     frame_start: Optional[int] = None,
                     frame_end: Optional[int] = None) -> Dict:
        """Extract depth maps from images in a directory.
        
        For GeometryCrafter, we process the sequence as a video
        to maintain temporal consistency. Can process a specific range of frames.
        
        Args:
            image_dir: Path to directory containing input images
            output_path: Optional path to save depth maps
            frame_start: Starting frame index (inclusive), None for 0
            frame_end: Ending frame index (exclusive), None for all frames
            
        Returns:
            Dictionary containing extraction results
        """
        # Get sorted image paths
        all_image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        if not all_image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        # Apply frame range if specified
        if frame_start is None:
            frame_start = 0
        if frame_end is None:
            frame_end = len(all_image_paths)
        
        # Validate range
        frame_start = max(0, frame_start)
        frame_end = min(len(all_image_paths), frame_end)
        
        if frame_start >= frame_end:
            raise ValueError(f"Invalid frame range: {frame_start} to {frame_end}")
        
        # Get subset of images
        image_paths = all_image_paths[frame_start:frame_end]
        
        logger.info(f"Processing frames {frame_start} to {frame_end-1} ({len(image_paths)} frames)")
        
        # Get output path - always use the same directory for all segments
        output_dir = self.get_output_path(image_dir, output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # If MoGe prior saving is enabled, create subdirectory
        if self.save_moge_prior:
            moge_dir = output_dir.parent / "MoGe"
            moge_dir.mkdir(parents=True, exist_ok=True)
        
        # Load images as video frames
        frames = self._load_images_as_video(image_paths)
        
        # Check dimensions
        height, width = frames.shape[1:3]
        if height % 64 != 0 or width % 64 != 0:
            logger.warning(f"Image dimensions ({height}x{width}) not divisible by 64. "
                         "Resizing to nearest multiple of 64.")
            height = (height // 64) * 64
            width = (width // 64) * 64
            frames_resized = []
            for frame in frames:
                frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
                frames_resized.append(frame_resized)
            frames = np.array(frames_resized)
        
        # Convert to tensor
        device = 'cpu' if self.low_memory_usage else self.device
        frames_tensor = torch.tensor(frames.astype("float32"), device=device).float().permute(0, 3, 1, 2)
        
        # Apply downsampling if needed
        if self.downsample_ratio > 1.0:
            import torch.nn.functional as F
            original_height, original_width = frames_tensor.shape[-2], frames_tensor.shape[-1]
            frames_tensor = F.interpolate(
                frames_tensor, 
                (round(frames_tensor.shape[-2]/self.downsample_ratio), 
                 round(frames_tensor.shape[-1]/self.downsample_ratio)), 
                mode='bicubic', 
                antialias=True
            ).clamp(0, 1)
        
        # Process with GeometryCrafter
        logger.info("Processing video with GeometryCrafter...")
        try:
            with torch.inference_mode():
                rec_point_map, rec_valid_mask = self.pipe(
                    frames_tensor,
                    self.point_map_vae,
                    self.prior_model,
                    height=height if self.downsample_ratio == 1.0 else frames_tensor.shape[-2],
                    width=width if self.downsample_ratio == 1.0 else frames_tensor.shape[-1],
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    window_size=min(self.window_size, len(frames_tensor)),
                    decode_chunk_size=self.decode_chunk_size,
                    overlap=self.overlap if len(frames_tensor) > self.window_size else 0,
                    force_projection=self.force_projection,
                    force_fixed_focal=self.force_fixed_focal,
                    use_extract_interp=self.use_extract_interp,
                    track_time=False,
                    low_memory_usage=self.low_memory_usage
                )
                
                # Upsample back if needed
                if self.downsample_ratio > 1.0:
                    import torch.nn.functional as F
                    rec_point_map = F.interpolate(
                        rec_point_map.permute(0,3,1,2), 
                        (original_height, original_width), 
                        mode='bilinear'
                    ).permute(0, 2, 3, 1)
                    rec_valid_mask = F.interpolate(
                        rec_valid_mask.float().unsqueeze(1), 
                        (original_height, original_width), 
                        mode='bilinear'
                    ).squeeze(1) > 0.5
                    
        except Exception as e:
            logger.error(f"GeometryCrafter inference failed: {e}")
            raise RuntimeError(f"Depth extraction failed: {e}")
        
        # Extract depth from point maps and save
        logger.info("Saving depth maps...")
        depth_maps = rec_point_map[..., 2].detach().cpu().numpy()
        valid_masks = rec_valid_mask.detach().cpu().numpy()
        
        # Extract point maps for PLY if needed before clearing GPU memory
        if self.save_ply:
            point_maps = rec_point_map.detach().cpu().numpy()
        
        # Clear GPU memory immediately after transferring to CPU
        del rec_point_map
        del rec_valid_mask
        del frames_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
        for i, (depth, mask, img_path) in enumerate(tqdm(
            zip(depth_maps, valid_masks, image_paths), 
            total=len(image_paths), 
            desc="Saving depth maps"
        )):
            output_file = output_dir / img_path.stem
            
            # Save depth map with segment information
            self.save_depth_map(
                depth, 
                output_file, 
                mask=mask,
                metadata={
                    'model': 'GeometryCrafter',
                    'model_type': self.model_type,
                    'window_size': self.window_size,
                    'overlap': self.overlap,
                    'frame_index': frame_start + i,
                    'segment_start': frame_start,
                    'segment_end': frame_end
                }
            )
            
            # Save visualization if requested
            if self.save_visualization:
                vis = self.create_visualization(depth, mask)
                vis_path = output_dir / f"{img_path.stem}_vis.png"
                cv2.imwrite(str(vis_path), vis)
        
        # Save point clouds as PLY files if requested
        if self.save_ply:
            logger.info("Saving point clouds as PLY files...")
            # point_maps already extracted before clearing GPU memory
            
            # Create PLY output directory
            ply_dir = output_dir.parent / "point_clouds"
            ply_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (points, mask, img_path) in enumerate(tqdm(
                zip(point_maps, valid_masks, image_paths), 
                total=len(image_paths), 
                desc="Saving PLY files"
            )):
                # Get RGB colors from original frames
                rgb_image = frames[i]  # Already in [0, 1] range
                
                # Save as PLY
                ply_path = ply_dir / f"{img_path.stem}.ply"
                self.save_point_cloud_ply(points, mask, rgb_image, ply_path)
        
        # Clear GPU memory after processing segment
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log memory usage
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            logger.info(f"GPU Memory - Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")
        
        # Create metadata file
        metadata = {
            'num_frames': len(image_paths),
            'total_frames': len(all_image_paths),
            'segment_start': frame_start,
            'segment_end': frame_end,
            'model': 'GeometryCrafter',
            'model_type': self.model_type,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'height': height,
            'width': width,
            'output_format': self.output_format,
            'downsample_ratio': self.downsample_ratio
        }
        
        # For segmented processing, save segment-specific metadata
        if frame_start > 0 or frame_end < len(all_image_paths):
            metadata_filename = f'depth_metadata_segment_{frame_start:06d}_{frame_end:06d}.json'
        else:
            metadata_filename = 'depth_metadata.json'
        
        with open(output_dir.parent / metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'depth_dir': output_dir,
            'num_frames': len(image_paths),
            'metadata': metadata
        }
    
    def _load_images_as_video(self, image_paths: List[Path]) -> np.ndarray:
        """Load images as video frames.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Numpy array of shape (T, H, W, 3) with values in [0, 1]
        """
        frames = []
        for img_path in tqdm(image_paths, desc="Loading images"):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        
        if not frames:
            raise ValueError("No valid images loaded")
        
        frames = np.array(frames, dtype=np.float32) / 255.0
        return frames
    
    def process_sequence(self, 
                        images: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Process a sequence of images to extract depth maps.
        
        This method is required by the base class but not typically used
        for GeometryCrafter which processes entire directories.
        
        Args:
            images: Video frames as numpy array (T, H, W, 3) or list
            
        Returns:
            Tuple of (depth_maps, masks)
        """
        # Convert to tensor
        if isinstance(images, list):
            images = np.array(images)
        
        device = 'cpu' if self.low_memory_usage else self.device
        frames_tensor = torch.tensor(images.astype("float32"), device=device).float()
        if frames_tensor.dim() == 4:
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # THWC -> TCHW
        
        # Check dimensions
        height, width = frames_tensor.shape[-2:]
        if height % 64 != 0 or width % 64 != 0:
            height = (height // 64) * 64
            width = (width // 64) * 64
            import torch.nn.functional as F
            frames_tensor = F.interpolate(frames_tensor, (height, width), mode='bicubic', antialias=True)
        
        # Process with GeometryCrafter
        with torch.inference_mode():
            rec_point_map, rec_valid_mask = self.pipe(
                frames_tensor,
                self.point_map_vae,
                self.prior_model,
                height=height,
                width=width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                window_size=min(self.window_size, len(frames_tensor)),
                decode_chunk_size=self.decode_chunk_size,
                overlap=self.overlap if len(frames_tensor) > self.window_size else 0,
                force_projection=self.force_projection,
                force_fixed_focal=self.force_fixed_focal,
                use_extract_interp=self.use_extract_interp,
                track_time=False,
                low_memory_usage=self.low_memory_usage
            )
        
        # Extract depth and masks
        depth_maps = rec_point_map[..., 2].detach().cpu().numpy()
        valid_masks = rec_valid_mask.detach().cpu().numpy()
        
        return list(depth_maps), list(valid_masks)
    
    def save_point_cloud_ply(self, 
                           point_map: np.ndarray, 
                           mask: np.ndarray, 
                           rgb_image: np.ndarray,
                           output_path: Path):
        """Save point cloud as PLY file.
        
        Args:
            point_map: 3D coordinates (H, W, 3)
            mask: Valid point mask (H, W)
            rgb_image: RGB image (H, W, 3) in [0, 1] range
            output_path: Path to save PLY file
        """
        # Get valid points
        valid_indices = np.where(mask)
        valid_points = point_map[valid_indices]
        
        # Get corresponding colors
        valid_colors = (rgb_image[valid_indices] * 255).astype(np.uint8)
        
        # Write PLY file
        num_points = len(valid_points)
        
        with open(output_path, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Write points
            for i in range(num_points):
                x, y, z = valid_points[i]
                r, g, b = valid_colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")