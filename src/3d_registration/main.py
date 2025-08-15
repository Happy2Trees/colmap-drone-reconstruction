import argparse
import os
import json
from typing import Optional, List

try:
    from .lib3d.keypoint_tracking import run_tracking
except Exception:  # fallback if executed as script
    from lib3d.keypoint_tracking import run_tracking  # type: ignore

try:
    from .lib3d.triangulate_ransac_ba import run_triangulation
except Exception:
    from lib3d.triangulate_ransac_ba import run_triangulation  # type: ignore

try:
    from .lib3d.evaluation.ply_from_measure import measure_to_ply, xyz_to_ply_sampled
except Exception:  # fallback if relative import fails when run as script
    from lib3d.evaluation.ply_from_measure import measure_to_ply, xyz_to_ply_sampled  # type: ignore

try:
    from .lib3d.evaluation.registration_o3d import register_and_eval
except Exception:
    from lib3d.evaluation.registration_o3d import register_and_eval  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser(description="3D registration pipeline orchestrator")
    # Inputs
    ap.add_argument('--json_dir', required=False, help='Detection JSON directory for tracking')
    ap.add_argument('--model_dir', required=False, help='COLMAP model dir (cameras.bin, images.bin)')
    ap.add_argument('--out_prefix', required=False, default='output/triangulated', help='Output prefix for triangulation results')
    ap.add_argument('--exp_dir', required=False, default=None, help='Experiment directory to store all outputs. When set, overrides out_prefix, measurement_ply_out, and reg_debug_dir to live under this directory.')

    # Optional measurement inputs
    ap.add_argument('--xyz_npy', default='/hdd2/0321_block_drone_video/colmap/data/measurement_xyz.npy', help='Measurement XYZ NPY path')
    ap.add_argument('--candidate_txt', default='/hdd2/0321_block_drone_video/colmap/data/candidate_list.txt', help='Candidate index txt path')
    ap.add_argument('--measurement_ply_out', default='output/measurement_candidates.ply', help='Output PLY path for measurement candidates')
    # Optional: Save a sampled PLY from xyz_npy (independent of candidate list)
    ap.add_argument('--xyz_sample_ply_out', default=None, help='Output PLY for sampled points from xyz_npy. Not saved unless set and count>0.')
    ap.add_argument('--xyz_sample_count', type=int, default=0, help='Number of points to sample from xyz_npy for PLY. 0 disables.')

    # Optional: Save aligned triangulated source PLY after registration
    ap.add_argument('--aligned_ply_out', default=None, help='Output path for aligned triangulated points PLY after registration. Defaults under exp_dir if set.')

    # Step control
    ap.add_argument('--skip_tracking', action='store_true', help='Skip tracking step')
    ap.add_argument('--reuse_tracks_csv', default=None, help='Reuse existing tracks.csv instead of running tracking')
    ap.add_argument('--skip_triangulation', action='store_true', help='Skip triangulation step')
    ap.add_argument('--reuse_triangulated_ply', default=None, help='Reuse existing triangulated PLY instead of running triangulation')
    ap.add_argument('--skip_measure_ply', action='store_true', help='Skip measurement->PLY step')
    ap.add_argument('--skip_register', action='store_true', help='Skip registration/evaluation step')
    # Measurement PLY options
    ap.add_argument('--dedup_measure', action='store_true', default=True, help='Deduplicate candidate indices (default True)')
    ap.add_argument('--no_dedup_measure', action='store_true', help='Disable deduplication of candidate indices')

    # Tracking params
    ap.add_argument('--pattern', default='*.json')
    ap.add_argument('--conf_min', type=float, default=0.8)
    ap.add_argument('--kp_conf_min', type=float, default=None)
    ap.add_argument('--take', default='all', choices=['all', 'first'])
    ap.add_argument('--fill_missing', action='store_true', default=True)
    ap.add_argument('--no_fill_missing', action='store_true', help='Disable gap filling')
    ap.add_argument('--max_match_dist', type=float, default=80.0)
    ap.add_argument('--max_missed', type=int, default=2)
    ap.add_argument('--prefer_hungarian', action='store_true', default=True)
    ap.add_argument('--no_prefer_hungarian', action='store_true', help='Disable Hungarian even if available')

    # Triangulation params
    ap.add_argument('--min_views', type=int, default=2)
    ap.add_argument('--ransac_thresh', type=float, default=3.0)
    ap.add_argument('--ransac_iters', type=int, default=400)
    ap.add_argument('--min_inliers', type=int, default=3)
    ap.add_argument('--pos_depth_ratio', type=float, default=0.5)

    # Registration/Evaluation (PLY-based Umeyama + IRLS)
    ap.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size for NN pairing downsampling (alignment uses matched points)')
    ap.add_argument('--reg_debug_dir', default='output/register_debug', help='Directory to save register_eval.json and debug indices')
    # ap.add_argument('--aligned_ply_out', default=None, help='Output path for aligned triangulated points PLY')
    # Explicit correspondences by vertex indices
    ap.add_argument('--est_order', default='', help='Comma-separated source vertex indices (EST_ORDER). If empty, uses NN matching')
    ap.add_argument('--gt_order',  default='', help='Comma-separated target vertex indices (GT_ORDER). If empty, uses NN matching')
    # IRLS/Eval options
    ap.add_argument('--eval_methods', default='L2,huber,cauchy', help='Comma-separated methods: L2,huber,cauchy')
    ap.add_argument('--eval_with_scale', default='true,false', help='Comma-separated booleans for with_scale options: true,false')
    # NN pairing options
    ap.add_argument('--nn_bidir', default='true', help='Mutual NN required when using NN match (true/false)')
    ap.add_argument('--nn_max_dist', type=float, default=None, help='Max NN distance threshold (same units as coordinates). None to disable')

    return ap.parse_args()


def main():
    args = parse_args()

    # Centralize outputs under experiment directory if provided
    if args.exp_dir:
        # If a bare name is given (not absolute and not starting with 'outputs/'),
        # place under outputs/<name> for consistency.
        exp_arg = args.exp_dir
        if not os.path.isabs(exp_arg) and not exp_arg.startswith('outputs'+os.sep) and not exp_arg.startswith('outputs/'):
            exp_dir = os.path.join('outputs', exp_arg)
        else:
            exp_dir = exp_arg
        os.makedirs(exp_dir, exist_ok=True)
        # Force all outputs to live under exp_dir for tidiness
        args.out_prefix = os.path.join(exp_dir, 'triangulated')
        args.measurement_ply_out = os.path.join(exp_dir, 'measurement_candidates.ply')
        args.reg_debug_dir = os.path.join(exp_dir, 'register_debug')
        # Do not auto-create sampled PLY path; only save when explicitly requested
        if not args.aligned_ply_out:
            # Standard name next to triangulated outputs
            args.aligned_ply_out = os.path.join(exp_dir, 'triangulated_points_aligned.ply')
        print(f'[*] Using experiment dir: {exp_dir}')

    # Resolve boolean flips
    fill_missing = False if args.no_fill_missing else args.fill_missing
    prefer_hungarian = False if args.no_prefer_hungarian else args.prefer_hungarian
    dedup_measure = False if args.no_dedup_measure else args.dedup_measure

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    # 1) Tracking
    tracks_csv: Optional[str] = None
    if args.reuse_tracks_csv:
        tracks_csv = args.reuse_tracks_csv
        print(f'[*] Reusing tracks CSV: {tracks_csv}')
    elif not args.skip_tracking:
        if not args.json_dir:
            raise ValueError('--json_dir is required unless --reuse_tracks_csv or --skip_tracking is set.')
        out_json = os.path.join(os.path.dirname(args.out_prefix) or '.', 'tracks.json')
        out_csv = os.path.join(os.path.dirname(args.out_prefix) or '.', 'tracks.csv')
        print('[*] Running tracking...')
        _, tracks_csv_path = run_tracking(
            json_dir=args.json_dir,
            pattern=args.pattern,
            conf_min=args.conf_min,
            kp_conf_min=args.kp_conf_min,
            take=args.take,
            fill_missing=fill_missing,
            max_match_dist=args.max_match_dist,
            max_missed=args.max_missed,
            prefer_hungarian=prefer_hungarian,
            out_json=out_json,
            out_csv=out_csv,
        )
        tracks_csv = tracks_csv_path
        print(f'[*] Tracking done. CSV: {tracks_csv}')
    else:
        print('[*] Skipping tracking step.')

    # 2) Triangulation
    triangulated_ply: Optional[str] = None
    if args.reuse_triangulated_ply:
        triangulated_ply = args.reuse_triangulated_ply
        print(f'[*] Reusing triangulated PLY: {triangulated_ply}')
    elif not args.skip_triangulation:
        if not (args.model_dir and tracks_csv):
            raise ValueError('--model_dir and tracks CSV are required unless reusing or skipping triangulation.')
        print('[*] Running triangulation...')
        out_csv, out_ply = run_triangulation(
            model_dir=args.model_dir,
            tracks_csv=tracks_csv,
            out_prefix=args.out_prefix,
            min_views=args.min_views,
            ransac_thresh=args.ransac_thresh,
            ransac_iters=args.ransac_iters,
            min_inliers=args.min_inliers,
            pos_depth_ratio=args.pos_depth_ratio,
        )
        triangulated_ply = out_ply
        print(f'[*] Triangulation done. PLY: {triangulated_ply}')
    else:
        print('[*] Skipping triangulation step.')

    # 3) Measurement -> PLY
    measurement_ply = args.measurement_ply_out
    if not args.skip_measure_ply:
        print('[*] Generating measurement candidates PLY...')
        count = measure_to_ply(args.xyz_npy, args.candidate_txt, measurement_ply, deduplicate=dedup_measure)
        print(f'[*] Saved measurement PLY: {measurement_ply} ({count} points)')
    else:
        print('[*] Skipping measure->PLY step.')

    # 3b) Optional: also save a sampled PLY from the raw measurement xyz_npy
    if (args.xyz_sample_count and args.xyz_sample_count > 0) and args.xyz_sample_ply_out:
        try:
            print('[*] Saving sampled measurement XYZ to PLY...')
            cnt = xyz_to_ply_sampled(args.xyz_npy, args.xyz_sample_ply_out, sample_count=int(args.xyz_sample_count))
            print(f'[*] Saved sampled measurement PLY: {args.xyz_sample_ply_out} ({cnt} points)')
        except Exception as e:
            print(f'[!] Failed saving sampled measurement PLY: {e}')

    # 4) Registration/Evaluation (PLY-based Umeyama + IRLS)
    if not args.skip_register:
        if not (triangulated_ply and os.path.exists(triangulated_ply)):
            raise ValueError('Triangulated PLY missing; provide --reuse_triangulated_ply or run triangulation.')
        if not os.path.exists(measurement_ply):
            raise ValueError('Measurement PLY missing; run measure_to_ply or provide correct path.')

        # Parse lists
        def parse_int_list(s: str) -> List[int]:
            s = s.strip()
            if not s:
                return []
            return [int(x.strip()) for x in s.split(',') if x.strip()]

        def parse_bool_list(s: str) -> List[bool]:
            out: List[bool] = []
            for tok in s.split(','):
                t = tok.strip().lower()
                if not t:
                    continue
                if t in ('1','true','t','yes','y'): out.append(True)
                elif t in ('0','false','f','no','n'): out.append(False)
                else: raise ValueError(f'Invalid boolean: {tok}')
            return out

        est_order_list = parse_int_list(args.est_order)
        gt_order_list = parse_int_list(args.gt_order)
        methods = [m.strip() for m in args.eval_methods.split(',') if m.strip()]
        with_scale_opts = parse_bool_list(args.eval_with_scale)
        nn_bidir = parse_bool_list(args.nn_bidir)[0] if args.nn_bidir else True

        # Decide explicit vs NN matching
        if est_order_list and gt_order_list:
            if len(est_order_list) != len(gt_order_list):
                raise ValueError('Length mismatch: est_order vs gt_order')
            source_indices = est_order_list
            target_indices = gt_order_list
        else:
            source_indices = None
            target_indices = None

        print('[*] Running PLY-based registration/evaluation (Umeyama + IRLS)...')
        stats = register_and_eval(
            source_ply=triangulated_ply,
            target_ply=measurement_ply,
            voxel_size=float(args.voxel_size),
            save_debug_dir=args.reg_debug_dir,
            aligned_ply_out=args.aligned_ply_out,
            # matching / eval options
            source_indices=source_indices,
            target_indices=target_indices,
            methods=methods,
            with_scale_options=with_scale_opts,
            nn_bidir=bool(nn_bidir),
            nn_max_dist=args.nn_max_dist,
        )
        print(f'stats : {stats}')
        # Inform about aligned PLY if saved
        try:
            if args.aligned_ply_out and os.path.exists(args.aligned_ply_out):
                print(f'[*] Saved aligned PLY: {args.aligned_ply_out}')
        except Exception:
            pass

        # Save summary next to out_prefix
        sum_path = os.path.join(os.path.dirname(args.out_prefix) or '.', 'register_eval.json')
        try:
            with open(sum_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f'[*] Saved: {sum_path}')
        except Exception as e:
            print(f'[!] Failed to save register_eval.json: {e}')
        print('[*] Registration/Eval done.')
    else:
        print('[*] Skipping registration step.')


if __name__ == '__main__':
    main()
