"""Reproducible dataset/large-export verification; never changes input files.

Example: python scripts/diagnose_dataset.py C:\Code\ngoai_nha\1 --mode auto --mp 100
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from panorama import PanoramaStitcher, StitchConfig
from panorama.geometry import make_thumbnail
from panorama.image_io import decode_image, encode_image


def peak_memory_mb():
    if os.name != 'nt':
        return None
    from ctypes import wintypes
    class MemoryCounters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                                        ctypes.c_void_p, wintypes.DWORD]
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return round(counters.PeakWorkingSetSize/1e6, 2)
    return None


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--mode', choices=['auto', 'planar', 'cylindrical'], default='auto')
    parser.add_argument('--mp', default='native', help='native, 80, 100, 150 (small values for quick QA)')
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--no-crop', action='store_true')
    parser.add_argument('--dimension', type=int, default=1920)
    parser.add_argument('--format', choices=['jpg', 'png'], default='jpg')
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    paths = sorted(p for p in args.directory.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    images = [decode_image(p.read_bytes(), p.name) for p in paths]
    target = None if args.mp == 'native' else float(args.mp)
    args.output.mkdir(parents=True, exist_ok=True)
    last = [-1]
    def progress(value, message):
        percent = int(value*10)*10
        if percent != last[0]:
            print(f'{percent}% {message}', flush=True)
            last[0] = percent
    started = perf_counter()
    result = PanoramaStitcher(StitchConfig(projection=args.mode, output_megapixels=target,
                auto_crop=not args.no_crop, max_dimension=args.dimension)).stitch(
                images, [p.name for p in paths], automatic_order=not args.manual, progress=progress)
    stem = f'panorama_{args.mode}_{args.mp}MP'+('_uncropped' if args.no_crop else '')
    output = args.output/(stem+'.'+args.format)
    encoded = encode_image(result.panorama, '.'+args.format)
    output.write_bytes(encoded)
    preview = args.output/(stem+'_preview.jpg')
    preview.write_bytes(encode_image(make_thumbnail(result.panorama, 1800), '.jpg'))
    report = {
        'inputs': [str(p) for p in paths], 'input_shapes': [list(im.shape) for im in images],
        'requested_mode': args.mode, 'selected_mode': result.projection,
        'reason': result.projection_reason, 'ordered_names': result.ordered_names,
        'target_megapixels': target, 'shape': list(result.panorama.shape),
        'actual_megapixels': result.panorama.shape[0]*result.panorama.shape[1]/1e6,
        'coverage_verified': result.coverage_verified, 'interpolated': result.interpolated,
        'natural_shape': list(result.natural_shape), 'crop_fraction': result.crop_fraction,
        'camera_focals': result.camera_focals, 'quality': result.quality,
        'pair_diagnostics': [{'source':p.source_position, 'destination':p.destination_position,
                             'matches':p.match_count, 'inliers':p.inlier_count, 'mean_error':p.mean_error}
                            for p in result.pair_diagnostics],
        'stitch_seconds': result.elapsed_seconds, 'total_seconds': perf_counter()-started,
        'peak_working_set_mb': peak_memory_mb(), 'output': str(output.resolve()),
        'output_bytes': len(encoded), 'preview': str(preview.resolve()), 'warnings': result.warnings}
    report_path = args.output/(stem+'.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
