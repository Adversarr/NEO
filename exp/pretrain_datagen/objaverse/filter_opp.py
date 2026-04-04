"""
[{'UID': '000074a334c541878360457c672b6c2e',
  'style': 'realistic',
  'score': 3,
  'is_multi_object': 'false',
  'is_scene': 'false',
  'is_figure': 'false',
  'is_transparent': 'false',
  'is_single_color': 'false',
  'density': 'low'}, ...]

inspect possible values only.
multi_object: {'false', 'true'}
scene: {'false', 'true'}
figure: {'false', 'true'}
transparent: {'false', 'true'}
single_color: {'false', 'true'}
density: {'mid', 'high', 'low'}
style: {'realistic', 'cartoon', 'anime', 'sci-fi', 'scanned', 'other', 'arcade'}
score: {0, 1, 2, 3}
"""

import json
from argparse import ArgumentParser
from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="annotated_800k.json", help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, default="outputs/objaverse_plus_plus.txt", help="Path to the output file.")
    parser.add_argument("--min-score", type=int, default=2, help="Minimum score to filter. (inclusive)")
    parser.add_argument("--max-score", type=int, default=3, help="Maximum score to filter. (inclusive)")

    # output
    parser.add_argument("--enable-multi-object", action="store_true", help="Enable multi-object scenes.")
    parser.add_argument("--enable-scene", action="store_true", help="Enable scene scenes.")
    parser.add_argument("--enable-figure", action="store_true", help="Enable figure scenes.")
    parser.add_argument("--enable-transparent", action="store_true", help="Enable transparent scenes.")
    parser.add_argument("--enable-single-color", action="store_true", help="Enable single-color scenes.")
    parser.add_argument("--min-density", type=str, default="mid", help="Minimum density to filter. (inclusive)")
    parser.add_argument("--max-density", type=str, default="high", help="Maximum density to filter. (inclusive)")

    # inspect input
    parser.add_argument("--inspect", action="store_true", help="Inspect the input JSON file.")

    return parser.parse_args()

def main(args) -> None:
    with open(args.input, "r") as f:
        data = json.load(f)

    if args.inspect:
        print('inspect possible values only.')
        all_multi_object = set()
        all_scene = set()
        all_figure = set()
        all_transparent = set()
        all_single_color = set()
        all_density = set()
        all_style = set()
        all_score = set()
        for item in data:
            all_multi_object.add(item['is_multi_object'])
            all_scene.add(item['is_scene'])
            all_figure.add(item['is_figure'])
            all_transparent.add(item['is_transparent'])
            all_single_color.add(item['is_single_color'])
            all_density.add(item['density'])
            all_style.add(item['style'])
            all_score.add(item['score'])

        print(f'multi_object: {all_multi_object}')
        print(f'scene: {all_scene}')
        print(f'figure: {all_figure}')
        print(f'transparent: {all_transparent}')
        print(f'single_color: {all_single_color}')
        print(f'density: {all_density}')
        print(f'style: {all_style}')
        print(f'score: {all_score}')

        exit(0)

    # filter
    filtered_data = []
    DENSITY_SCORE = {'low': 0, 'mid': 1, 'high': 2}
    min_density_score = DENSITY_SCORE[args.min_density]
    max_density_score = DENSITY_SCORE[args.max_density]
    print(args)
    for item in data:
        available = True
        if item['score'] < args.min_score or item['score'] > args.max_score:
            available = False
        if item['is_multi_object'] == 'true':
            available = available if args.enable_multi_object else False
        if item['is_scene'] == 'true':
            available = available if args.enable_scene else False
        if item['is_figure'] == 'true':
            available = available if args.enable_figure else False
        if item['is_transparent'] == 'true':
            available = available if args.enable_transparent else False
        if item['is_single_color'] == 'true':
            available = available if args.enable_single_color else False
        density_score = DENSITY_SCORE[item['density']]
        if density_score < min_density_score or density_score > max_density_score:
            available = False

        if available:
            filtered_data.append(item['UID'])

    print(f'filtered: {len(filtered_data)}/{len(data)}, ratio={len(filtered_data)/len(data)}')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(filtered_data))
    print(f'Written output: {out}')

if __name__ == "__main__":
    args = parse_args()
    main(args)