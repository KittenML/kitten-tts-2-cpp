#!/usr/bin/env python3
"""Copy publisher exports into a KittenTTS2 model repository and update its config."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True, help='Default decoder export including model-tq2_1.gguf')
    parser.add_argument('--student-w4', type=Path)
    parser.add_argument('--student-w8', type=Path)
    args = parser.parse_args()
    config_path = args.repo / 'config.json'
    config = json.loads(config_path.read_text())
    if config['type'] != 'KITTEN2':
        raise ValueError('only KITTEN2 is supported')

    def copy(source, relative):
        target = args.repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        return {'file': relative, 'size': target.stat().st_size}

    sources = {'default': args.assets}
    if args.student_w4:
        sources['student_w4'] = args.student_w4
    if args.student_w8:
        sources['student_w8'] = args.student_w8
    for name, source in sources.items():
        exported = json.loads((source / 'config.json').read_text())
        if exported['type'] != 'KITTEN2' or exported.get('cpp_decoder', 'default') != name:
            raise ValueError(f'{source}: wrong decoder export for {name}')
        for field in ('token_map', 'sample_rate'):
            if exported[field] != config[field]:
                raise ValueError(f'{source}: {field} does not match repository')
        for file in ('decoder.pt', 'voices.json'):
            if not (source / file).is_file():
                raise ValueError(f'missing {source / file}')
    manifest = {'version': 1, 'gguf': copy(args.assets / 'model-tq2_1.gguf', 'cpp/model-tq2_1.gguf'), 'decoders': {}}
    for name, source in sources.items():
        manifest['decoders'][name] = {
            'torchscript': copy(source / 'decoder.pt', f'cpp/{name}/decoder.pt'),
            'voices': copy(source / 'voices.json', f'cpp/{name}/voices.json'),
        }
    config['cpp'] = manifest
    temporary = config_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(config, indent=2) + '\n')
    temporary.replace(config_path)
    print(f'Prepared {args.repo}/cpp and {config_path}. Upload cpp/ first, then config.json to the same model repository.')


if __name__ == '__main__':
    main()
