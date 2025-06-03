import os
import sys
import glob
import json
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_rootdir', type=str, required=True, help='input root directory')
    parser.add_argument('--output_json', type=str, default='casper.json', help='saved json file')
    parser.add_argument('--repeat_omnimatte_tripod', type=int, defauilt=256, help='repeat times for omnimatte+tripod')
    parser.add_argument('--repeat_kubric', type=int, default=40, help='repeat times for kubric')
    parser.add_argument('--repeat_object_paste', type=int, default=1, help='repeat times for object paste')
    args = parser.parse_args()

    root_dir = args.input_rootdir
    output_json_path = args.output_json
    export_list = []
    count = 0
    data_list = []

    water_examples = [
        'flamingo', 'kite-surf', 'pexels-bike-puddle', 'pexels-bike-puddle2', 'pexels-boat-bridge', 
        'pexels-boat-ocean', 'pexels-skate-puddle', 'pexels-speedboat', 'pexels-suv-rain', 'pexels-towing-boat',
    ]
    water_weight = 2  # we double the training weight of water examples for public models

    r_omnimatte_tripod = args.repeat_omnimatte_tripod
    r_kubric = args.repeat_kubric
    r_op = args.repeat_object_paste

    # Please adjust the repeat times for each category (r_omnimatte+tripod, r_kubric, r_op) according to your needs.
    # Supppose you generate N_kubric Kubric videos, N_op ObjectPaste videos and with our released 46 omnimatte and tripod videos,
    # Total number of instances =   46 * r_omnimatte+tripod + N_kubric * r_kubric + N_op * r_op
    # the weight of each category is:
    #   Omnimatte+tripod:           46 * r_omnimatte+tripod / Total number of instances
    #   Kubric:                     N_kubric * r_kubric / Total number of instances
    #   Object Paste:               N_op * r_op / Total number of instances
    # In our paper, we make the training set contain 50% omnimatte+tripod, 48% kubric, and 2% object-paste.
    for category, repeat_times in [
        ('omnimatte+tripod', r_omnimatte_tripod), ('kubric', r_kubric), ('object-paste', r_op)
    ]:
        if os.path.isdir(os.path.join(root_dir, category)):
            caption_path = os.path.join(root_dir, category).rstrip('/') + '.json'
            assert os.path.exists(caption_path), f"Caption file {caption_path} does not exist."

            captions = json.load(open(caption_path))

            data_dirs = sorted(list(glob.glob(os.path.join(root_dir, category, '*'))))

            data_dirs = [data_dir for data_dir in data_dirs if os.path.isdir(data_dir)]
            for data_dir in data_dirs:
                vid = os.path.basename(data_dir)
                found = False
                for caption in captions:
                    if vid == caption['path']:
                        found = True
                        break
                if found:
                    if vid in water_examples:
                        r = repeat_times * water_weight
                        print(f'water examples! {vid} {r}')
                    else:
                        r = repeat_times
                    for _ in range(r):
                        data_list.append((data_dir, caption['short_prompt'].lstrip('"').rstrip('"')))
                else:
                    print(f"Caption not found for {vid}")

    for data_dir, prompt in tqdm(data_list):
        export_list.append(
            {
                "file_path": os.path.abspath(data_dir),
                "text": prompt,
                "type": "video_mask_tuple",
            }
        )
        count += 1
    print(count)
    json.dump(export_list, open(output_json_path, 'w'), indent=4)
