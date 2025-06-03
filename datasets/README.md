# Training Data for a Casper model

### Table of Contents:
- [Omnimatte & Tripod](#omnimatte&tripod)
- [Kubric](#kubric)
- [ObjectPaste](#objectpaste)
- [Merge all categories](#merge)


## Omnimatte & Tripod <a name="omnimatte&tripod"></a>
- Please download the data and the caption json file from the google drive [here](https://drive.google.com/drive/folders/1qnIduW8v9U6cobOyYyDP3yGcxo1PjU97?usp=sharing).

## Kubric <a name="kubric"></a>
This code was modified from the original [Kubric](https://github.com/google-research/kubric/tree/main) with an additional transparent material.

- The Kubric data used to train our models can also be found [here](https://drive.google.com/drive/folders/1qnIduW8v9U6cobOyYyDP3yGcxo1PjU97?usp=sharing).

### Environment
```
cd ./datasets
pip install -e .
```

### Generate a sequence
```
python effect-removal-synthesis.py \
    --seed 43 \
    --output_dir ./train-casper/kubric \
    --video_length 80 \
    --mask_mode trimask
```

- You may use a smaller number for `--video_length` to quickly test if the installation is successful.
- The outputs will be saved in `./train-casper/kubric`:
    - `rgb_full`: the input of the model (before object effect removal)
    - `mask`: the input mask. 1 for the preservation and 0 for the removal. And 0.5 for the background if the `mask_mode` is trimask.
    - `rgb_removed`: the target output of the object-effect removal.

Please note that the current transparent material still cannot handle a few complex objects, causing some uncleaned pixels remaining in the target removal outputs. Some manual filtering may require to remove such cases.


#### Generate a batch of sequences
You may use a simple iteration script to run the above generation command with different random seeds.

#### Text Prompt Generation
You may use `caption_videos.py` to get the input text prompts for the generated kubric videos.


## ObjectPaste <a name="objectpaste"></a>

### Data source
Due to the license, please download a video dataset on your own (e.g., YouTube Video Object Segmentation). Once you have your own set of videos, you will need to obtain masks (e.g., via segmenters such as SAM2.) and captions.

You may use the following script to generate the captions. The scripts is modified from [here](https://gist.github.com/a-r-r-o-w/4dee20250e82f4e44690a02351324a4a) (Thanks to the authors!)

The source videos and masks should be 
```
object-paste
├─ videos
│  ├─ XXXXX.mp4
│  └─ YYYYY.mp4
├─ masks
│  ├─ XXXXX.mp4
│  └─ YYYYY.mp4
└─ caption.json
```

### Generate ObjectPaste data
```
python generate_batch \
    --source_rootdir ./train-casper/object-paste \
    --output_dir OUTPUT_DIR \
    --num_tuples 1
```

- set `--num_tuples` as a larger number to generate more examples.
- The outputs will be saved in `./train-casper/object-paste`:
    - `rgb_full`: the input of the model (before object effect removal)
    - `mask`: the input mask. 1 for the preservation and 0 for the removal. And 0.5 for the background if the `mask_mode` is trimask.
    - `rgb_removed`: the target output of the object-effect removal.

## Merge All Categories
### Data preparation
Put all geneated categories under the same directory (e.g., `./train-casper`):

```
./train-casper/
├── kubric/
├── object-paste/
├── omnimatte+tripod/
├── kubric.json
├── object-paste.json
└── omnimatte+tripod.json
```
The json files are generated captions for the categories.

### Run merging script
```
cd datasets
python merge_training_data.py --input_rootdir ./train-casper --output_json casper.json
```
- Balancing the categories:
    - Please adjust the repeat times `--repeat_omnimatte_tripod`, `--repeat_kubric`, and `--repeat_object_paste` ($r_{ot}$, $r_{kubric}$, $r_{op}$, respectively) according to your needs.
    - Supppose you have $N_{kubric}$ Kubric videos, $N_{op}$ ObjectPaste videos, along with our released 46 omnimatte and tripod videos,
        - Total number of instances $N_{total}$ =   46 * $r_{ot}$ + $N_{kubric} * $r_{kubric}$ + $N_{op}$ * $r_{op}$
        - the weight of each category is:
            - Omnimatte+tripod:           46 * $r_{ot}$ / $N_{total}$
            - Kubric:                     $N_{kubric}$ * $r_{kubric}$ / $N_{total}$
            - Object Paste:               $N_{op}$ * $r_{op}$ / $N_{total}$
    - In our paper, we make the training set contain 50% omnimatte+tripod, 48% kubric, and 2% object-paste.

- Replace the output json path for in the training scripts of `scripts/xxxxx/train_casper.sh`. Please check [Casper Training](../README.md#casper-training) for the details.