# CTC Forced Aligner

A repository that runs audio alignment with the transcriptions given. This repository needs to be built from source with the edited `ctc_forced_aligner/forced_align_impl.cpp` file, as the original source code is very fragile to exception handling at the lower level C++ code. To read the original README, refer to `README_source.md`.   

**As of now, only filtering audio via coverage threshold will be supported. Future work will include filtering audio via negative log confidence score at the word level.**   

## Download the alignment model
You can download the alignment model weights [here](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)    

## Environment Setup

Clone the repository, take note of cloning the submodule `uroman` with the `--recursive` flag.   

```shell
git clone --recursive git@github.com:nicholasneo78/ctc-forced-aligner.git
```

Build the docker image.   

```shell
docker-compose -f build/docker-compose.yml build
```

Check the `build/docker-compose.yml` file and change the mounted code, dataset and model as desired (the directories which stores this code repo, target dataset, and the downloaded alignment model). Enter into the docker image.   

```shell
docker-compose -f build/docker-compose.yml up -d
docker exec -it ctc_forced_aligner bash
```

Once you are in the docker container, you are ready to run alignment.
   
## Data Preparation
Ensure that your target dataset manifest is in the nemo format. That is for example

```shell
some_data/
    data/
    manifest.json
```
where manifest.json is in this format
```
{"audio_filepath": "data/xxx1.wav", "duration": 3.0, "text": "this is text one", "language": "en", "subset": "a"}
{"audio_filepath": "data/xxx2.wav", "duration": 4.0, "text": "this is text two","language": "en", "subset": "b"}
...
```

Note that `audio_filepath`, `text` and `language` are required field for alignment.   

## Execution
As of now, the model only supports the 10 main languages (refer to `aligner.py`) due to the ISO639-1 to ISO639-3 mapping limitations. We will work on it to support more languages as supported by the MMS's 1130 languages.    

Go to `aligner.py` change the following parameters to your paths:   
`root_dir`: the root directory to the input manifest directory and the aligned manifest directory   
`input_manifest_dir`: full absolute input manifest directory   
`output_manifest_dir`: full absolute output manifest directory    
`model_dir`: full absolute model directory that contains the alignment model   
`emission_batch_size`: ctc decoding batch size per timestep during the alignment process   
`coverage_threshold`: how much coverage the transcriptions align to the audio, this is to weed out extreme short transcriptions with long audio due to poor transcription    

Run the following code:
```shell
python3 aligner.py
```

The output manifest will skip entries that either caused error during the alignment due to oversized ctc prediction tree or entries that does not meet the coverage threshold.   

## Extra Note
Data on NAS is also support, just need to uncomment `- nas:/nas` line in `build/docker-compose.yml`, and the data path will be mounted to `/nas` in the docker container