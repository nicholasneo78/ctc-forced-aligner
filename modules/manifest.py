"""
List down all the helper functions that would be called by the main code that is related to importing or exporting of the manifest file throughout the whole repository
"""

import json
from typing import List, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset, IterableDataset, IterableDatasetDict


def load_manifest_nemo(input_manifest_path: str) -> List[Dict[str, str]]:

    '''
    loads the manifest file in Nvidia NeMo format to process the entries and store them into a list of dictionaries

    the manifest file would contain entries in this format:

    {"audio_filepath": "subdir1/xxx1.wav", "duration": 3.0, "language": "en"}
    {"audio_filepath": "subdir1/xxx2.wav", "duration": 4.0, "language": "vi"}
    ---

    input_manifest_path: the manifest path that contains the information of the audio clips of interest
    ---
    returns: a list of dictionaries of the information in the input manifest file
    '''

    dict_list = []

    with open(input_manifest_path, 'r+') as f:
        for line in f:
            dict_list.append(json.loads(line))

    return dict_list


def create_huggingface_manifest(input_manifest_path: str) -> List[Dict[str, str]]:

    '''
    loads the list of dictionaries of the data information, preprocess the manifest format and create the finalized list of dictionaries ready to export into a json file that is ready to be accepted by the huggingface datasets class
    ---

    input_manifest_path: the manifest path that contains the information of the audio clips of interest
    ---
    returns a list of dictionaries of the information in the huggingface format
    '''    

    dict_list = load_manifest_nemo(input_manifest_path=input_manifest_path)

    data_list = []

    for entries in tqdm(dict_list):

        data = {
            'file': entries['audio_filepath'],
                'audio': {
                    'path': entries['audio_filepath'],
                    'sampling_rate': 16000
                },
                'duration': entries['duration'],
                'language': entries['language']
        }

        data_list.append(data)

    return data_list

def export_huggingface_manifest_from_list(manifest_dir: str, dict_list: List[Dict[str, str]]) -> List[Dict[str, str]]:

    '''
    loads the list of dictionaries of the data information, preprocess the manifest format and create the finalized list of dictionaries ready to export into a json file that is ready to be accepted by the huggingface datasets class
    ---

    dict_list: the manifest dict list that contains the information of the audio clips of interest
    ---
    returns a list of dictionaries of the information in the huggingface format
    '''    

    data_list = []

    for entries in tqdm(dict_list):

        data = {
            'file': f"{entries['audio_filepath']}",
                'audio': {
                    'path': f"{entries['audio_filepath']}",
                    'sampling_rate': 16000
                },
                'duration': entries['duration'],
                'language': entries['language']
        }

        data_list.append(data)

    # form the final json manifest that is ready for export
    data_dict = {}
    data_dict['data'] = data_list

    # export to the final json format
    with open(manifest_dir, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data_dict, indent=2, ensure_ascii=False))

    return data_list


def export_splits(manifest_dir: str, data_list: List[Dict[str, str]]) -> None:

    '''
    outputs the respective (train, dev or test) manifest with the split data entries from a list 
    ---

    manifest_dir: the output manifest directory to be exported (train, dev or test)
    data_list: the list of splitted entries 
    '''
    
    with open(manifest_dir, 'w+', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def load_huggingface_manifest(train_dir: str, dev_dir: str) -> IterableDatasetDict:

    '''
    to read the train and dev json manifest file and form the transformers IterableDatasetDict for further preprocessing afterwards
    ---

    train_dir: the manifest file directory of the train set
    dev_dir: the manifest file directory of the dev set
    ---

    returns a huggingface IterableDatasetDict
    '''

    # initiate the path and form the final dataset
    data_files = {
        'train': train_dir,
        'dev': dev_dir
    }

    data = load_dataset("json", data_files=data_files, field="data", streaming=True)

    # load the manifest using vanilla json method to get the length of the train set
    with open(train_dir, 'r+', encoding='utf-8') as fr:
        json_train_data = json.load(fr)['data']

    return data, len(json_train_data)


def load_huggingface_manifest_from_memory(
    train_dict: Dict[str, str],
    dev_dict: Dict[str, str]
)-> Tuple[IterableDatasetDict, int]:

    '''
    to read the train and dev json manifest file and form the transformers IterableDatasetDict for further preprocessing afterwards
    ---

    train_dict: the manifest file directory of the train set
    dev_dict: the manifest file directory of the dev set
    ---

    returns a huggingface IterableDatasetDict
    '''

    train_data_list = train_dict.get('data', [])
    dev_data_list = dev_dict.get('data', [])
    
    def train_gen():
        for item in train_data_list:
            yield item

    def dev_gen():
        for item in dev_data_list:
            yield item

    data = IterableDatasetDict({
        'train': IterableDataset.from_generator(train_gen),
        'dev': IterableDataset.from_generator(dev_gen),
    })

    # load the manifest using vanilla json method to get the length of the train set
    train_data_len = len(train_data_list)

    return data, train_data_len


def load_huggingface_manifest_evaluation(test_dir: str) -> IterableDatasetDict:
        
    '''
    to read the test json manifest file and form the transformers IterableDatasetDict for further preprocessing afterwards
    ---

    test_dir: the manifest file directory of the test set
    ---

    returns a huggingface IterableDatasetDict
    '''

    # initiate the path and form the final dataset
    data_files = {
        'test': test_dir,
    }

    data = load_dataset("json", data_files=data_files, field="data", streaming=True)

    return data


def extract_file_path_from_json(test_dir: str) -> List[str]:

    '''
    reads the json test file and extracts a list of file path of the audio clips
    ---
    
    test_dir: the manifest file directory of the test set
    ---
    
    returns a list of directories of the test set
    '''

    dir_list = []

    with open(test_dir, 'rb') as f:
        data = json.load(f)

    # iterate the list to get the list of filenames
    for entry in data['data']:
        dir_list.append(entry['file'])

    return dir_list


def extract_duration_from_json(test_dir: str) -> List[str]:

    '''
    reads the json test file and extracts a list of file path of the audio clips
    ---
    
    test_dir: the manifest file directory of the test set
    ---
    
    returns a list of directories of the test set
    '''

    dir_list = []

    with open(test_dir, 'rb') as f:
        data = json.load(f)

    # iterate the list to get the list of filenames
    for entry in data['data']:
        dir_list.append(entry['duration'])

    return dir_list


def extract_languages_from_json(test_dir: str) -> List[str]:

    '''
    reads the json test file and extracts a list of languages of the audio clips
    ---
    
    test_dir: the manifest file directory of the test set (hf format)
    ---
    
    returns a list of directories of the test set
    '''

    dir_list = []

    with open(test_dir, 'rb') as f:
        data = json.load(f)

    # iterate the list to get the list of filenames
    for entry in data['data']:
        dir_list.append(entry['language'])

    return dir_list


def extract_unique_languages_from_json(manifest_dir: str, is_hf_format_data: bool) -> List[str]:

    '''
    reads the json test file and extracts the classifier label from the manifest 
    ---
    
    manifest_dir: the manifest file directory of the test set (hf format)
    ---
    
    returns a list of unique classes for the classification
    '''

    languages = set()

    if is_hf_format_data:
        with open(manifest_dir, 'rb') as f:
            data = json.load(f)

        languages = set([entry['language'] for entry in data['data']])
    else:
        data = load_manifest_nemo(input_manifest_path=manifest_dir)
        languages = set([entry['language'] for entry in data])

    return sorted(list(languages))