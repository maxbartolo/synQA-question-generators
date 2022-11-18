import logging
import os
import requests
import tarfile
from tqdm import tqdm


"""
Example usage:
python download_models.py
"""

# Set up logging
logging.basicConfig(level=logging.DEBUG)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

model_urls = {
    'generator_qa_squad.tgz': 'https://dl.fbaipublicfiles.com/dynabench/qa/qgen_squad.tgz',
    'generator_qa_adversarialqa.tgz': 'https://dl.fbaipublicfiles.com/dynabench/qa/qgen_dcombined.tgz',
    'generator_qa_squad_plus_adversarialqa.tgz': 'https://dl.fbaipublicfiles.com/dynabench/qa/qgen_dcombined_plus_squad_10k.tgz',
}


def download(url: str, fname: str, desc: str=None) -> None:
    """Download with progress bar. See: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads"""
    desc = desc if desc is not None else fname
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(desc=fname, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


if __name__ == '__main__':
    for model_filename, url in model_urls.items():
        model_name = model_filename.split('.')[0]
        model_tarfile_path = os.path.join(MODELS_DIR, model_filename)
        model_dir = os.path.join(MODELS_DIR, model_name)

        if not os.path.exists(os.path.join(model_dir, "checkpoint_best.pt")):
            ## Download without progress bar
            # if not os.path.exists(model_tarfile_path):
                # logging.info(f"Downloading {model_filename} from {url}")
                # logging.info(f"This can take a while, please be patient...")
                # r = requests.get(url)
                # with open(model_tarfile_path, 'wb') as f:
                #     f.write(r.content)
            # else:
            #     logging.info(f"Skipping download. {model_tarfile_path} already exists.")

            # Download with progress bar
            if not os.path.exists(model_tarfile_path):
                download(url, model_tarfile_path, url)
            else:
                logging.info(f"Skipping download. The file {model_tarfile_path} already exists.")

            # Extract
            logging.info(f"Extracting {model_filename} to {model_dir}")
            with tarfile.open(model_tarfile_path) as f:
                # Get only the members with extensions (i.e. no directories)
                members = [m for m in f.getmembers() if os.path.splitext(os.path.join(model_dir, m.name))[-1]]
                # Flatten (i.e. remove directory info)
                for m in members:
                    m.name = os.path.basename(m.name)
                # Extract
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(f, model_dir, members=members)

            # Remove tarfile
            logging.info(f"Deleting {model_tarfile_path}")
            os.remove(model_tarfile_path)

            logging.info(f"Processing {model_filename} complete")
            logging.info("---")

        else:
            logging.info(f"Skipping {model_name} as this model is already downloaded.")
