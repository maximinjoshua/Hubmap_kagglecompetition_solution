import subprocess
import os
import zipfile


os.chdir(os.path.dirname(__file__))
subprocess.run(['kaggle', 'competitions', 'download', 'hubmap-organ-segmentation'])

with zipfile.ZipFile('hubmap-organ-segmentation.zip', 'r') as zip_ref:
    zip_ref.extractall('hubmap-organ-segmentation/')