import modal
import pandas as pd 
import torchaudio 
from torch.utils.data import DataLoader
app = modal.App("audio-classification")

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")
image.apt_install(["ffmpeg", "wget","libsndfile1", "uzip"]).run_commands(["cd/tmp && wget https://github.com/karlopiczak/ESC-50/archive/master.zip -0 esc50.zip",
                                         "cd /tmp && unzip esc50.zip",
                                        "mkdir -p /opt/ESC-50-data",
                                        "cp -r /tmp/ESC-50-master/* /opt/ESC-50-data/" ,"rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
                                        ]).add_local_python_source("model")

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
modal_volume = modal.Volume.from_name("esc50-data", create_if_missing=True)


class ESC50Dataset(Dataset):
    def __init__(self, data_path, metadata_file ,split='train',transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata_file = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split=='train':
            self.metadata = self.metadata[self.metadata_file['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata_file['fold'] == 5]
        self.classes=sorted(self.metadata_file['category'].unique())
        self.class_to_idx={cls_name:i for i,cls_name in enumerate(self.classes)}
        self.metadata['label']=self.metadata_file['category'].map(self.class_to_idx) 

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio"/ row['filename']
        label = row['label']
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0]>1:
            waveform=torch.mean(waveform,dim=0,keepdim=True)
    


@app.function(image=image , gpu="T4" , volumes={"/data": volume,"/models":modal_volume},timeout=3600)
def train():
    print("Training started...")


@app.local_entrypoint()
def main():
    
    print("Invoking train() locally for quick test...")
    train.remote()
    print("Finished.")
