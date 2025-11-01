import modal
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
    def __init__(self, data_path, metadata_file):
        super.__init__()

@app.function(image=image , gpu="T4" , volumes={"/data": volume,"/models":modal_volume},timeout=3600)
def train():
    print("Training started...")


@app.local_entrypoint()
def main():
    
    print("Invoking train() locally for quick test...")
    train.remote()
    print("Finished.")
