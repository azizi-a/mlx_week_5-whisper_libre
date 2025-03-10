import torch
import datasets
import whisper
import numpy as np


class LibreSpeechDataset(torch.utils.data.Dataset):
  def __init__(self, split, streaming, data_dir="ahazeemi/librispeech10h"):
    super().__init__()
    self.split = split
    self.streaming = streaming
    self.dataset = datasets.load_dataset(
      data_dir,
      split=split,
      streaming=streaming,
    )
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def __len__(self):
    if self.streaming:
      return 100000  # Return a large number since streaming datasets are effectively infinite
    return len(self.dataset)

  def __getitem__(self, idx):
    if self.streaming:
      # Convert idx to an iterator if streaming
      data = next(iter(self.dataset.skip(idx).take(1)))
    else:
      data = self.dataset[idx]

    audio = data["audio"]["array"]
    audio = whisper.pad_or_trim(audio)
    audio = np.array(audio, dtype=np.float32)
    mel = whisper.log_mel_spectrogram(audio)
    text = data["text"].capitalize()
    return audio, mel, text

  def collate(self, batch):
    audios = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    mels = [item[1] for item in batch]
    texts = [item[2] for item in batch]

    audios = torch.stack(audios, dim=0)
    mels = torch.stack(mels, dim=0)

    return audios, mels, texts

  def get_dataset(self):
    return self.dataset


if __name__ == "__main__":
  ds = LibreSpeechDataset(split="train.10", streaming=True)
  # print(ds.dataset.keys())
  print(len(ds))
  print(ds[0])
