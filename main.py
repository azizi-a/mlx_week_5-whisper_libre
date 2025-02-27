import torch
import whisper
import editdistance
from data.libre_speech import LibreSpeechDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the dataset
ds_train = LibreSpeechDataset(split="train.10", streaming=False)
ds_test = LibreSpeechDataset(split="test", streaming=False)

data_loader_train = DataLoader(ds_train, batch_size=16, collate_fn=ds_train.collate, num_workers=1)
data_loader_test = DataLoader(ds_test, batch_size=16, collate_fn=ds_test.collate, num_workers=1)

# Load the model
model = whisper.load_model("tiny.en")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


#
#
#
def forward_pass(model, text, mel, tokenizer, start_token, pad_token):
  mel = mel.to(device)
  # Encode all texts in batch
  target_ids = [tokenizer.encode(t) for t in text]
  # Convert to padded tensor in using pad_sequence
  target_ids = [torch.tensor(ids, dtype=torch.long, device=device) for ids in target_ids]
  target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_token).to(device)
  # Create the input tokens
  start_token_tensor = torch.tensor([start_token], dtype=torch.long, device=device).repeat(len(text), 1)
  input_tks = torch.cat([start_token_tensor, target_ids], dim=1)

  # Forward pass
  predictions = model(tokens=input_tks, mel=mel)
  input_removed_sot = input_tks[:, 1:].to(device)
  predictions = predictions[:, :-1, :]

  return predictions, input_removed_sot


#
#
#
def evaluate_model(model, data_loader_test):
  model.eval()

  tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
  start_token = tokenizer.sot
  pad_token = tokenizer.eot

  wers = []

  for _, mel, text in tqdm(data_loader_test, desc="Evaluating"):
    predictions, _ = forward_pass(model, text, mel, tokenizer, start_token, pad_token)

    # Get predicted tokens and convert to text
    predicted_tokens = predictions.argmax(dim=-1)
    predicted_texts = [tokenizer.decode(t) for t in predicted_tokens]

    # Calculate word error rate
    for prd_text, target_text in zip(predicted_texts, text):
      # Split into words
      prd_words = prd_text.split()
      target_words = target_text.split()

      # Calculate Levenshtein distance
      distance = editdistance.eval(prd_words, target_words)

      # Calculate WER
      if len(target_words) > 0:
        wer = distance / len(target_words)
        wers.append(wer)

  average_wer = sum(wers) / len(wers)
  print(f"Average WER: {average_wer * 100:.2f}%")


#
#
#
def train_model(model, data_loader_train, epoch=0):
  model.train()

  tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
  start_token = tokenizer.sot
  pad_token = tokenizer.eot

  # Define the optimizer and criterion
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
  criterion = torch.nn.CrossEntropyLoss()

  for _, mel, text in tqdm(data_loader_train, desc="Training epoch"):
    predictions, input_removed_sot = forward_pass(model, text, mel, tokenizer, start_token, pad_token)

    # Backward pass
    optimizer.zero_grad()
    loss = criterion(predictions.transpose(1, 2), input_removed_sot)
    loss.backward()
    optimizer.step()

  print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


#
#
#
evaluate_model(model, data_loader_test)
for epoch in range(3):
  train_model(model, data_loader_train, epoch)
evaluate_model(model, data_loader_test)
