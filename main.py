import torch
import whisper
from data.libre_speech import LibreSpeechDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the dataset
ds_train = LibreSpeechDataset(split="train.10", streaming=True)
ds_test = LibreSpeechDataset(split="test", streaming=True)

data_loader_train = DataLoader(ds_train, batch_size=16, collate_fn=ds_train.collate, num_workers=1)
data_loader_test = DataLoader(ds_test, batch_size=16, collate_fn=ds_test.collate, num_workers=1)

# Load the model
model = whisper.load_model("tiny.en")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


#
#
#
def evaluate_model(model, data_loader_test):
  model.eval()

  tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
  start_token = tokenizer.sot
  pad_token = tokenizer.eot

  accuracy = []

  for _, mel, text in tqdm(data_loader_test, desc="Evaluating"):
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
    remove_sot = input_tks[:, 1:].to(device)
    predictions = predictions[:, :-1, :]

    # Calculate accuracy
    pred_tokens = predictions.argmax(dim=-1)
    correct = (pred_tokens == remove_sot).sum().item()
    total = remove_sot.numel()
    accuracy.append(correct / total)
  average_accuracy = sum(accuracy) / len(accuracy)
  print(f"Average accuracy: {average_accuracy * 100:.2f}%")


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
    mel = mel.to(device)
    target_ids = [tokenizer.encode(t) for t in text]
    target_ids = [torch.tensor(ids, dtype=torch.long, device=device) for ids in target_ids]
    target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_token).to(device)
    start_token_tensor = torch.tensor([start_token], dtype=torch.long, device=device).repeat(len(text), 1)
    input_tks = torch.cat([start_token_tensor, target_ids], dim=1)

    # Forward pass
    predictions = model(tokens=input_tks, mel=mel)
    remove_sot = input_tks[:, 1:].to(device)
    predictions = predictions[:, :-1, :]
    loss = criterion(predictions.transpose(1, 2), remove_sot)

    # Backward pass
    optimizer.zero_grad()
    loss = criterion(predictions.transpose(1, 2), target_ids)
    loss.backward()
    optimizer.step()

  print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


#
#
#
evaluate_model(model, data_loader_test)
for epoch in range(8):
  train_model(model, data_loader_train, epoch)
evaluate_model(model, data_loader_test)
