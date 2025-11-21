import argparse
import pickle
import torch

from pretrain import InceptionTime, Trainer, Logger, Augmenter, Dataset, normalize, zero_pad

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger()
    augmenter = Augmenter()
    args = parse_args()

    with open("pretrain/dataset/cmr.pkl", "rb") as f:
        data = pickle.load(f)
    normalized_data = normalize(data)
    padded_data = zero_pad(normalized_data)

    data_loader = torch.utils.data.DataLoader(Dataset(padded_data), batch_size=args.batch_size, shuffle=True)
    model = InceptionTime(embed_dim=args.embed_dim, out_dim=args.out_dim, depth=args.depth).to(device)
    trainer = Trainer(model, augmenter, logger, device)

    trainer(data_loader, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)

def parse_args():
    parser = argparse.ArgumentParser(description="representation learning for carnatic music transcription")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--embed-dim', type=int, default=30)
    parser.add_argument('--out-dim', type=int, default=16)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
