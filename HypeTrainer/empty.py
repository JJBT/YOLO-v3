from torch.utils.data import Dataset


class EmptyDataset(Dataset):
    def __getitem__(self, item):
        raise StopIteration

    def __len__(self):
        return 0
