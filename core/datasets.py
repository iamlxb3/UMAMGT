import ipdb
import torch
import collections


class StoryTuringTestData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    @property
    def size(self):
        assert len(self.encodings['input_ids']) == len(self.labels)
        return len(self.labels)

    @property
    def label(self):
        return collections.Counter(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
