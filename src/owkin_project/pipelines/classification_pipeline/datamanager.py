from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class MILDataset(Dataset):
    def __init__(self, X, y = None, n_instances = 1000, scaler = None):
        n_bags, n_features = len(X)//n_instances, X.shape[-1]
        self.n_instances = n_instances
        self.scaler = scaler
        if scaler is not None:
            self.X = scaler.transform(X)
        self.X = X.reshape(n_bags, n_instances, n_features) # create n bags of 1000 instances of dim = n_features
        self.y = y # managed in __getitem__
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx,:,:]
        target = None
        if self.y is not None:
            target = float(self.y[idx*self.n_instances])
        return features, target 

def get_dataset(X, y = None, n_instances: int = 1000, scaler = None) -> MILDataset:   
    dataset = MILDataset(X, y, n_instances, scaler)
    return dataset

def get_dataloader(dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
