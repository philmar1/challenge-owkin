
class myDataset(Dataset):
    def __init__(self, X, y = None, scaler = None):
        self.X = X
        self.y = y
        self.scaler = scaler
        if scaler is not None:
            self.X = scaler.transform(self.X)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx,:]
        target = None
        if self.y is not None:
            target = self.y[idx]
        
        return features, target 
    
hyperarameters = {'lr': 0.00001,
                  'epochs': 50,
                  'positive_weight': 1,
                  'gamma': 0.95}
emb_data_train = myDataset(X_train, y_clustered_train, scaler)    
emb_data_eval = myDataset(X_eval, y_clustered_eval, scaler)    
emb_loader_train = DataLoader(emb_data_train, batch_size=128, shuffle=True)
emb_loader_eval = DataLoader(emb_data_eval, batch_size=64, shuffle=False)
    
embedder = NN_classifier(n=n_features, n_mid=1024, n_emb=512)
optimizer = optim.Adam(embedder.parameters(), lr=hyperarameters['lr'])
train(embedder, optimizer, emb_loader_train, emb_loader_eval, hyperarameters=hyperarameters)


if __name__ == '__main__':
    y_clustered_train = catalog.load('y_clustered_train')
    y_clustered_eval = catalog.load('y_clustered_eval')