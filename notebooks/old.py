       
    # params:
    n_features = 2048 #catalog.load('params:umap.umap_kwargs.n_components')
    lr = catalog.load('params:learning_rate')
    train_batch_size = catalog.load('params:train_batch_size')
    val_batch_size = catalog.load('params:val_batch_size')
    batch_size = catalog.load('params:batch_size')
    positive_weight = 1/y_train.mean()
    low_epoch_lr = 2
    milestones=[low_epoch_lr, 10, 50, 100, 500]
    gamma_start = 2
    gamma = 0.9
    epochs = 20
    # %%
    n_train, n_test = 50, 500
    X_train = shuffleX(X_train, 1000)
    train_data = get_MIL_X(X_train, y_train, n=n_train)
    eval_data = get_MIL_X(X_eval, y_eval, n=n_test)
    train_loader, eval_loader = get_data_loaders(train_data, eval_data, batch_size, val_batch_size)
    
   
    # %%
    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        if epoch < low_epoch_lr:
            lr = lr * gamma_start
        if epoch in milestones:
            lr = gamma * lr
        #set_lr(lr)
        lr = optimizer.param_groups[0]['lr']
            
        total_loss = 0

        # progress bar (works in Jupyter notebook too!)
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

        # ----------------- TRAINING  -------------------- 
        # set model to training
        model.train()
        for i, data in progress: 
            X, y = data[0].to(device), data[1].to(device)
            y = y.type(torch.FloatTensor)
            y = y.reshape(-1,1)
            
            # training step for single batch
            model.zero_grad() # to make sure that all the grads are 0 (same than optimizer.zero_grad)
            
            outputs = model(X) # forward
            loss = loss_function(outputs, y) # get loss
            loss.backward() # accumulates the gradient (by addition) for each parameter.
            optimizer.step() # performs a parameter update based on the current gradient 

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            # updating progress bar
            progress.set_description("Loss: {:.4f}, learning_rate : {}".format(total_loss/(i+1), lr))
            
        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # ----------------- VALIDATION  ----------------- 
        val_losses = 0
        y_pred, y_true = [], []
        precision, recall, f1, accuracy = [], [], [], []
        
        # set model to evaluating (testing)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                X, y = data[0].to(device), data[1].to(device)
                y = y.reshape(-1,1)
                outputs = model(X) 
                prediced_classes = outputs.detach().round()
                val_losses += loss_function(outputs, y)
                y_pred.extend(prediced_classes.reshape(-1).tolist())
                y_true.extend(y.reshape(-1).tolist())
                
                # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1, accuracy), 
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    metric(y_true, y_pred)
                )
                
        print(f"Epoch {epoch+1}/{epochs}, lr {lr:.7f}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
        print_scores(precision[0], recall[0], f1[0], accuracy[0])
        losses.append(total_loss/batches) # for plotting learning curve
    print(f"Training time: {time.time()-start_ts}s")
    