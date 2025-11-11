import os
import training

preview_dataset = False
batch_size = 64
learning_rate = 1e-3
momentum = 0.9
num_epochs = 16
model_file_name = 'my_model.pth'


train_dataset, val_dataset, train_dataloader, val_dataloader, labels_dict = training.prepare_dataset_and_labels(batch_size=batch_size)

if not os.path.isfile(model_file_name): # No saved
    model = training.make_new_model(num_categories=len(labels_dict))
    
    # training.random_preview_train_dataset(train_dataset, labels_dict)
    training.random_preview_model(model, val_dataset, labels_dict, title="Before Train")
    
    # Training
    results = training.train_and_save_model(model, num_epochs, learning_rate, momentum, batch_size, model_file_name,
                                         train_dataloader, val_dataloader, show_graph=True)
    
    training.random_preview_model(model, val_dataset, labels_dict, title="After Train")
else:  
    model = training.load_model(model_file_name, num_categories=len(labels_dict))
    
    training.random_preview_model(model, val_dataset, labels_dict, title="After Train")