import training
import torch

model_file_name = 'my_model.pth'

labels_dict = {}

model = None
device = None

def preparing_stuff():
    global device
    global labels_dict
    global model
    
    labels_dict = training.get_labels_dict()
    model = training.load_model(model_file_name, num_categories=len(labels_dict))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    print(f"Labels: {labels_dict}")


def get_answer(data):
    data = data / data.max() # Normalize to 0-1
    
    # Convert to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    
    data_tensor = data_tensor.unsqueeze(0) # Turns the shape [28, 28] -> [1, 28, 28]
    data_tensor = data_tensor.unsqueeze(0) # Turns the shape [1, 28, 28] -> [1, 1, 28, 28] torch model needs to format to be this way
    prediction = model(data_tensor)

    class_prob = torch.softmax(prediction, dim=1).tolist()

    results = []
    for i, prob in enumerate(class_prob[0]):
        results.append(f"{labels_dict[i]}: {prob*100:.2f}%")
    
    return results