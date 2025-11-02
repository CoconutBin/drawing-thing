import training
import cv2
import matplotlib.pyplot as plt

import torch

model_file_name = 'my_model.pth'

_,_,_,_, labels_dict = training.prepare_dataset_and_labels(batch_size=64)
model = training.load_model(model_file_name, num_categories=len(labels_dict))


def get_answer(data):
    # Convert to tensor
    data_tensor = torch.tensor(data)
    # data_tensor = abs((data_tensor / 253) - 1) # Normalize from 0-255 to 1 (black) - 0 (white)
    data_tensor = 1 - ((data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min() + 1e-12))
    data_tensor = data_tensor * 2
    data_tensor = torch.clamp(data_tensor, 0, 1)
    
    # data_tensor = torch.flatten(data_tensor) # CNN input is 2D array no need to flat
    
    # prediction = model(data_tensor)
    prediction = model(data_tensor)[0]
    
    
    class_prob = torch.softmax(prediction, dim=0).tolist()
    # print(data_tensor)
    
    # plt.imshow(data_tensor.reshape([28, 28]))
    # plt.show()
    
    results = []
    for i, prob in enumerate(class_prob):
        results.append(f"{labels_dict[i]}: {prob*100:.2f}%")
    
    # print(data_tensor)
    return results

# img = cv2.imread("./test.png", 0)
# data = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
# data = cv2.bitwise_not(data)
# print(get_answer(data))