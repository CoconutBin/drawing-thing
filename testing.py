import training
import cv2
import matplotlib.pyplot as plt

import torch

model_file_name = 'my_model.pth'

labels_dict = {
    0: 'cake',
    1: 'cat',
    2: 'tornado',
}
_,_,_,_, labels_dict = training.prepare_dataset_and_labels(batch_size=64)
model = training.load_model(model_file_name, num_categories=len(labels_dict))

print(labels_dict)
# test = cv2.imread('test.png', 0)
# test = test.reshape([28, 28])

# _,ax = plt.subplots(1, 2)
# ax[0].imshow(test)
# ax[1].imshow(1 - ((test - test.min()) / (test.max() - test.min() + 1e-12)))
# plt.show()

def get_answer(data):
    # Convert to tensor
    data_tensor = torch.tensor(data)
    data_tensor = 1 - ((data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min() + 1e-12))
    data_tensor = torch.clamp(data_tensor, 0, 1)
    
    data_tensor = data_tensor.unsqueeze(0) # Turns the shape [28, 28] -> [1, 28, 28] torch model needs to format to be this way
    print(data_tensor.shape)
    prediction = model(data_tensor)

    print(prediction)
    class_prob = torch.softmax(prediction, dim=1).tolist()
    print(class_prob)
    # print(data_tensor)
    
    # plt.imshow(data_tensor.reshape([28, 28]))
    # plt.show()
    results = []
    for i, prob in enumerate(class_prob[0]):
        print(labels_dict[i])
        print(prob)
        results.append(f"{labels_dict[i]}: {prob*100:.2f}%")
    
    return results

# img = cv2.imread("./test.png", 0)
# data = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
# data = cv2.bitwise_not(data)
# print(get_answer(data))