import training
import cv2
import matplotlib.pyplot as plt

import torch

model_file_name = 'my_model.pth'

_,_,_,_, labels_dict = training.prepare_dataset_and_labels(batch_size=64)
model = training.load_model(model_file_name, num_categories=len(labels_dict))

img = cv2.imread("./data/testing/test.png", 0)
data = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
data = cv2.bitwise_not(data)

# Convert to tensor
data_tensor = torch.tensor(data)
data_tensor = data_tensor / 255 # abs((data_tensor / 255) - 1) # Normalize from 0-255 to 1 (black) - 0 (white)
data_tensor = torch.flatten(data_tensor)


prediction = model(data_tensor)
answer_list = prediction.tolist()
answer = torch.argmax(prediction).item()

class_prob = torch.softmax(prediction, dim=0).tolist()

for i, prob in enumerate(class_prob):
    print(f"{labels_dict[i]}: {prob*100:.2f}%")

print(f"\nFinal Answer: {labels_dict[answer]}")

plt.imshow(data)
plt.show()