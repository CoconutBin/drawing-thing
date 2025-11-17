import training

training.save_current_categories_to_labels_txt()

with open('labels.txt', 'r') as f:
    print(f.readlines().split())