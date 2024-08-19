from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import numpy as np


# We prepare the data:
x_train, y_train = read_hoda_dataset(dataset_path='Train 60000.cdb', images_height=32, images_width=32, one_hot=False, reshape=False)
x_validation, y_validation = read_hoda_dataset('RemainingSamples.cdb', images_height=32, images_width=32, one_hot=False, reshape=False)
x_test, y_test = read_hoda_dataset(dataset_path='Test 20000.cdb', images_height=32, images_width=32, one_hot=False, reshape=False)

print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)
print(x_test.shape)
print(y_test.shape)

x_train = x_train.flatten()
print(x_train)
print(len(x_train))

x_validation = x_validation.flatten()
print(x_validation)
print(len(x_validation))

x_test = x_test.flatten()
print(x_test)
print(len(x_test))

print(y_train)
print(len(y_train))

print(y_validation)
print(len(y_validation))

print(y_test)
print(len(y_test))

# We save our data into files so we can do classification on them elsewhere:
with open('x_train.npy', 'wb') as file:
    np.save(file, x_train, allow_pickle=True)

with open('y_train.npy', 'wb') as file:
    np.save(file, y_train, allow_pickle=True)

with open('x_validation.npy', 'wb') as file:
    np.save(file, x_validation, allow_pickle=True)

with open('y_validation.npy', 'wb') as file:
    np.save(file, y_validation, allow_pickle=True)

with open('x_test.npy', 'wb') as file:
    np.save(file, x_test, allow_pickle=True)

with open('y_test.npy', 'wb') as file:
    np.save(file, y_test, allow_pickle=True)
