### 0.9.0
- add per-layer activation, initialization, regularization, dropout
- add builder starting with NN::new_input(usize)
- add learning rate scheduling
- rename some enums and methods

### 0.8.0
- add dropout
- fix report classification for binary crossentropy
- use BCE for moons example
- move gradient calc into Loss; allow deserialization from different case; tidy
- change Adam epsilon
- show digit in terminal for MNIST 

### 0.7.0
- change `run_and_report` to `train`, and add to `NN`
- add `report` to `NN`
- add more tests
- fix examples and tests
- cleanup

### 0.6.0
- changed pass in borrowed data to add_data for dataset
- added 2 moons example
- added `with_softmax_and_crossentropy` this can significantly speed up learning on categorical data
- fix get_test_data_zip to return test data not train data
- can now add same column multiple times
- add ability to apply function to column before conversion
- added `OneHotTop` to only take top N observations of a categorical feature
- fixed bug where report was using training data insteead of test data
- added RSquared report metric
- added custom report metric
- renamed `fit` to `fit_batch` and `fit_batch_size` to `fit`
### 0.5.0
- significant speedup with more matrix multiplications
- add optional accuracy to `run_and_report` for categorisation tasks
- added allocate fixed range to test data for dataset `allocate_range_to_test_data`
- made `learning_rate` pub so it can be changed 

### 0.4.0
- fixed save/load not working for larger networks
- fixed save/load not saving regularization
- added `add_input_column_range` to make it easy to input a large number of columns at once
- added convenience function to run network for number of epochs, and report `run_and_report`

### 0.3.0
- added Dataset manager enabling:
    - one hot encoding (this enables classification)
    - normalisation
    - quickly read from csv
- added iris example
- added some utility functions:
    - `forward_errors` to forward an entire batch inputs to get the MSE
    - `max_index_equal` used with one hot encoding to calc the accuracy

### 0.2.0
- added regularization L1, L2, L1 and L2
- added examples:
    - mnist 
    - square (demos regularization)
