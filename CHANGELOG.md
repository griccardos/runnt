### 0.5.1
- changed pass in borrowed data to add_data for dataset
- added 2 moons example

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