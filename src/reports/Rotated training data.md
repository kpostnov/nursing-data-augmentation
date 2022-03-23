# Rotated training data
Lol, hallo @orhan_konak



|   | Model "standard_recordings # {'epochs': 1}" | Model "standard_recordings # {'epochs': 2}" | Model "euler_recordings # {'epochs': 1}" | Model "euler_recordings # {'epochs': 2}" | Model "rotated_recordings # {'epochs': 1}" | Model "rotated_recordings # {'epochs': 2}" |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| correct_classification_acc | 0.11 | 0.34 | 0.28 | 0.24 | 0.81 | 0.9 | 
| avg_failure_rate | 0.83 | 0.68 | 0.75 | 0.76 | 0.27 | 0.14 | 

### Model "standard_recordings | {'epochs': 1}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.04 | 0.9 | {'running': 57, 'squats': 37, 'stairs_down': 89, 'stairs_up': 58, 'standing': 344, 'walking': 831} | 
| 1 | 0.18 | 0.76 | {'running': 80, 'squats': 52, 'stairs_down': 34, 'stairs_up': 90, 'standing': 157, 'walking': 312} | 
|  |  |  |  | 
| min | 0.04 | 0.76 | - | 
| max | 0.18 | 0.9 | - | 
| mean | 0.11 | 0.83 | - | 
| median | 0.11 | 0.8300000000000001 | - | 

### Model "standard_recordings | {'epochs': 2}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.04 | 0.86 | {'running': 57, 'squats': 37, 'stairs_down': 89, 'stairs_up': 58, 'standing': 344, 'walking': 831} | 
| 1 | 0.65 | 0.5 | {'running': 80, 'squats': 52, 'stairs_down': 34, 'stairs_up': 90, 'standing': 157, 'walking': 312} | 
|  |  |  |  | 
| min | 0.04 | 0.5 | - | 
| max | 0.65 | 0.86 | - | 
| mean | 0.34 | 0.68 | - | 
| median | 0.34500000000000003 | 0.6799999999999999 | - | 

### Model "euler_recordings | {'epochs': 1}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 30, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.44 | 0.65 | {'running': 57, 'squats': 37, 'stairs_down': 89, 'stairs_up': 58, 'standing': 344, 'walking': 831} | 
| 1 | 0.12 | 0.86 | {'running': 80, 'squats': 52, 'stairs_down': 34, 'stairs_up': 90, 'standing': 157, 'walking': 312} | 
|  |  |  |  | 
| min | 0.12 | 0.65 | - | 
| max | 0.44 | 0.86 | - | 
| mean | 0.28 | 0.76 | - | 
| median | 0.28 | 0.755 | - | 

### Model "euler_recordings | {'epochs': 2}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 30, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.32 | 0.73 | {'running': 57, 'squats': 37, 'stairs_down': 89, 'stairs_up': 58, 'standing': 344, 'walking': 831} | 
| 1 | 0.17 | 0.8 | {'running': 80, 'squats': 52, 'stairs_down': 34, 'stairs_up': 90, 'standing': 157, 'walking': 312} | 
|  |  |  |  | 
| min | 0.17 | 0.73 | - | 
| max | 0.32 | 0.8 | - | 
| mean | 0.24 | 0.76 | - | 
| median | 0.245 | 0.765 | - | 

### Model "rotated_recordings | {'epochs': 1}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 30, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.7 | 0.41 | {'running': 411, 'squats': 267, 'stairs_down': 369, 'stairs_up': 444, 'standing': 1503, 'walking': 3429} | 
| 1 | 0.92 | 0.14 | {'running': 411, 'squats': 267, 'stairs_down': 369, 'stairs_up': 444, 'standing': 1503, 'walking': 3429} | 
|  |  |  |  | 
| min | 0.7 | 0.14 | - | 
| max | 0.92 | 0.41 | - | 
| mean | 0.81 | 0.28 | - | 
| median | 0.81 | 0.275 | - | 

### Model "rotated_recordings | {'epochs': 2}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 30, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.85 | 0.2 | {'running': 411, 'squats': 267, 'stairs_down': 369, 'stairs_up': 444, 'standing': 1503, 'walking': 3429} | 
| 1 | 0.96 | 0.08 | {'running': 411, 'squats': 267, 'stairs_down': 369, 'stairs_up': 444, 'standing': 1503, 'walking': 3429} | 
|  |  |  |  | 
| min | 0.85 | 0.08 | - | 
| max | 0.96 | 0.2 | - | 
| mean | 0.9 | 0.14 | - | 
| median | 0.905 | 0.14 | - | 

