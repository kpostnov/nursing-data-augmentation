# test_report
we compare epochs



|   | Model "epoch 2" | Model "epoch 1" |
|-------------- | -------------- | -------------- | 
| correct_classification_acc | 0.63 | 0.48 | 
| avg_failure_rate | 0.45 | 0.62 | 

### Model "epoch 2"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.11 | 0.82 | {'running': 55, 'squats': 16, 'stairs_down': 13, 'stairs_up': 21, 'standing': 114, 'walking': 346} | 
| 1 | 0.5 | 0.63 | {'running': 11, 'squats': 4, 'stairs_down': 34, 'stairs_up': 32, 'standing': 30, 'walking': 72} | 
| 2 | 0.7 | 0.31 | {'running': 23, 'squats': 8, 'stairs_down': 41, 'stairs_up': 29, 'standing': 98, 'walking': 358} | 
| 3 | 0.94 | 0.22 | {'running': 12, 'squats': 20, 'stairs_down': 17, 'stairs_up': 32, 'standing': 230, 'walking': 139} | 
| 4 | 0.88 | 0.27 | {'running': 36, 'squats': 41, 'stairs_down': 18, 'stairs_up': 34, 'standing': 29, 'walking': 228} | 
|  |  |  |  | 
| min | 0.11 | 0.22 | - | 
| max | 0.94 | 0.82 | - | 
| mean | 0.63 | 0.45 | - | 
| median | 0.7 | 0.31 | - | 

### Model "epoch 1"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.1 | 0.79 | {'running': 55, 'squats': 16, 'stairs_down': 13, 'stairs_up': 21, 'standing': 114, 'walking': 346} | 
| 1 | 0.45 | 0.72 | {'running': 11, 'squats': 4, 'stairs_down': 34, 'stairs_up': 32, 'standing': 30, 'walking': 72} | 
| 2 | 0.68 | 0.51 | {'running': 23, 'squats': 8, 'stairs_down': 41, 'stairs_up': 29, 'standing': 98, 'walking': 358} | 
| 3 | 0.36 | 0.69 | {'running': 12, 'squats': 20, 'stairs_down': 17, 'stairs_up': 32, 'standing': 230, 'walking': 139} | 
| 4 | 0.82 | 0.37 | {'running': 36, 'squats': 41, 'stairs_down': 18, 'stairs_up': 34, 'standing': 29, 'walking': 228} | 
|  |  |  |  | 
| min | 0.1 | 0.37 | - | 
| max | 0.82 | 0.79 | - | 
| mean | 0.48 | 0.62 | - | 
| median | 0.45 | 0.69 | - | 

