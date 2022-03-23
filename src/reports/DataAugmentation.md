# DataAugmentation
How does data augmentation improve the model?



|   | Model "without augmentation @ {'epochs': 15}" | Model "augmentation 5 @ {'epochs': 15}" | Model "augmentation 2 @ {'epochs': 15}" |
|-------------- | -------------- | -------------- | -------------- | 
| correct_classification_acc | 0.78 | 0.95 | 0.87 | 
| avg_failure_rate | 0.24 | 0.05 | 0.14 | 

### Model "without augmentation @ {'epochs': 15}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.96 | 0.05 | {'running': 74, 'squats': 15, 'stairs_down': 21, 'stairs_up': 29, 'standing': 46, 'walking': 235} | 
| 1 | 0.2 | 0.81 | {'running': 11, 'squats': 42, 'stairs_down': 26, 'stairs_up': 28, 'standing': 42, 'walking': 37} | 
| 2 | 0.94 | 0.1 | {'running': 29, 'squats': 17, 'stairs_down': 38, 'stairs_up': 38, 'standing': 13, 'walking': 296} | 
| 3 | 0.94 | 0.08 | {'running': 13, 'squats': 2, 'stairs_down': 26, 'stairs_up': 7, 'standing': 378, 'walking': 92} | 
| 4 | 0.88 | 0.14 | {'running': 10, 'squats': 13, 'stairs_down': 12, 'stairs_up': 46, 'standing': 22, 'walking': 483} | 
|  |  |  |  | 
| min | 0.2 | 0.05 | - | 
| max | 0.96 | 0.81 | - | 
| mean | 0.78 | 0.24 | - | 
| median | 0.94 | 0.1 | - | 

### Model "augmentation 5 @ {'epochs': 15}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.96 | 0.05 | {'running': 74, 'squats': 15, 'stairs_down': 21, 'stairs_up': 29, 'standing': 46, 'walking': 235} | 
| 1 | 0.87 | 0.13 | {'running': 11, 'squats': 42, 'stairs_down': 26, 'stairs_up': 28, 'standing': 42, 'walking': 37} | 
| 2 | 0.96 | 0.05 | {'running': 29, 'squats': 17, 'stairs_down': 38, 'stairs_up': 38, 'standing': 13, 'walking': 296} | 
| 3 | 1.0 | 0.0 | {'running': 13, 'squats': 2, 'stairs_down': 26, 'stairs_up': 7, 'standing': 378, 'walking': 92} | 
| 4 | 0.99 | 0.02 | {'running': 10, 'squats': 13, 'stairs_down': 12, 'stairs_up': 46, 'standing': 22, 'walking': 483} | 
|  |  |  |  | 
| min | 0.87 | 0.0 | - | 
| max | 1.0 | 0.13 | - | 
| mean | 0.96 | 0.05 | - | 
| median | 0.96 | 0.05 | - | 

### Model "augmentation 2 @ {'epochs': 15}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.97 | 0.03 | {'running': 74, 'squats': 15, 'stairs_down': 21, 'stairs_up': 29, 'standing': 46, 'walking': 235} | 
| 1 | 0.42 | 0.59 | {'running': 11, 'squats': 42, 'stairs_down': 26, 'stairs_up': 28, 'standing': 42, 'walking': 37} | 
| 2 | 0.96 | 0.05 | {'running': 29, 'squats': 17, 'stairs_down': 38, 'stairs_up': 38, 'standing': 13, 'walking': 296} | 
| 3 | 1.0 | 0.01 | {'running': 13, 'squats': 2, 'stairs_down': 26, 'stairs_up': 7, 'standing': 378, 'walking': 92} | 
| 4 | 1.0 | 0.01 | {'running': 10, 'squats': 13, 'stairs_down': 12, 'stairs_up': 46, 'standing': 22, 'walking': 483} | 
|  |  |  |  | 
| min | 0.42 | 0.01 | - | 
| max | 1.0 | 0.59 | - | 
| mean | 0.87 | 0.14 | - | 
| median | 0.97 | 0.03 | - | 

