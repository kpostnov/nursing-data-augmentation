# TestTheReport
How does data augmentation improve the model?



|   | Model "without augmentation @ {'epochs': 1}" |
|-------------- | -------------- | 
| correct_classification_acc | 0.21 | 
| avg_failure_rate | 0.81 | 

### Model "without augmentation @ {'epochs': 1}"

{'window_size': 180, 'stride_size': 180, 'test_percentage': 0.2, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.74 | 0.66 | {'running': 74, 'squats': 15, 'stairs_down': 21, 'stairs_up': 29, 'standing': 46, 'walking': 235} | 
| 1 | 0.21 | 0.84 | {'running': 11, 'squats': 42, 'stairs_down': 26, 'stairs_up': 28, 'standing': 42, 'walking': 37} | 
| 2 | 0.07 | 0.76 | {'running': 29, 'squats': 17, 'stairs_down': 38, 'stairs_up': 38, 'standing': 13, 'walking': 296} | 
| 3 | 0.03 | 0.91 | {'running': 13, 'squats': 2, 'stairs_down': 26, 'stairs_up': 7, 'standing': 378, 'walking': 92} | 
| 4 | 0.02 | 0.88 | {'running': 10, 'squats': 13, 'stairs_down': 12, 'stairs_up': 46, 'standing': 22, 'walking': 483} | 
|  |  |  |  | 
| min | 0.02 | 0.66 | - | 
| max | 0.74 | 0.91 | - | 
| mean | 0.21 | 0.81 | - | 
| median | 0.07 | 0.84 | - | 

