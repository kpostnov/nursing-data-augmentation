# best_window_size
20 models: from window_size 30 up to window_size 430 (20ms steps)

- 100 epochs always
-window_size=stride_size



|   | Model "window_size 30" | Model "window_size 50" | Model "window_size 70" | Model "window_size 90" | Model "window_size 110" | Model "window_size 130" | Model "window_size 150" | Model "window_size 170" | Model "window_size 190" | Model "window_size 210" | Model "window_size 230" | Model "window_size 250" | Model "window_size 270" | Model "window_size 290" | Model "window_size 310" | Model "window_size 330" | Model "window_size 350" | Model "window_size 370" | Model "window_size 390" | Model "window_size 410" |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| correct_classification_acc | 0.96 | 0.8 | 0.97 | 0.95 | 0.81 | 0.98 | 0.98 | 0.97 | 0.97 | 0.96 | 0.87 | 0.88 | 0.94 | 0.97 | 0.98 | 0.96 | 0.91 | 0.88 | 0.93 | 0.83 | 
| avg_failure_rate | 0.04 | 0.21 | 0.03 | 0.06 | 0.2 | 0.02 | 0.03 | 0.03 | 0.03 | 0.04 | 0.13 | 0.13 | 0.07 | 0.04 | 0.03 | 0.04 | 0.09 | 0.12 | 0.08 | 0.17 | 

### Model "window_size 30"

{'window_size': 30, 'stride_size': 30, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.95 | 0.05 | {'running': 340, 'squats': 104, 'stairs_down': 81, 'stairs_up': 134, 'standing': 689, 'walking': 2082} | 
| 1 | 0.91 | 0.09 | {'running': 74, 'squats': 26, 'stairs_down': 219, 'stairs_up': 210, 'standing': 200, 'walking': 442} | 
| 2 | 0.97 | 0.03 | {'running': 145, 'squats': 51, 'stairs_down': 258, 'stairs_up': 178, 'standing': 594, 'walking': 2156} | 
| 3 | 0.99 | 0.01 | {'running': 75, 'squats': 121, 'stairs_down': 107, 'stairs_up': 199, 'standing': 1396, 'walking': 842} | 
| 4 | 0.99 | 0.01 | {'running': 221, 'squats': 250, 'stairs_down': 109, 'stairs_up': 217, 'standing': 183, 'walking': 1380} | 
|  |  |  |  | 
| min | 0.91 | 0.01 | - | 
| max | 0.99 | 0.09 | - | 
| mean | 0.96 | 0.04 | - | 
| median | 0.97 | 0.03 | - | 

### Model "window_size 50"

{'window_size': 50, 'stride_size': 50, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.77 | 0.24 | {'running': 203, 'squats': 62, 'stairs_down': 49, 'stairs_up': 80, 'standing': 413, 'walking': 1248} | 
| 1 | 0.93 | 0.07 | {'running': 45, 'squats': 15, 'stairs_down': 130, 'stairs_up': 125, 'standing': 120, 'walking': 264} | 
| 2 | 0.8 | 0.2 | {'running': 88, 'squats': 30, 'stairs_down': 152, 'stairs_up': 106, 'standing': 356, 'walking': 1293} | 
| 3 | 0.5 | 0.51 | {'running': 45, 'squats': 73, 'stairs_down': 63, 'stairs_up': 118, 'standing': 836, 'walking': 505} | 
| 4 | 0.99 | 0.02 | {'running': 132, 'squats': 150, 'stairs_down': 64, 'stairs_up': 128, 'standing': 109, 'walking': 827} | 
|  |  |  |  | 
| min | 0.5 | 0.02 | - | 
| max | 0.99 | 0.51 | - | 
| mean | 0.8 | 0.21 | - | 
| median | 0.8 | 0.2 | - | 

### Model "window_size 70"

{'window_size': 70, 'stride_size': 70, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.96 | 0.05 | {'running': 144, 'squats': 44, 'stairs_down': 35, 'stairs_up': 56, 'standing': 295, 'walking': 892} | 
| 1 | 0.96 | 0.04 | {'running': 32, 'squats': 11, 'stairs_down': 93, 'stairs_up': 88, 'standing': 85, 'walking': 188} | 
| 2 | 0.97 | 0.03 | {'running': 61, 'squats': 22, 'stairs_down': 109, 'stairs_up': 75, 'standing': 254, 'walking': 923} | 
| 3 | 0.99 | 0.01 | {'running': 31, 'squats': 52, 'stairs_down': 45, 'stairs_up': 84, 'standing': 597, 'walking': 360} | 
| 4 | 0.97 | 0.03 | {'running': 94, 'squats': 106, 'stairs_down': 46, 'stairs_up': 92, 'standing': 78, 'walking': 590} | 
|  |  |  |  | 
| min | 0.96 | 0.01 | - | 
| max | 0.99 | 0.05 | - | 
| mean | 0.97 | 0.03 | - | 
| median | 0.97 | 0.03 | - | 

### Model "window_size 90"

{'window_size': 90, 'stride_size': 90, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.91 | 0.12 | {'running': 112, 'squats': 33, 'stairs_down': 27, 'stairs_up': 44, 'standing': 229, 'walking': 693} | 
| 1 | 0.88 | 0.12 | {'running': 24, 'squats': 8, 'stairs_down': 72, 'stairs_up': 68, 'standing': 64, 'walking': 145} | 
| 2 | 0.97 | 0.04 | {'running': 47, 'squats': 16, 'stairs_down': 84, 'stairs_up': 58, 'standing': 196, 'walking': 717} | 
| 3 | 1.0 | 0.0 | {'running': 24, 'squats': 40, 'stairs_down': 35, 'stairs_up': 64, 'standing': 464, 'walking': 280} | 
| 4 | 0.98 | 0.02 | {'running': 73, 'squats': 82, 'stairs_down': 36, 'stairs_up': 71, 'standing': 60, 'walking': 458} | 
|  |  |  |  | 
| min | 0.88 | 0.0 | - | 
| max | 1.0 | 0.12 | - | 
| mean | 0.95 | 0.06 | - | 
| median | 0.97 | 0.04 | - | 

### Model "window_size 110"

{'window_size': 110, 'stride_size': 110, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.97 | 0.04 | {'running': 90, 'squats': 27, 'stairs_down': 21, 'stairs_up': 35, 'standing': 187, 'walking': 567} | 
| 1 | 0.92 | 0.09 | {'running': 19, 'squats': 7, 'stairs_down': 58, 'stairs_up': 56, 'standing': 54, 'walking': 119} | 
| 2 | 0.74 | 0.27 | {'running': 39, 'squats': 14, 'stairs_down': 68, 'stairs_up': 47, 'standing': 162, 'walking': 585} | 
| 3 | 0.42 | 0.58 | {'running': 19, 'squats': 33, 'stairs_down': 28, 'stairs_up': 54, 'standing': 379, 'walking': 228} | 
| 4 | 0.99 | 0.01 | {'running': 60, 'squats': 66, 'stairs_down': 28, 'stairs_up': 57, 'standing': 49, 'walking': 375} | 
|  |  |  |  | 
| min | 0.42 | 0.01 | - | 
| max | 0.99 | 0.58 | - | 
| mean | 0.81 | 0.2 | - | 
| median | 0.92 | 0.09 | - | 

### Model "window_size 130"

{'window_size': 130, 'stride_size': 130, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.98 | 0.04 | {'running': 76, 'squats': 22, 'stairs_down': 19, 'stairs_up': 29, 'standing': 158, 'walking': 480} | 
| 1 | 0.99 | 0.01 | {'running': 16, 'squats': 6, 'stairs_down': 48, 'stairs_up': 46, 'standing': 44, 'walking': 100} | 
| 2 | 0.98 | 0.02 | {'running': 32, 'squats': 11, 'stairs_down': 57, 'stairs_up': 39, 'standing': 136, 'walking': 494} | 
| 3 | 1.0 | 0.0 | {'running': 17, 'squats': 28, 'stairs_down': 24, 'stairs_up': 44, 'standing': 320, 'walking': 193} | 
| 4 | 0.97 | 0.03 | {'running': 50, 'squats': 56, 'stairs_down': 23, 'stairs_up': 48, 'standing': 41, 'walking': 317} | 
|  |  |  |  | 
| min | 0.97 | 0.0 | - | 
| max | 1.0 | 0.04 | - | 
| mean | 0.98 | 0.02 | - | 
| median | 0.98 | 0.02 | - | 

### Model "window_size 150"

{'window_size': 150, 'stride_size': 150, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.96 | 0.06 | {'running': 65, 'squats': 19, 'stairs_down': 16, 'stairs_up': 25, 'standing': 137, 'walking': 415} | 
| 1 | 0.94 | 0.06 | {'running': 14, 'squats': 5, 'stairs_down': 42, 'stairs_up': 40, 'standing': 37, 'walking': 86} | 
| 2 | 0.98 | 0.02 | {'running': 28, 'squats': 10, 'stairs_down': 48, 'stairs_up': 35, 'standing': 118, 'walking': 429} | 
| 3 | 1.0 | 0.0 | {'running': 13, 'squats': 24, 'stairs_down': 20, 'stairs_up': 38, 'standing': 277, 'walking': 167} | 
| 4 | 1.0 | 0.0 | {'running': 43, 'squats': 48, 'stairs_down': 20, 'stairs_up': 41, 'standing': 36, 'walking': 275} | 
|  |  |  |  | 
| min | 0.94 | 0.0 | - | 
| max | 1.0 | 0.06 | - | 
| mean | 0.98 | 0.03 | - | 
| median | 0.98 | 0.02 | - | 

### Model "window_size 170"

{'window_size': 170, 'stride_size': 170, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.94 | 0.06 | {'running': 57, 'squats': 16, 'stairs_down': 13, 'stairs_up': 21, 'standing': 120, 'walking': 366} | 
| 1 | 0.94 | 0.06 | {'running': 12, 'squats': 4, 'stairs_down': 36, 'stairs_up': 35, 'standing': 33, 'walking': 76} | 
| 2 | 0.98 | 0.02 | {'running': 24, 'squats': 8, 'stairs_down': 43, 'stairs_up': 30, 'standing': 103, 'walking': 378} | 
| 3 | 1.0 | 0.0 | {'running': 12, 'squats': 21, 'stairs_down': 17, 'stairs_up': 33, 'standing': 244, 'walking': 147} | 
| 4 | 0.99 | 0.01 | {'running': 38, 'squats': 43, 'stairs_down': 18, 'stairs_up': 36, 'standing': 30, 'walking': 242} | 
|  |  |  |  | 
| min | 0.94 | 0.0 | - | 
| max | 1.0 | 0.06 | - | 
| mean | 0.97 | 0.03 | - | 
| median | 0.98 | 0.02 | - | 

### Model "window_size 190"

{'window_size': 190, 'stride_size': 190, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.95 | 0.07 | {'running': 51, 'squats': 16, 'stairs_down': 12, 'stairs_up': 20, 'standing': 107, 'walking': 328} | 
| 1 | 0.97 | 0.03 | {'running': 11, 'squats': 4, 'stairs_down': 33, 'stairs_up': 30, 'standing': 30, 'walking': 68} | 
| 2 | 0.98 | 0.02 | {'running': 22, 'squats': 8, 'stairs_down': 38, 'stairs_up': 26, 'standing': 92, 'walking': 338} | 
| 3 | 1.0 | 0.0 | {'running': 11, 'squats': 19, 'stairs_down': 16, 'stairs_up': 29, 'standing': 218, 'walking': 132} | 
| 4 | 0.97 | 0.03 | {'running': 33, 'squats': 39, 'stairs_down': 15, 'stairs_up': 31, 'standing': 28, 'walking': 216} | 
|  |  |  |  | 
| min | 0.95 | 0.0 | - | 
| max | 1.0 | 0.07 | - | 
| mean | 0.97 | 0.03 | - | 
| median | 0.97 | 0.03 | - | 

### Model "window_size 210"

{'window_size': 210, 'stride_size': 210, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.98 | 0.03 | {'running': 44, 'squats': 12, 'stairs_down': 11, 'stairs_up': 16, 'standing': 97, 'walking': 296} | 
| 1 | 0.88 | 0.14 | {'running': 10, 'squats': 3, 'stairs_down': 30, 'stairs_up': 27, 'standing': 26, 'walking': 60} | 
| 2 | 0.98 | 0.02 | {'running': 20, 'squats': 6, 'stairs_down': 34, 'stairs_up': 24, 'standing': 83, 'walking': 306} | 
| 3 | 1.0 | 0.0 | {'running': 9, 'squats': 17, 'stairs_down': 14, 'stairs_up': 26, 'standing': 197, 'walking': 119} | 
| 4 | 0.98 | 0.01 | {'running': 30, 'squats': 33, 'stairs_down': 14, 'stairs_up': 29, 'standing': 25, 'walking': 196} | 
|  |  |  |  | 
| min | 0.88 | 0.0 | - | 
| max | 1.0 | 0.14 | - | 
| mean | 0.96 | 0.04 | - | 
| median | 0.98 | 0.02 | - | 

### Model "window_size 230"

{'window_size': 230, 'stride_size': 230, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.99 | 0.02 | {'running': 42, 'squats': 12, 'stairs_down': 9, 'stairs_up': 16, 'standing': 89, 'walking': 270} | 
| 1 | 0.98 | 0.02 | {'running': 9, 'squats': 3, 'stairs_down': 26, 'stairs_up': 26, 'standing': 23, 'walking': 55} | 
| 2 | 0.98 | 0.02 | {'running': 18, 'squats': 6, 'stairs_down': 30, 'stairs_up': 22, 'standing': 76, 'walking': 279} | 
| 3 | 0.42 | 0.6 | {'running': 8, 'squats': 15, 'stairs_down': 12, 'stairs_up': 24, 'standing': 180, 'walking': 108} | 
| 4 | 0.99 | 0.02 | {'running': 28, 'squats': 30, 'stairs_down': 13, 'stairs_up': 26, 'standing': 22, 'walking': 179} | 
|  |  |  |  | 
| min | 0.42 | 0.02 | - | 
| max | 0.99 | 0.6 | - | 
| mean | 0.87 | 0.14 | - | 
| median | 0.98 | 0.02 | - | 

### Model "window_size 250"

{'window_size': 250, 'stride_size': 250, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.67 | 0.38 | {'running': 38, 'squats': 11, 'stairs_down': 9, 'stairs_up': 15, 'standing': 81, 'walking': 249} | 
| 1 | 0.76 | 0.25 | {'running': 9, 'squats': 3, 'stairs_down': 24, 'stairs_up': 22, 'standing': 23, 'walking': 51} | 
| 2 | 0.98 | 0.02 | {'running': 17, 'squats': 6, 'stairs_down': 28, 'stairs_up': 21, 'standing': 70, 'walking': 257} | 
| 3 | 1.0 | 0.0 | {'running': 7, 'squats': 14, 'stairs_down': 11, 'stairs_up': 22, 'standing': 165, 'walking': 99} | 
| 4 | 1.0 | 0.0 | {'running': 25, 'squats': 28, 'stairs_down': 11, 'stairs_up': 24, 'standing': 21, 'walking': 165} | 
|  |  |  |  | 
| min | 0.67 | 0.0 | - | 
| max | 1.0 | 0.38 | - | 
| mean | 0.88 | 0.13 | - | 
| median | 0.98 | 0.02 | - | 

### Model "window_size 270"

{'window_size': 270, 'stride_size': 270, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.92 | 0.09 | {'running': 36, 'squats': 10, 'stairs_down': 8, 'stairs_up': 13, 'standing': 75, 'walking': 230} | 
| 1 | 0.8 | 0.22 | {'running': 7, 'squats': 2, 'stairs_down': 23, 'stairs_up': 22, 'standing': 20, 'walking': 46} | 
| 2 | 0.99 | 0.01 | {'running': 15, 'squats': 4, 'stairs_down': 26, 'stairs_up': 18, 'standing': 64, 'walking': 237} | 
| 3 | 1.0 | 0.0 | {'running': 7, 'squats': 13, 'stairs_down': 11, 'stairs_up': 19, 'standing': 152, 'walking': 92} | 
| 4 | 0.97 | 0.02 | {'running': 24, 'squats': 25, 'stairs_down': 11, 'stairs_up': 23, 'standing': 20, 'walking': 152} | 
|  |  |  |  | 
| min | 0.8 | 0.0 | - | 
| max | 1.0 | 0.22 | - | 
| mean | 0.94 | 0.07 | - | 
| median | 0.97 | 0.02 | - | 

### Model "window_size 290"

{'window_size': 290, 'stride_size': 290, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 1.0 | 0.0 | {'running': 33, 'squats': 10, 'stairs_down': 8, 'stairs_up': 12, 'standing': 70, 'walking': 214} | 
| 1 | 0.88 | 0.13 | {'running': 7, 'squats': 2, 'stairs_down': 20, 'stairs_up': 19, 'standing': 19, 'walking': 43} | 
| 2 | 0.98 | 0.02 | {'running': 13, 'squats': 4, 'stairs_down': 25, 'stairs_up': 16, 'standing': 60, 'walking': 220} | 
| 3 | 1.0 | 0.0 | {'running': 7, 'squats': 12, 'stairs_down': 10, 'stairs_up': 18, 'standing': 143, 'walking': 85} | 
| 4 | 0.97 | 0.03 | {'running': 21, 'squats': 24, 'stairs_down': 10, 'stairs_up': 21, 'standing': 17, 'walking': 141} | 
|  |  |  |  | 
| min | 0.88 | 0.0 | - | 
| max | 1.0 | 0.13 | - | 
| mean | 0.97 | 0.04 | - | 
| median | 0.98 | 0.02 | - | 

### Model "window_size 310"

{'window_size': 310, 'stride_size': 310, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.98 | 0.02 | {'running': 29, 'squats': 8, 'stairs_down': 7, 'stairs_up': 10, 'standing': 65, 'walking': 200} | 
| 1 | 0.94 | 0.06 | {'running': 6, 'squats': 2, 'stairs_down': 19, 'stairs_up': 18, 'standing': 18, 'walking': 40} | 
| 2 | 0.97 | 0.02 | {'running': 12, 'squats': 4, 'stairs_down': 23, 'stairs_up': 16, 'standing': 56, 'walking': 205} | 
| 3 | 1.0 | 0.0 | {'running': 6, 'squats': 11, 'stairs_down': 10, 'stairs_up': 17, 'standing': 133, 'walking': 80} | 
| 4 | 0.98 | 0.02 | {'running': 20, 'squats': 22, 'stairs_down': 10, 'stairs_up': 18, 'standing': 16, 'walking': 132} | 
|  |  |  |  | 
| min | 0.94 | 0.0 | - | 
| max | 1.0 | 0.06 | - | 
| mean | 0.97 | 0.02 | - | 
| median | 0.98 | 0.02 | - | 

### Model "window_size 330"

{'window_size': 330, 'stride_size': 330, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.97 | 0.03 | {'running': 28, 'squats': 7, 'stairs_down': 6, 'stairs_up': 10, 'standing': 61, 'walking': 188} | 
| 1 | 0.93 | 0.08 | {'running': 6, 'squats': 2, 'stairs_down': 18, 'stairs_up': 18, 'standing': 17, 'walking': 38} | 
| 2 | 0.98 | 0.02 | {'running': 12, 'squats': 4, 'stairs_down': 21, 'stairs_up': 15, 'standing': 53, 'walking': 193} | 
| 3 | 1.0 | 0.0 | {'running': 6, 'squats': 11, 'stairs_down': 9, 'stairs_up': 16, 'standing': 124, 'walking': 75} | 
| 4 | 0.94 | 0.08 | {'running': 19, 'squats': 21, 'stairs_down': 9, 'stairs_up': 17, 'standing': 16, 'walking': 124} | 
|  |  |  |  | 
| min | 0.93 | 0.0 | - | 
| max | 1.0 | 0.08 | - | 
| mean | 0.96 | 0.04 | - | 
| median | 0.97 | 0.03 | - | 

### Model "window_size 350"

{'window_size': 350, 'stride_size': 350, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.96 | 0.07 | {'running': 27, 'squats': 6, 'stairs_down': 6, 'stairs_up': 10, 'standing': 58, 'walking': 177} | 
| 1 | 0.96 | 0.04 | {'running': 6, 'squats': 2, 'stairs_down': 16, 'stairs_up': 16, 'standing': 15, 'walking': 36} | 
| 2 | 0.99 | 0.01 | {'running': 11, 'squats': 4, 'stairs_down': 19, 'stairs_up': 14, 'standing': 50, 'walking': 184} | 
| 3 | 1.0 | 0.0 | {'running': 6, 'squats': 10, 'stairs_down': 8, 'stairs_up': 16, 'standing': 117, 'walking': 70} | 
| 4 | 0.67 | 0.34 | {'running': 18, 'squats': 20, 'stairs_down': 9, 'stairs_up': 15, 'standing': 14, 'walking': 117} | 
|  |  |  |  | 
| min | 0.67 | 0.0 | - | 
| max | 1.0 | 0.34 | - | 
| mean | 0.92 | 0.09 | - | 
| median | 0.96 | 0.04 | - | 

### Model "window_size 370"

{'window_size': 370, 'stride_size': 370, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.97 | 0.05 | {'running': 26, 'squats': 6, 'stairs_down': 5, 'stairs_up': 9, 'standing': 55, 'walking': 167} | 
| 1 | 0.48 | 0.52 | {'running': 5, 'squats': 2, 'stairs_down': 16, 'stairs_up': 14, 'standing': 13, 'walking': 34} | 
| 2 | 0.98 | 0.02 | {'running': 10, 'squats': 4, 'stairs_down': 18, 'stairs_up': 14, 'standing': 47, 'walking': 172} | 
| 3 | 1.0 | 0.0 | {'running': 5, 'squats': 9, 'stairs_down': 7, 'stairs_up': 15, 'standing': 111, 'walking': 66} | 
| 4 | 0.98 | 0.02 | {'running': 16, 'squats': 19, 'stairs_down': 7, 'stairs_up': 15, 'standing': 14, 'walking': 110} | 
|  |  |  |  | 
| min | 0.48 | 0.0 | - | 
| max | 1.0 | 0.52 | - | 
| mean | 0.88 | 0.12 | - | 
| median | 0.98 | 0.02 | - | 

### Model "window_size 390"

{'window_size': 390, 'stride_size': 390, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.95 | 0.08 | {'running': 23, 'squats': 6, 'stairs_down': 5, 'stairs_up': 7, 'standing': 51, 'walking': 159} | 
| 1 | 0.97 | 0.03 | {'running': 5, 'squats': 2, 'stairs_down': 14, 'stairs_up': 14, 'standing': 13, 'walking': 31} | 
| 2 | 0.75 | 0.25 | {'running': 9, 'squats': 3, 'stairs_down': 17, 'stairs_up': 12, 'standing': 45, 'walking': 162} | 
| 3 | 0.99 | 0.01 | {'running': 5, 'squats': 9, 'stairs_down': 7, 'stairs_up': 12, 'standing': 104, 'walking': 63} | 
| 4 | 0.99 | 0.01 | {'running': 15, 'squats': 18, 'stairs_down': 6, 'stairs_up': 15, 'standing': 12, 'walking': 105} | 
|  |  |  |  | 
| min | 0.75 | 0.01 | - | 
| max | 0.99 | 0.25 | - | 
| mean | 0.93 | 0.08 | - | 
| median | 0.97 | 0.03 | - | 

### Model "window_size 410"

{'window_size': 410, 'stride_size': 410, 'test_percentage': 0.3, 'n_features': 15, 'n_outputs': 6}


|  k_fold_idx | correct_classification_acc | avg_failure_rate | test_activity_distribution |
|-------------- | -------------- | -------------- | -------------- | 
| 0 | 0.74 | 0.26 | {'running': 21, 'squats': 6, 'stairs_down': 5, 'stairs_up': 7, 'standing': 49, 'walking': 151} | 
| 1 | 0.42 | 0.59 | {'running': 5, 'squats': 1, 'stairs_down': 14, 'stairs_up': 14, 'standing': 13, 'walking': 31} | 
| 2 | 0.98 | 0.02 | {'running': 9, 'squats': 2, 'stairs_down': 15, 'stairs_up': 11, 'standing': 42, 'walking': 154} | 
| 3 | 1.0 | 0.0 | {'running': 4, 'squats': 8, 'stairs_down': 6, 'stairs_up': 11, 'standing': 98, 'walking': 60} | 
| 4 | 1.0 | 0.0 | {'running': 14, 'squats': 14, 'stairs_down': 6, 'stairs_up': 14, 'standing': 12, 'walking': 100} | 
|  |  |  |  | 
| min | 0.42 | 0.0 | - | 
| max | 1.0 | 0.59 | - | 
| mean | 0.83 | 0.17 | - | 
| median | 0.98 | 0.02 | - | 

