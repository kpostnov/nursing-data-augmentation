- load_dataset of our dataset needs to map its values to 0 to n_classes - 1 -> internal representation for all labels
- settings.init(mapping) - we need to pass the mapping from label to activity string
(- the same for the subject -> turn it into subject_id)