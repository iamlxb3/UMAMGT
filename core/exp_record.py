import pandas as pd


class ExpRecorder:
    def __init__(self):
        self.result_df = {
            'repeat_i': [],
            'is_change_apply_to_test': [],
            'classifier_name': [],
            'dataset_name': [],
            'semantic_change': [],
            'char_freq_range': [],
            'test_acc': [],
            'test_f1': [],
            'train_loss': [],
            'test_loss': [],
            'train_size': [],
            'val_size': [],
            'test_size': []
        }

    def add_one_repeat_result(self,
                              repeat_i,
                              is_change_apply_to_test,
                              classifier_name,
                              dataset_name,
                              semantic_change,
                              char_freq_range,
                              test_acc,
                              test_f1,
                              train_loss,
                              test_loss,
                              train_size,
                              val_size,
                              test_size
                              ):
        self.result_df['repeat_i'].append(repeat_i)
        self.result_df['is_change_apply_to_test'].append(is_change_apply_to_test)
        self.result_df['classifier_name'].append(classifier_name)
        self.result_df['dataset_name'].append(dataset_name)
        self.result_df['semantic_change'].append(semantic_change)
        self.result_df['char_freq_range'].append(char_freq_range)
        self.result_df['test_acc'].append(test_acc)
        self.result_df['test_f1'].append(test_f1)
        self.result_df['train_loss'].append(train_loss)
        self.result_df['test_loss'].append(test_loss)
        self.result_df['train_size'].append(train_size)
        self.result_df['val_size'].append(val_size)
        self.result_df['test_size'].append(test_size)
        return self.result_df

    def save_to_disk(self, path):
        save_df = pd.DataFrame(self.result_df)
        save_df.to_csv(path, index=False)
        print(f"Save to {path} done, shape: {save_df.shape}")
