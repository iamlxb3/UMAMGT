def create_plot_df(src_df, target_df):
    for char_freq_range, char_freq_df in src_df.groupby('char_freq_range'):

        if char_freq_range == 0:
            continue

        origin_mask = (char_freq_df['char_freq_range'] == char_freq_range) & (
                    char_freq_df['semantic_change'] == "origin")
        origin_df = char_freq_df[origin_mask]
        other_df = char_freq_df[~origin_mask]

        for i, row in other_df.iterrows():
            test_acc = row['test_acc']
            semantic_change = row['semantic_change']
            repeat_i = row['repeat_i']
            origin_row_df = origin_df[origin_df['repeat_i'] == repeat_i]
            origin_acc = float(origin_row_df['test_acc'])

            relative_improve = test_acc - origin_acc

            target_df['char_freq_range'].append(char_freq_range)
            target_df['acc'].append(test_acc)
            target_df['origin_acc'].append(origin_acc)
            target_df['relative_improve'].append(relative_improve)
            target_df['type'].append('acc')
            target_df['semantic_change'].append(semantic_change)
    target_df = pd.DataFrame(target_df)


sinha2021masked,