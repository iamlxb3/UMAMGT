import pandas as pd
import ipdb


def main():
    file_name = 'Concretenss Ratings of 9877 Two Character Chinese Words.xlsx'
    xl_file = pd.ExcelFile(file_name)
    dfs = {sheet_name: xl_file.parse(sheet_name)
           for sheet_name in xl_file.sheet_names}
    save_df = {'Word': [], 'Conc.M': [], 'Conc.SD': []}
    df = dfs['Concreteness Ratings']

    for i, row in df.iterrows():
        word = row['Word']
        rating = row['Mean of Valid Ratings']
        std = row['SD of Valid Ratings']
        save_df['Word'].append(word)
        save_df['Conc.M'].append(6 - rating)
        save_df['Conc.SD'].append(std)

    save_df = pd.DataFrame(save_df)
    save_df.to_csv('Concreteness_ratings_cn_bigrams.csv', index=False)
    ipdb.set_trace()


if __name__ == '__main__':
    main()
