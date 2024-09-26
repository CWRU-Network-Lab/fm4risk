# Check how relevant the response is to see if the llm will get better
# nathaniel hahn
#

import os
import plotext as plt
import pandas as pd

# determine difference between top result and llm generated result as a percentage
def calc_diff(n_results, similarity):
    print(similarity)
    sim = float(similarity[0][0].cpu().detach().numpy()[0])
    return sim / float(n_results[0][2])

# read in output directory and summarize the results
def summarize(directory):
    summary = os.path.join(directory, 'summary.txt')

    with open(summary, 'w') as summary:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= 4:
                        scores = lines[2].strip()
                        query = lines[0].strip()
                        summary.write(f'{filename}: {query}: {scores}\n')
                    else:
                        summary.write(f'{filename}: (no line 4)\n')

    print(f'Summary created: {summary}')


#summarize('./data/dirt')
#summarize('./data/sand')
#summarize('./data/gravel')
#summarize('./data/openai_gpt4')


# build dataframe for analysis from summary text
def create_df(directory):
    filename = os.path.join(directory, 'summary.txt')

    with open(filename, 'r') as file:
        data = file.readlines()
    print(data)

    filenames = []
    scores = []
    queries = []

    for line in data:
        try:
            filename, rest = line.split(': ', 1)
            print(rest)
            query, score = rest.split(': ', 1)
            series = list(map(float, score.split()))
            filenames.append(filename)
            scores.append(series)
            queries.append(str(query))
        except:
            print("failed on line: " + str(line))

    df = pd.DataFrame({
        'Filenames': filenames,
        'Query': queries,
        'Scores': scores
    })

    return df

# graph results 
def graph_changes(data):
    parsed_data = data

    plt.clf()
    for key, values in parsed_data.items():
        plt.plot(values, label=key)

    plt.title('Graph of Values from summary.txt')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

# generate basic stats from dataframe
def generate_stats(dataframe):
    df = dataframe
    df['final_change'] = df['Scores'].apply(compute_change)
    df[['max_value', 'max_index', 'max_change']] = pd.DataFrame(df['Scores'].apply(find_max_and_index).tolist(), index=df.index)


    df = df.sort_values('Query')
    print(df[['Query', 'final_change', 'max_value', 'max_index', 'max_change']])


# generate initial vs final rank/similarity change
def compute_change(score_list):
    if len(score_list) > 1:
        return score_list[-1] - score_list[0]
    else:
        return 0

# find best/worst snippets generated
def find_max_and_index(score_list):
    max_value = max(score_list)
    max_index = score_list.index(max_value)
    change = max_value - score_list[0]

    return max_value, max_index, change

#data = create_df('./data/dirt')


#g_data = create_df('./data/gravel')
#s_data = create_df('./data/sand')
#b_data = create_df('./data/openai_gpt4')
#graph_changes(data)

#generate_stats(data)
#generate_stats(s_data)
#generate_stats(g_data)
#generate_stats(b_data)


# ranking them:
#df_combined = pd.merge(data[['Query', 'max_value']], g_data[['Query', 'max_value']], on='Query', suffixes=('_5try_10s', '_10try_5s'))
#df_combined = pd.merge(df_combined, s_data[['Query', 'max_value']], on='Query')
#df_combined = pd.merge(df_combined, b_data[['Query', 'max_value']], on='Query', suffixes=('_5try_5s', '_base'))

#df_combined.columns = ['Query', 'max_value_5try_10s', 'max_value_10try_5s', 'max_value_5try_5s', 'max_value_base']

#df_combined['max_combined'] = df_combined[['max_value_5try_10s', 'max_value_10try_5s', 'max_value_5try_5s', 'max_value_base']].max(axis=1)

# Print the DataFrame with ranks
#print(df_combined[['Query', 'max_value_5try_10s', 'max_value_10try_5s', 'max_value_5try_5s', 'max_value_base', 'max_combined']])

