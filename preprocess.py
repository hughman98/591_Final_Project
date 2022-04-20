# Combines all assistments CSVs into one dataset called "Processed_Data.csv"
# Output contains all "step 0" problem's text, KCs, total time to solve, num attempts,
# average time to solve, num successful attempts, num unsuccessful attempts

import pandas as pd
from datetime import datetime


# removes all values before the first two colons.
# so "Step0:3684: What is an apple?" would become "What is an apple?"
def custom_strip(text):
    text_list = text.split(":")

    ret = ""

    # To deal with instances where there are colons in the problem that we want to keep
    for i in text_list[2:]:
        ret += ':' + i

    return ret[1:]


# Import the Assistments CSV
path1 = "ds92_tx_All_Data_172_2016_0504_081852.txt"
path2 = "ds120_tx_All_Data_265_2017_0414_065125.txt"
path3 = "ds339_tx_All_Data_1059_2015_0729_215742.txt"

# List of columns we want to import
cols = ["Step Name", "Attempt At Step", "Is Last Attempt", "KC (WPI-Apr-2005)", "KC (WPI-Apr-2005).1",
        "KC (WPI-Apr-2005).2", "KC (WPI-Apr-2005).3", "Time", "Outcome", "Problem Start Time"]

df1 = pd.read_csv(path1, sep='\t', usecols=cols)
df2 = pd.read_csv(path2, sep='\t', usecols=cols)
df3 = pd.read_csv(path3, sep='\t', usecols=cols)

print('Loaded!')

df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

print('Concatenated!')

pd.set_option('display.max_columns', None)

count = 0
content_questions = {}
empty_questions = {}
figure_questions = {}
t_format = "%Y-%m-%d %H:%M:%S"
for index, row in df.iterrows():
    text = row['Step Name']

    if type(text) == str and "Step0" in text and row['Is Last Attempt'] == 1:
        # Detect and remove questions we don't want, while tracking how many we removed and for what reason...
        if (text.endswith(':') and len(text) < 25):
            empty_questions[text] = True
        elif 'Graph' in text or 'graph' in text:
            text = custom_strip(text)
            figure_questions[text] = True

        # Add useful problems to the set
        else:
            text = custom_strip(text)
            # KCs, total time spent, num students, average time to solve, num successful attempts, num unsuccessful
            # attempts, and num hint requests
            if text not in content_questions:
                delta_time = datetime.strptime(row["Time"], t_format) - \
                             datetime.strptime(row["Problem Start Time"], t_format)
                delta_time = delta_time.seconds

                content_questions[text] = [row["KC (WPI-Apr-2005)"],  # 0
                                           row["KC (WPI-Apr-2005).1"],  # 1
                                           row["KC (WPI-Apr-2005).2"],  # 2
                                           row["KC (WPI-Apr-2005).3"],  # 3
                                           delta_time,  # total time       4
                                           1,  # num students      5
                                           delta_time,  # Average time     6
                                           0,  # num successful attempts   7
                                           0]  # num failed attempts       8

                if row['Outcome'] == 'CORRECT':
                    content_questions[text][7] = 1
                    content_questions[text][8] = 0
                else:
                    content_questions[text][7] = 0
                    content_questions[text][8] = 1

            else:  # question already in the data frame
                delta_time = datetime.strptime(row["Time"], t_format) - \
                             datetime.strptime(row["Problem Start Time"], t_format)
                delta_time = delta_time.seconds

                content_questions[text][4] += delta_time
                content_questions[text][5] += 1
                content_questions[text][6] = content_questions[text][4] / content_questions[text][5]

                if row['Outcome'] == 'CORRECT':
                    content_questions[text][7] += 1
                else:
                    content_questions[text][8] += 1

print("Number of usable questions:", len(content_questions))

print("Number of empty questions removed:", len(empty_questions))
print("Number of graph questions removed:", len(figure_questions))

export_cols = ["KC0", "KC1", "KC2", "KC3", "tot_time_spent", "num students", "average time to solve", "num_success",
               "num_failed"]
export = pd.DataFrame.from_dict(content_questions, orient='index', columns=export_cols)

export.to_csv("Processed_Data.csv", index='True')
