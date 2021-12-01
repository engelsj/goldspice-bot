import pandas as pd
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
import gpt_2_simple as gpt2


# pull in data sets and stack them into singular df
df1 = pd.read_csv("whole-breaking-financial.csv")
df1 = df1.loc[df1['Author'] == 'Wizard Lizard#6804']
df2 = pd.read_csv("whole-der-analysis.csv")
df2 = df2.loc[df2['Author'] == 'Wizard Lizard#6804']

# concat dfs together and isolate messages
full_set = pd.concat([df1, df2])
# remove data that has less than 3 spaces (assuming 3 spaces = 3 full words)
full_set = full_set[full_set["Content"].str.count('\s+').gt(4)]
full_set = full_set["Content"] 

full_set = full_set.array

with open("output.txt", "w",  encoding='utf-8') as txt_file:
    for line in full_set:
        txt_file.write("".join(line) + "\n")

model_name = "124M"

# Download the model if it is not present
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)

# Start a Tensorflow session to pass to gpt2_simple
sess = gpt2.start_tf_sess()

# Define the number of steps we want our model to take we want this to be such that
# we only pass over the data set 1-2 times to avoid overfitting.
num_steps = 100

# This is the path to the text file we want to use for training.
text_path = "output.txt"

# Pass in the session and the
sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=text_path,
              model_name='124M',
              steps=300,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=100,
              )

gpt2.generate(sess)