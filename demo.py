"""TABLET Demo"""
import argparse
import numpy as np
import pandas as pd

import streamlit as st
import requests

import openai

from Tablet.prototypes import create_one_v_all_style
from Tablet.synthetic_language import load_openai_key

OVERVIEW = """While tabular data is highly prevalent in many applications, it's often difficult to gather sufficient training data, due issues like privacy concerns or costs.
What if we could use _large language models_ (LLMs) to learn from task _instructions_ to solve tabular prediction problems, with _few or no examples_?
We've built a benchmark called **TABLET** that contains 20 tabular prediction datasets annotated with several different types of task instructions that vary 
in their phrasing, granularity, and how they were collected. In our evaluation, we found instructions help improve performance on tabular datasets, and that
there's still room for improvement to achieve performance similar to fully supervised models trained on all the data. It's our hope that TABLET helps researchers develop models that 
achieve strong results on tabular prediction tasks using only instructions and perhaps a few examples.
"""

INSTRUCTIONSTEXT = """We've provided three tabular data sets each with three different instructions for you to play around with. You can choose the data set, data point, and LLM using the left hand panel. You can also edit the instructions to see the effects on the model's predictions."""

INSTRUCTIONS = {
    "Heart Disease": ["""Patients with heart disease tend to have higher values for age, thalach (maximum heart rate achieved), oldpeak (ST depression induced by exercise relative to rest), ca (number of major vessels colored by flourosopy) and thal (normal, fixed defect, reversable defect). Patients without heart disease typically have lower values for these features. Additionally, patients with heart disease are more likely to have a positive value for exang (exercise induced angina) and slope (slope of the peak exercise ST segment).""", """patients with heart disease tend to have higher age, restecg values of 2 or greater, lower thalach values, higher oldpeak values, exang values of yes, slope values of 2, sex of male, cp values of 4, ca values of 0.98 or greater, and thal values of 7 or greater. Patients without heart disease tend to have lower age, restecg values of 0 or 1, higher thalach values, lower oldpeak values, exang values of no, slope values of 1 or 2, sex of male, cp values of 3 or less, ca values of 0.25 or less, and thal values of 3 or less.""", """Patients with a thal value of 3.0 have a 45% chance of not having heart disease, while patients with any other thal value have a 55% chance of having heart disease."""],
    "Wine": ["""Wines can be classified by their origin based on the values of their features. Wines from origin 1 tend to have higher hue, 0D280 0D315 of diluted wines, magnesium, alcohol, total phenols, and color intensity values, while wines from origin 2 tend to have higher malicacid, proline, and flavanoids values. Wines from origin 3 tend to have lower hue, 0D280 0D315 of diluted wines, magnesium, alcohol, total phenols, and color intensity values, as well as higher malicacid, proline, and flavanoids values.""",
             """Wines can be classified by their origin in three classes based on the features Proline, Color intensity and Flavanoids. Wines with a Proline of greater than 755 are likely to have an origin of class 1 with 65% probability. Alternatively, wines with low Color Intensity values of less than 3.5 are more likely to originate from class 2 at 59% probability. Lastly, wines with Flavanoid levels lower than 1.4 should be classified as belonging to class 3 with a 75% probability.""",
             """Wines can be classified into three different origins based on their features. Wines from origin 1 tend to have a hue of around 1.03-1.1, 0D280 0D315 of diluted wines of around 3.3-2.98, malicacid of around 2.18-1.79, magnesium of around 107.39-106.29, alcohol of around 13.63-13.81, ash of around 2.45-2.48, total phenols of around 2.74-2.95, proline of around 939.82-1289.05, color intensity of around 4.82-6.29, and flavanoids of around 2.84-3.15. Wines from origin 2 tend to have a hue of around 1.07-1.06, 0D280 0D315 of diluted wines of around 2.71-2.8, malicacid of around 1.86-1.97, magnesium of around 100.96-90.34, alcohol of around 12.21-12.34, ash of around 2.29-2.22, total phenols of around 2.25-2.32, proline of around 679.3-421.97, color intensity of around 2.99-3.16, and flavanoids of around 1.97-2.22. Wines from origin 3 tend to have a hue of around 0.67-0.65, 0D280 0D315 of diluted wines of around 1.6-1.68, malicacid of around 3.31-3.35, magnesium of around 100.6-99.53, alcohol of around 13.11-13.29, ash of around 2.48-2.37, total phenols of around 1.69-1.73, proline of around 549.25-725.33, color intensity of around 7.32-7.87, and flavanoids of around 0.93-0.69."""][::-1],
    "Viral pharyngitis": ["""Since there are a lot of different conditions that often share similar symptoms, a differential diagnosis lists the possible conditions that could cause the symptoms. Here are instructions for the differential diagnosis of Viral pharyngitis:
Viral pharyngitis has the following causes in patients:
Pharyngitis may occur as part of a viral infection that also involves other organs, such as the lungs or bowel. Most sore throats are caused by viruses.
Viral pharyngitis has the following symptoms:
- Discomfort when swallowing
- Fever
- Joint pain or muscle aches
- Sore throat
- Tender swollen lymph nodes in the neck
If the patient has similar symptoms, then the answer is yes, Viral pharyngitis should be included in the differential diagnosis. Otherwise, the answer is no.
""",
    """Since there are a lot of different conditions that often share similar symptoms, a differential diagnosis lists the possible conditions that could cause the symptoms. Here are instructions for the differential diagnosis of Viral pharyngitis:
Viral pharyngitis has the following symptoms: Pain with swallowing is the hallmark of tonsillopharyngitis and is often referred to the ears. Very young children who are not able to complain of sore throat often refuse to eat. High fever, malaise, headache, and GI upset are common, as are halitosis and a muffled voice. The tonsils are swollen and red and often have purulent exudates. Tender cervical lymphadenopathy may be present. Fever, adenopathy, palatal petechiae, and exudates are somewhat more common with GABHS than with viral tonsillopharyngitis, but there is much overlap. With GABHS, a scarlatiniform rash (scarlet fever) may be present.
If the patient has similar symptoms, then the answer is yes, Viral pharyngitis should be included in the differential diagnosis. Otherwise, the answer is no.""",
    """Since there are a lot of different conditions that often share similar symptoms, a differential diagnosis lists the possible conditions that could cause the symptoms. Here are instructions for the differential diagnosis of Viral pharyngitis:
Viral pharyngitis has the following symptoms:
- a painful throat, especially when swallowing
- a dry, scratchy throat
- redness in the back of your mouth
- bad breath
- a mild cough
- swollen neck glands
The symptoms are similar for children, but children can also get a temperature and appear less active.
If the patient has similar symptoms, then the answer is yes, Viral pharyngitis should be included in the differential diagnosis. Otherwise, the answer is no."""]
}


def update_multiselect_style():
    st.markdown(
        """
        <style>
            .stMultiSelect [data-baseweb="tag"] {
                height: fit-content;
            }
            .stMultiSelect [data-baseweb="tag"] span[title] {
                white-space: normal; max-width: 100%; overflow-wrap: anywhere;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def update_selectbox_style():
    st.markdown(
        """
        <style>
            .stSelectbox [data-baseweb="select"] div[aria-selected="true"] {
                white-space: normal; overflow-wrap: anywhere;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup():
    st.set_page_config(layout="wide")
    update_selectbox_style()
    update_multiselect_style()


def feature_select(feature_names: list[str], default: list[str]):
    with st.expander("View one-hot features."):
        features = st.multiselect("One-hot features:",
                                  feature_names,
                                  default)
    return features


def get_oh(row: pd.Series, oh_v: str):
    positives = []
    for i, v in row.items():
        if v == oh_v:
            positives.append(str(i))
    return positives


def encode(features: list[str],
           values: list[str],
           one_hot: bool,
           model: str,
           label: str = None):
    data_point = ""
    if one_hot:
        values = [str(v) for v in values]
        values, features = create_one_v_all_style(values, features)
    for f, v in zip(features, values):
        v = str(v)
        if v.isnumeric():
            v = f"{float(v):.2f}"
        data_point += f"{f}: {v}\n"
    if label is not None:
        if model == "Tk-Instruct 11b":
            data_point += f"Answer: {label}"
        elif model == "Flan-T5 11b" or model == "Flan-UL2 20b":
            data_point += f"Answer: {label}"
        else:
            data_point += f"The answer is {label}"
    return data_point


def create_prompt(features: list[str],
                  values: list[str],
                  k_shot_columns: list[str],
                  k_shot_values: list[str],
                  k_shot_labels: list[str],
                  instruction: str,
                  model: str,
                  one_hot: bool):
    data_point = encode(features, values, one_hot=one_hot, model=model)
    k_shot_encoded = [encode(ksc, d, one_hot, model, l)
                      for ksc, d, l in zip(k_shot_columns, k_shot_values, k_shot_labels)]
    if model == "Tk-Instruct 11b":
        example = f"\nNow complete the following example -\n{data_point}"
        k_shot = "\n".join([f"\nPositive Example {i + 1} -\n{ex}" for i, ex in enumerate(k_shot_encoded)])
        prompt = f"Definition: {instruction}\n{k_shot}\n{example}Answer:"
    elif model == "Flan-T5 11b" or model == "Flan-UL2 20b" or model == "ChatGPT":
        example = f"\nNow complete the following example -\n{data_point}"
        k_shot = "\n".join([f"\nExample -\n{ex}" for i, ex in enumerate(k_shot_encoded)])
        prompt = f"{instruction}\n{k_shot}\n{example}Answer:"
    else:
        example = f"\n{data_point}"
        k_shot = "\n".join([f"\n{ex}" for i, ex in enumerate(k_shot_encoded)])
        prompt = f"{instruction}" \
                 f"\n{k_shot}\n{example}The answer is"
    return prompt


def flan_query(payload):
    payload = {"inputs": payload}
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
    headers = {"Authorization": args.key}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def ul2_query(payload):
    payload = {"inputs": payload, "parameters": {"temperature": 1.0, }}
    API_URL = "https://api-inference.huggingface.co/models/google/flan-ul2"
    headers = {"Authorization": args.key}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def tk_query(payload):
    payload = {"inputs": payload}
    API_URL = "https://api-inference.huggingface.co/models/allenai/tk-instruct-11b-def-pos"
    headers = {"Authorization": args.key}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def chatgpt_query(payload):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": payload}
    ]
    results = openai.ChatCompletion.create(
        top_p=0.1,
        max_tokens=256,
        model="gpt-3.5-turbo",
        messages=messages
    )
    answer = results["choices"][0]["message"]["content"]
    return [{"generated_text": answer}]


def compute_answer(prompt: str, model: str):
    if model == "Flan-T5 11b":
        output = flan_query(prompt)
    elif model == "Flan-UL2 20b":
        output = ul2_query(prompt)
    elif model == "Tk-Instruct 11b":
        output = tk_query(prompt)
    elif model == "ChatGPT":
        output = chatgpt_query(prompt)
    else:
        output = "Model not setup!"
    return output


def start_demo(demo_args: argparse.Namespace):
    # st.title("TABLET")
    st.image(image="./site/assets/logo.png", use_column_width=True)
    link = '### [Website](https://dylanslacks.website/Tablet)   [Paper](https://dylanslacks.website/Tablet)   [Code](https://dylanslacks.website/Tablet)'
    st.markdown(link, unsafe_allow_html=True)
    st.markdown(OVERVIEW)
    st.subheader("Instructions")
    st.markdown(INSTRUCTIONSTEXT)
    with st.sidebar:
        st.image("./demo_data/logo.png")
        st.subheader("Configuration")
        dataset = st.radio(
            "Dataset", ['Heart Disease', 'Viral pharyngitis', 'Wine', 'Upload my own'],
            0, help="The dataset to use."
        )
        if dataset == "Viral pharyngitis":
            path = "./demo_data/Viral-pharyngitis.csv"
            df = pd.read_csv(path, index_col=0)
        elif dataset == 'Upload my own':
            notes = "Note: the label column must have the name _y_temp_"
            st.markdown(notes)
            spectra = st.file_uploader("upload file", type={"csv"})
            if spectra is not None:
                df = pd.read_csv(spectra)
            else:
                return
        elif dataset == "Heart Disease":
            df = pd.read_csv("./demo_data/HeartDisease.csv", index_col=0)
        else:
            path = "./demo_data/Wine.csv"
            df = pd.read_csv(path, index_col=0)

        help_text = (
            "One-hot will just put features into the prompt that have a specific value. "
            "You can set this value in the One-hot text box below."
        )
        help_text_model = "This will determine the prediction model."
        model = st.radio("Model", ['Flan-T5 11b', 'Flan-UL2 20b', 'ChatGPT'], 1, help=help_text_model)

        if model == "ChatGPT":
            key = st.text_area(label="OpenAI Key for ChatGPT. We will not store this.")
            if key == "":
                return
            openai.api_key = key

        st.caption("It's highly recommended to use one-hot for Viral pharyngitis and all for Wine.")
        answer = st.radio("One-hot or All Data", ['All', 'One-hot'], 1 if dataset == "Viral pharyngitis" else 0,
                          help=help_text)
        k_shot = st.selectbox("K-Shot", [0, 1, 2, 3, 4], 0, help="The number of example points included.")

    y_vals = df['y_temp']
    df = df.drop(['y_temp'], axis=1)
    df["Ground Truth"] = y_vals
    # st.subheader("Demo")
    st.write(f"Current dataset: :blue[{dataset}]")
    with st.expander("View dataset.", expanded=False):
        st.dataframe(df)

    if dataset == "Heart Disease":
        dp = 0
    elif dataset == "Viral pharyngitis":
        dp = 1
    else:
        dp = 0
    indx = st.selectbox(
        'Datapoint',
        list(df.index),
        index=dp,
        help="The index of the data point to use. Though, you can change the features as you wish.")
    yv = y_vals.loc[indx]
    np.random.seed(demo_args.seed)
    if answer == "One-hot":
        with st.sidebar:
            pos_val = st.text_input("One-hot value", value="yes", help="One hot value used to "
                                                                       "select one-hot features.")
        st.write(f'Data point :blue[{indx}] has the ground truth label: :orange[{yv}]')
        enc_df = df.drop(["Ground Truth"], axis=1)
        oh_pos = get_oh(enc_df.loc[indx], pos_val)
        features = feature_select(list(df.columns), oh_pos)
        values = [pos_val] * len(features)

        k_shot = df[df.index != indx].sample(k_shot)
        gt = k_shot["Ground Truth"]
        k_shot = k_shot.drop(["Ground Truth"], axis=1)
        k_shot_columns, k_shot_values, k_shot_labels = [], [], []
        for i, row in k_shot.iterrows():
            k_shot_oh_features = get_oh(row, pos_val)
            k_shot_oh_values = [pos_val] * len(k_shot_oh_features)
            label = gt.loc[i]
            k_shot_columns.append(k_shot_oh_features)
            k_shot_values.append(k_shot_oh_values)
            k_shot_labels.append(label)
    else:
        st.write(
            f'Data point :blue[{indx}] has the ground truth label: :orange[{yv}]')
        enc_df = df.drop(["Ground Truth"], axis=1)
        features = list(enc_df.columns)
        values = list(enc_df.loc[indx].tolist())

        k_shot = df[df.index != indx].sample(k_shot)
        gt = k_shot["Ground Truth"]
        k_shot = k_shot.drop(["Ground Truth"], axis=1)
        k_shot_columns = [list(k_shot.columns)] * len(k_shot)
        k_shot_values = k_shot.values.tolist()
        k_shot_labels = gt.values.tolist()

    if dataset == "Wine" or dataset == "Viral pharyngitis" or dataset == "Heart Disease":
        instruction_ind = st.selectbox("Default Instruction",
                                       options=[0, 1, 2],
                                       index=0)
        instruction_init = INSTRUCTIONS[dataset][instruction_ind]
        if dataset == "Wine":
            instruction_init = "Determine the origin of the Wine. " + \
                               instruction_init + \
                               " Answer with one of: 1 | 2 | 3"
            no_instructions_text = "Determine the origin of the Wine. Answer with one of: 1 | 2 | 3."
        elif dataset == "Heart Disease":
            instruction_init = "Determine if the patient has heart disease. " + \
                               instruction_init + \
                               " Answer with one of: yes | no."
            no_instructions_text = "Determine if the patient has heart disease. Answer with one of: yes | no."
        else:
            instruction_init = "Follow the instructions to determine if Viral pharyngitis should be included in a " \
                               "differential diagnosis for the patient. " + \
                               instruction_init + \
                               " Answer with one of: yes | no."
            no_instructions_text = "Follow the instructions to determine if Viral pharyngitis should be included in a " \
                                   "differential diagnosis for the patient. Answer with one of: yes | no."
    else:
        instruction_init = "Write some instructions here..."
        no_instructions_text = "Write a task description..."
    # instructions = st.text_area(label=f"Instructions:", value=instruction_init, height=300)

    instruction_init = st.text_area(label="Instruction", value=instruction_init, height=200)

    prompt = create_prompt(features,
                           values,
                           k_shot_columns=k_shot_columns,
                           k_shot_values=k_shot_values,
                           k_shot_labels=k_shot_labels,
                           instruction=instruction_init,
                           model=model,
                           one_hot=answer == "One-hot")
    updated_prompt = st.text_area(label=f"Prompt with :green[instructions]", value=prompt, height=200)

    prompt_no_instruct = create_prompt(features,
                           values,
                           k_shot_columns=k_shot_columns,
                           k_shot_values=k_shot_values,
                           k_shot_labels=k_shot_labels,
                           instruction=no_instructions_text,
                           model=model,
                           one_hot=answer == "One-hot")
    updated_prompt_no_instructions = st.text_area(label=f"Prompt :red[without instructions]",
                                                  value=prompt_no_instruct, height=200)

    if st.button('Predict!', use_container_width=True):
        prediction = compute_answer(updated_prompt, model)
        if "error" in prediction:
            st.write(prediction)
        else:
            predicted_text = prediction[0]["generated_text"]
            st.write(f":green[With instructions], {model} predicts this data point as class: :orange[{predicted_text}]")
            no_instruct_prediction = compute_answer(updated_prompt_no_instructions, model)
            no_instruct_predicted_text = no_instruct_prediction[0]["generated_text"]
            st.write(f":red[Without instructions], {model} predicts this data point as class: :orange[{no_instruct_predicted_text}]")
    else:
        st.write('Click predict to see the answer.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Streamlit TABLET demo.')
    parser.add_argument('-s', '--seed',
                        help='random seed',
                        default=90210)
    parser.add_argument('-k', '--key',
                        help='hf key',
                        required=True)
    args = parser.parse_args()
    setup()
    start_demo(args)
