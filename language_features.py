import pandas as pd
import numpy as np
import spacy
from spacy import displacy


def read_from_file(file_name):
    print("Importing csv")
    processed_data = pd.read_csv(file_name)
    return processed_data


def compute_question_length(input_data):
    questions = input_data['Question']
    question_length = []
    for i in range(0, questions.shape[0]):
        question_length.append(len(questions[i]))
    return question_length


def named_entity_recognition(input_data, entity_label_type):
    questions = input_data['Question']
    named_entities = []
    # question = questions[1]
    # doc = nlp(question)
    # print(doc)
    # for ent in doc.ents:
    #     print(ent.text, " ", ent.label_)
    # displacy.serve(doc, style="ent")

    for question in questions:
        names = []
        doc = nlp(question)
        for ent in doc.ents:
            if ent.label_ == entity_label_type:
                names.append(ent.text)
        named_entities.append(names)

    return named_entities
    #     displacy.serve(doc, style="ent")


def get_name_count(names):
    name_count = []
    name_boolean = []
    for name in names:
        name_count.append(len(name))
        if len(name) > 0:
            name_boolean.append(True)
        else:
            name_boolean.append(False)
    return name_count, name_boolean


def get_person_name_count_details(input_data):
    person_names = named_entity_recognition(input_data, "PERSON")
    name_count, name_boolean = get_name_count(person_names)
    # print(person_name_count)
    # print(person_name_boolean)
    return name_count, name_boolean


def get_conjunction_phrases(input_data):
    questions = input_data["Question"]
    conjunction_count = []
    for question in questions:
        count = 0
        doc = nlp(question)
        for token in doc:
            if token.dep_ == "cc":
                count += 1
        conjunction_count.append(count)
    return conjunction_count


def get_prepositional_phrases(input_data):
    questions = input_data["Question"]
    preposition_count = []
    for question in questions:
        count = 0
        doc = nlp(question)
        for token in doc:
            if token.dep_ == "prep":
                count += 1
        preposition_count.append(count)
    return preposition_count


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    data = read_from_file("More_Processed_Data.csv")

    # This will be the final dataframe
    data_language_features = data.copy()

    # Extract sentence length
    ques_length = compute_question_length(data)

    # Find all person names in a given sentence
    person_name_count, person_name_boolean = get_person_name_count_details(data)

    # Find all conjunction phrases in a given sentence
    conjunction_phrase_count = get_conjunction_phrases(data)

    # Find all prepositional phrases in a given sentence
    preposition_phrase_count = get_prepositional_phrases(data)

    # Finally append all the relevant columns
    data_language_features['question_length'] = np.array(ques_length)
    print(data_language_features.head())