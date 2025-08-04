import spacy

nlp = spacy.load("en_core_web_sm")

text = """
John Doe, a software engineer at Google, moved to London in 2023. He previously worked at IBM in New York.
He graduated from MIT in 2018 with a degree in Computer Science.
"""

doc = nlp(text)

print("Named Entities: \n")
for ent in doc.ents:
    print(f"{ent.text:30} --> {ent.label_}")

print("\nEntity Pairs (sample rule-based): \n")
for sent in doc.sents:
    subj = ""
    verb = ""
    obj = ""
    for token in sent:
        if "subj" in token.dep_:
            subj = token.text
        if token.pos_ == "VERB":
            verb = token.lemma_
        if "obj" in token.dep_:
            obj = token.text
    if subj and verb and obj:
        print(f"{subj} {verb} {obj}")
