import contractions

sentence = "I can't go to the party because I haven't finished my homework. She'll be upset if I don't show up."
expanded_sentence = contractions.fix(sentence)

print("Original Sentence:")
print(sentence)

print("\nExpanded Sentence:")
print(expanded_sentence)
