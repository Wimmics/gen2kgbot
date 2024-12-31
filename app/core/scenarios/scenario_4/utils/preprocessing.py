import spacy


# Function to preprocess and extract relevant entities
def extract_relevant_entities_spacy(question):
    
    # Load the pre-trained spaCy model (en_core_web_sm is a small English model)
    # nlp = spacy.load("en_core_web_sm")
    # nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_sci_lg")  # YT Best
    # nlp = spacy.load("en_ner_craft_md")
    # nlp = spacy.load("en_ner_bionlp13cg_md")
    
    # Step 1: Process the question through the spaCy pipeline
    doc = nlp(question)
    
    # Step 2: Extract relevant entities
    # relevant_entities = []

    # Define a list of entity types to keep (you can modify this as per your needs)
    # relevant_entity_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME']

    relevant_entities = ", ".join([doc.text for doc in doc.ents])

    # for ent in doc.ents:
    #     if ent.label_ in relevant_entity_types:
    #         relevant_entities.append(ent.text)

    # Return the relevant entities
    return relevant_entities

# # Example question
# question = "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 10 µM?"

# # Get relevant entities
# print("Relevant Entities Spacy:", extract_relevant_entities_spacy(question))
