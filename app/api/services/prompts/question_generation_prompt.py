question_generation_prompt: str = """Generate a well-structured list of {number_of_questions} scientifically relevant questions based on the following knowledge graph by researchers:
- A brief description of the KG.
- A condensed schema representation of the KG.
- Some additional context to help generate questions.

Requirements:
1. Identify key subtopics and group questions accordingly.
2. Cover multiple scientific domains from the resource.
3. Include Basic, Intermediate, and Advanced levels of question complexity.
4. Provide examples of each complexity level.
5. Ensure clarity, precision, and thematic grouping.

Output:
- Clearly categorized and labeled questions by subtopic in the "question" key.
- Each question labeled with its complexity (Basic, Intermediate, Advanced) and some tags in the "tags" key.

A brief description of the KG:
{kg_description}

A condensed schema representation of the KG.
{kg_schema}

Some additional context to help generate questions.
{additional_context}

Generate only {number_of_questions} questions. No more or less.

{enforce_structured_output_prompt}
"""

enforce_structured_output_prompt = """Make SURE to generate a JSON file following this schema and use the ```json``` code block to format it:

------------------------
"question": string
"complexity": string
"tags": list[string]
------------------------
"""
