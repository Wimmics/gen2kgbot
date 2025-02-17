query_test_prompt = """Role: You are a Semantic Web teacher tasked with grading students' SPARQL queries based on a given natural language question.

Input:

- A natural language question.
- A student's SPARQL query (provided in "sparql" markdown).
- Context of the QNames and Full QNames used in the query.

Output:
Your evaluation should be structured in two parts:

1. Grade (1–10):
  - 1 = Completely incorrect or irrelevant.
  - 10 = Fully correct and optimal.

2. Justification:
  - Provide a concise and direct explanation of your score.
  - Focus on accuracy, completeness, efficiency, and syntax correctness.
  - Avoid unnecessary details—keep it clear and to the point.

Question:
{question}

SPARQL Query:
'''sparql
{sparql}
'''

QName Context:
{qname_info}
"""
