"""
Rule-based decision tree null subject detector for Italian.

Two-stage architecture:
1. Binary detection: is this finite verb a referential null-subject position?
2. Person/number classification: read from Italian verb morphology.
"""
