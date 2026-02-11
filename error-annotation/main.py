
from ast_error_detection import *
import ast

from ast_error_detection import get_typology_based_code_error

if __name__ == '__main__':

    code1 ="""
for i in range(7):
    print(i+1)
"""

    code2 = """
for i in range(6):
    print(i*2)
"""
    # visualize_custom_ast_from_code(code1)
    print(get_primary_code_errors(code1, code2))
    #print(get_typology_based_code_error(code1, [code2]))