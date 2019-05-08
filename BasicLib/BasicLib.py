"""
    Basic toolset, Made by Robbert Brand
"""

def print_table_from_flat_list(flat_list, table_row_length):
    """
    print table from a flat list. flat list is divided in rows, based on given parameters.
    :param flat_list: input list.
    :param table_row_length: print given amount flat_list items per row.
    """
    print_format = print_format_creator(flat_list[0], table_row_length)
    for row in range(0, len(flat_list), table_row_length):
        print('|', print_format.format(*flat_list[row:row + table_row_length]), '|')


def print_format_creator(type_spec, length):
    """
    returns string format spec, to print a list of items. Each item in the list should be of the same type.
    :param type_spec: data sample, to determine type from.
    :param length: amount of items to be printed.
    :return: format string.
    """
    if type(type_spec) is int:
        print_format = ''.join(['{:5.0}'] * length)
    elif type(type_spec) is float:
        print_format = ''.join([' {:0.4f} '] * length)
    else:
        print_format = ' '.join([' {} '] * length)
    return print_format
