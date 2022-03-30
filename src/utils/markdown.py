def markdown_table_str(input_list: list) -> str:

    """Input: Python list with rows of table as lists
               First element as header.
        Output: String to put into a .md file

    Ex Input:
        [["Name", "Age", "Height"],
         ["Jake", 20, 5'10],
         ["Mary", 21, 5'7]]
    """

    markdown = "\n" + str("| ")

    for e in input_list[0]:
        to_add = " " + str(e) + str(" |")
        markdown += to_add
    markdown += "\n"

    markdown += "|"
    for i in range(len(input_list[0])):
        markdown += str("-------------- | ")
    markdown += "\n"

    for entry in input_list[1:]:
        markdown += str("| ")
        for e in entry:
            to_add = str(e) + str(" | ")
            markdown += to_add
        markdown += "\n"

    return markdown + "\n"
