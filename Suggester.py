
import csv

def Suggesters(input):
 # Take user input
 user_input = input

 # Open the CSV file
 with open('Remedies.csv', 'r') as file:
    # Create a CSV reader
    reader = csv.reader(file)
    rows = list(reader)
    
    # Search for the input value and retrieve the corresponding output
    output_value = None
    for row in rows:
        if row[0] == user_input:
            # Each cell is Formated on the Particular Row
            formatted_row = [cell.replace('', '') for cell in row]
            # Each cell is separated by \n\n\n
            output_value = '\n\n\n'.join(formatted_row)
            break

    return output_value

