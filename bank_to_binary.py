def format_to_binary(number, width=20):
    """Convert a number to binary with a fixed width for alignment."""
    return format(int(number), f'0{width}b')

def extract_and_format_numbers(file_path, width=20):
    """Extract the first two numbers in each line, convert them to binary, and output aligned results."""
    with open(file_path, 'r') as file:
        prev_first = None
        prev_second = None
        for line in file:
            # Check if the line starts with a number
            if line.strip() and line.split()[0].replace('/', '').isdigit():
                # Split and extract the first two numbers
                parts = line.split()
                first_number, second_number = parts[0].split('/')

                # Convert to binary with fixed width for alignment
                first_binary = format_to_binary(first_number, width)
                second_binary = format_to_binary(second_number, width)

                # If first numbers match, show XOR of second numbers
                if prev_first == first_number and prev_second is not None:
                    xor = int(second_number) ^ int(prev_second)
                    xor_binary = format_to_binary(xor, width)
                    if xor_binary == '00000000000000000010':
                        continue
                    print(f"{first_binary}/{second_binary}")
                else:
                    print(f"{first_binary}/{second_binary}")

                prev_first = first_number
                prev_second = second_number

# Usage example
extract_and_format_numbers("bankdata_workingpart.txt")