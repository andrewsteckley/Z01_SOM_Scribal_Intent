#  Copyright (c) 2024 QuantumLynx(TM) Research.  All Rights Reserved.
import nbformat
import os
import argparse
import glob


def notebook_to_script(notebook_name, output_dir='.', script_name=None):
    """
    Converts a Jupyter Notebook to a Python script.

    :param notebook_name: Name of the notebook file (with .ipynb extension).
    :param output_dir: Directory where the Python script will be saved.
    :param script_name: Name of the output Python script (optional).
    """

    def extract_leading_spaces(text):
        # Find the first non-space character
        first_non_space = len(text) - len(text.lstrip(' '))
        if first_non_space >= 0:
            return first_non_space * ' '
        else:
            return ' ' * len(text)

    # Check if the notebook file exists
    if not os.path.exists(notebook_name):
        raise FileNotFoundError(f"The notebook {notebook_name} was not found.")

    # Load the notebook
    with open(notebook_name, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)

    # Create the Python script content
    script_content = ''
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            # Add the code cell to the script
            script_content += '# ' + '-'*20 + ' Cell ' + '-'*20 + '\n'
            for line in cell.source.splitlines():
                if line.startswith('%reload_ext') or line.startswith('%autoreload'):
                    # Comment out the line
                    script_content += '# ' + line + '\n'
                elif 'plt.show()' in line:
                    indent_str = extract_leading_spaces(line)
                    #
                    script_content += f"""
{indent_str}plt.ion()  # Interactive mode on
{line}
{indent_str}plt.pause(2)
{indent_str}plt.close()  
"""
                    # script_content += line + '\n'
                else:
                    # Add the line as is
                    script_content += line + '\n'
            script_content += '\n\n'
            script_content = script_content.replace('autoclose=False', 'autoclose=True')
            # script_content += cell.source + '\n\n'

    # Set the default script name if not provided
    if script_name is None:
        script_name = os.path.splitext(os.path.basename(notebook_name))[0] + '.py'

    # Save the script
    script_path = os.path.join(output_dir, script_name)
    with open(script_path, 'w', encoding='utf-8') as file:
        file.write(script_content)

    return script_content

def process_notebooks_with_prefix(directory, filename_prefix, output_dir, script_name):
    # Split the prefix into directory and filename prefix
    # directory, filename_prefix = os.path.split(prefix)
    # if directory == '':
    #     directory = '.'

    # Find all .ipynb files with the given prefix in the directory
    notebook_pattern = os.path.join(directory, filename_prefix + '*.ipynb')
    notebooks = sorted(glob.glob(notebook_pattern))

    # Initialize the content of the final script
    final_script_content = ''

    # Process each notebook
    for notebook in notebooks:
        print(f"Processing {notebook}...")
        basename = os.path.basename(notebook)
        final_script_content += f"# {basename}\n"
        final_script_content += 'print(80*"=")\n'
        final_script_content += 'print(80*"=")\n'
        final_script_content += f'print(f"||  Running {basename}")\n'
        final_script_content += 'print(80*"=")\n'
        final_script_content += 'print(80*"=")\n'
        final_script_content += '\n'

        final_script_content += notebook_to_script(notebook, output_dir, None) + '\n\n'

    # Save the final script
    final_script_path = os.path.join(output_dir, script_name)
    with open(final_script_path, 'w', encoding='utf-8') as file:
        file.write(final_script_content)

    return final_script_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Jupyter Notebooks with a specific prefix to a Python script.")
    parser.add_argument('notebook_dir', type=str, help="The directory of the notebook files")
    parser.add_argument('notebook_prefix', type=str, help="The prefix of the notebook files")
    parser.add_argument('--output_dir', type=str, default='.', help="Directory to save the Python script")
    parser.add_argument('--script_name', type=str, default='combined_script.py', help="Name of the output Python script")

    args = parser.parse_args()

    # Process the notebooks with the given prefix
    script_path = process_notebooks_with_prefix(args.notebook_dir, args.notebook_prefix, args.output_dir, args.script_name)
    print(f"Python script created at: {script_path}")