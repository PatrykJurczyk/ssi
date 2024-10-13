import pandas as pd
import re

def load_sample_data_with_text(values_filename, attributes_filename):
    try:
        attributes_data = pd.read_csv(attributes_filename, sep=r'\s+', header=None, names=['attribute_name', 'attribute_type'])
        attribute_names = attributes_data['attribute_name'].tolist()
        is_symbolic_attribute = attributes_data['attribute_type'].apply(lambda x: x == 's').tolist()

        samples_df = pd.read_csv(values_filename, sep=r'\s+', header=None, names=attribute_names)
        
        symbolic_column = attribute_names[is_symbolic_attribute.index(True)]
        match = re.search(r'class\((.+)\)', symbolic_column)

        if match:
            class_mapping = dict(item.split('=') for item in match.group(1).split(','))
            samples_df[symbolic_column] = samples_df[symbolic_column].apply(lambda x: class_mapping.get(str(x), x))

        return samples_df, is_symbolic_attribute, attribute_names

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")


df, is_symbolic_attribute, attribute_names = load_sample_data_with_text('iris.txt', 'iris-type.txt')
print("Sample table:")
print(df)