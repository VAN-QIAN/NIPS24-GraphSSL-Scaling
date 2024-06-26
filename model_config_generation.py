import json
import os

# def generate_json_files(strings, output_path):
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     for s in strings:
#         nhid, layers = map(int, s.split('_'))
#         data = {
#             "nhid": nhid,
#             "layers": layers
#         }
#         file_name = "{}.json".format(s)
#         file_path = os.path.join(output_path, file_name)
        
#         with open(file_path, 'w') as json_file:
#             json.dump(data, json_file, indent=4)
#         print("Generated {}".format(file_path))

# strings = ["192_1", "192_2", "192_3", "192_4", "192_5", 
#            "384_1", "384_2", "384_3", "384_4", "384_5"]

# output_path = 'config_layers'

# generate_json_files(strings, output_path)

def generate_json_files(strings, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for s in strings:
        num_layers, hidden_dim = map(int, s.split('_'))
        data = {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim
        }
        file_name = "{}.json".format(s)
        file_path = os.path.join(output_path, file_name)
        
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print("Generated {}".format(file_path))

strings = ["1_32", "2_32", "3_32", "4_32", "5_32", 
           "1_64", "2_64", "3_64", "4_64", "5_64", 
           "1_128", "2_128", "3_128", "4_128", "5_128", 
           "1_256", "2_256", "3_256", "4_256", "5_256"]
# ("1_32" "2_32" "3_32" "4_32" "5_32" "1_64" "2_64" "3_64" "4_64" "5_64" "1_128" "2_128" "3_128" "4_128" "5_128" "1_256" "2_256" "3_256" "4_256" "5_256")
# strings = ["1_192", "2_192", "3_192", "4_192", "5_192",
#            "1_384", "2_384", "3_384", "4_384", "5_384"]

output_path = 'config_model_GraphCL'

generate_json_files(strings, output_path)