import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--comp', required=True, default="")
parser.add_argument('-s', '--sub_comp', default="")

args = parser.parse_args()
comp = args.comp
sub_comp = args.sub_comp
folder_path = os.getcwd() + '\gdb'
img_files = os.listdir(folder_path)
folder_path = folder_path + '\\'

comp = comp.lower()
comp = comp.replace(" ", "_")
sub_comp = sub_comp.lower()
sub_comp = sub_comp.replace(" ", "_")
comp_exists = False
if len(comp)> 1:
    comp_exists = True
sub_comp_exists = False
if len(sub_comp)> 1:
    sub_comp_exists = True
req_file = ''
if comp_exists:
    for file in img_files:
        if sub_comp_exists and '__' in file:
            if comp in file.split("__")[0] and sub_comp in file.split("__")[1]:
                req_file = folder_path + file
                #print("1")
        elif comp in file and '__' not in file and not sub_comp_exists:
            req_file = folder_path + file
            #print("2")
    if len(req_file) > 1:
        req_file = req_file.replace("\\", "/")
        print(req_file)
        #print("3")
    else:
        req_file = folder_path + 'invalid_input.JPG'
        req_file = req_file.replace("\\", "/")
        print(req_file)
        #print("4")
else:
    req_file = folder_path + 'invalid_input.JPG'
    req_file = req_file.replace("\\", "/")
    print(req_file)
    #print("5")