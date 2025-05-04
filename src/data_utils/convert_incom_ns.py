import os
import h5py
import numpy as np


def convert(input_folder, output_folder):
    files = os.listdir(input_folder)

    os.makedirs(output_folder, exist_ok=True)

    for file in files:
        if not file.endswith(".h5"):
            continue

        path = os.path.join(input_folder, file)
        with h5py.File(path, "r") as f:
            d = dict()
            for key in f.keys():
                if key in ["velocity", "particles"]:
                    d[key] = f[key][:, :, ::4, ::4]
                elif key in ["force"]:
                    d[key] = f[key][:, ::4, ::4]
                else:
                    d[key] = f[key][()]

        new_path = os.path.join(output_folder, file)
        with h5py.File(new_path, "w") as f:
            for key in d.keys():
                f.create_dataset(key, data=d[key], compression="gzip")

        print(f"Converted source ({file}) into target ({new_path})\n\n")

        # os.remove(path)


if __name__ == "__main__":
    input_folder = "/data/shared/dataset/pdebench/2D/NS_incom"
    output_folder = "/data/shared/dataset/pdebench/2D/NS_incom/converted"

    convert(input_folder, output_folder)
