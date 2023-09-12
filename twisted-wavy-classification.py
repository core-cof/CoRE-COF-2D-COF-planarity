import re
import numpy as np
import pandas as pd
from openbabel import openbabel as ob
from pymatgen.core import Element
from openpyxl.styles import PatternFill
import os

def read_car_file(file_path):
    ob_converter = ob.OBConversion()
    ob_converter.SetInFormat('cif')
    mol = ob.OBMol()
    ob_converter.ReadFile(mol, file_path)
    atoms = [(atom.GetAtomicNum(), np.array([atom.GetX(), atom.GetY(), atom.GetZ()])) for atom in ob.OBMolAtomIter(mol)]
    return atoms

def calculate_centroid(atoms):
    masses = np.array([Element.from_Z(atom[0]).atomic_mass for atom in atoms])
    positions = np.array([atom[1] for atom in atoms])
    return np.average(positions, axis=0, weights=masses)

def find_best_fitting_plane(atoms, centroid):
    positions = np.array([atom[1] for atom in atoms])
    masses = np.array([Element.from_Z(atom[0]).atomic_mass for atom in atoms])
    weights = masses / np.sum(masses)
    centered_positions = positions - centroid
    cov_matrix = np.dot(centered_positions.T, centered_positions * weights[:, np.newaxis])
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    return eig_vectors[:, np.argmin(eig_values)]

def calculate_angle_between_planes(plane1_normal, plane2_normal):
    dot_product = np.dot(plane1_normal, plane2_normal)
    norm_product = np.linalg.norm(plane1_normal) * np.linalg.norm(plane2_normal)
    cos_angle = dot_product / norm_product
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

def read_lattice_parameters_cif(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    lattice_parameters = {
        'a': None,
        'b': None,
        'c': None,
        'alpha': None,
        'beta': None,
        'gamma': None
    }
    for param in lattice_parameters.keys():
        match = re.search(fr"_{param}\s+([\d.]+)", content, re.IGNORECASE)
        if match:
            lattice_parameters[param] = float(match.group(1))
        else:
            raise ValueError(f"Error: Unable to parse {param} value.")
    return lattice_parameters

def calculate_vertices(a, b, c, alpha, beta, gamma):
    alpha, beta, gamma = map(np.deg2rad, [alpha, beta, gamma])
    c1 = c * np.cos(beta)
    c2 = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    c3 = np.sqrt(c ** 2 - c1 ** 2 - c2 ** 2)
    return np.array([[0, 0, 0], [a, 0, 0], [b * np.cos(gamma), b * np.sin(gamma), 0], [c1, c2, c3]])

def find_plane_through_line_and_centroid(centroid, vertices):
    O = vertices[0]
    D = vertices[1] + vertices[2] - vertices[0]
    OD = D - O
    normal = np.cross(OD, centroid - O)
    return normal / np.linalg.norm(normal)

def plot_combined(atoms, centroid, plane1_normal, plane3_normal, lattice_params):
    vertices = calculate_vertices(lattice_params['a'], lattice_params['b'], lattice_params['c'], lattice_params['alpha'], lattice_params['beta'], lattice_params['gamma'])
    positions = np.array([atom[1] for atom in atoms])
    position_vectors = positions - centroid
    dot_products_plane1 = np.dot(position_vectors, plane1_normal)
    dot_products_plane3 = np.dot(position_vectors, plane3_normal)
    distances_to_plane1 = np.abs(np.dot(position_vectors, plane1_normal)) / np.linalg.norm(plane1_normal)
    threshold = 0.2
    above_left_indices = (dot_products_plane1 > 0) & (dot_products_plane3 > 0) & (distances_to_plane1 > threshold)
    above_right_indices = (dot_products_plane1 > 0) & (dot_products_plane3 <= 0) & (distances_to_plane1 > threshold)
    below_left_indices = (dot_products_plane1 <= 0) & (dot_products_plane3 > 0) & (distances_to_plane1 > threshold)
    below_right_indices = (dot_products_plane1 <= 0) & (dot_products_plane3 <= 0) & (distances_to_plane1 > threshold)
    return np.sum(above_left_indices), np.sum(above_right_indices), np.sum(below_left_indices), np.sum(below_right_indices)

def process_cif_file(file_path):
    atoms = read_car_file(file_path)
    centroid = calculate_centroid(atoms)
    plane1_normal = np.array([0, 0, 1])
    plane2_normal = find_best_fitting_plane(atoms, centroid)
    lattice_params = read_lattice_parameters_cif(file_path)
    plane3_normal = find_plane_through_line_and_centroid(centroid, calculate_vertices(lattice_params['a'], lattice_params['b'], lattice_params['c'], lattice_params['alpha'], lattice_params['beta'], lattice_params['gamma']))
    return plot_combined(atoms, centroid, plane2_normal, plane3_normal, lattice_params)

def process_all_cif_files(directory):
    results = {}
    for file in os.listdir(directory):
        if file.endswith('.cif'):
            file_path = os.path.join(directory, file)
            atom_counts = process_cif_file(file_path)
            results[file] = atom_counts
    return results


def export_to_excel(results, output_file):
    df = pd.DataFrame(results, index=['Above Left', 'Above Right', 'Below Left', 'Below Right']).T
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1')


def main():
    directory = r'C:\Users\ding\Desktop\1'
    results = process_all_cif_files(directory)
    output_file = os.path.join(directory, r'C:\Users\ding\Desktop\1\.xlsx')
    export_to_excel(results, output_file)

if __name__ == '__main__':
    main()

